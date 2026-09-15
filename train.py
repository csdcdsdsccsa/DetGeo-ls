# -*- coding: utf8 -*-

import os
import sys
import argparse
import csv
import time
import random
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2

from torch.autograd import Variable
from torch.utils.data import DataLoader, get_worker_info
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader import RSDataset
from dataset.trogeo_loader import TROGeoRSDataset
from model.DetGeo import DetGeo
from dataset.sam_prompt_loader import SAMPromptDataset
from model.DetGeo_sam_prompt import DetGeoSAMPrompt
from dataset.gaussian_prompt_loader import GaussianPromptDataset
from model.DetGeo_gaussian import DetGeoGaussian
from dataset.adaptive_sam_prompt_loader import AdaptiveSAMPromptDataset
from model.DetGeo_adaptive_sam import DetGeoAdaptiveSAM
from dataset.sam_multimask_loader import SAMMultiMaskDataset
from model.DetGeo_prompt_interaction import DetGeoPromptInteraction
from model.DetGeo_hisym_pae import DetGeoHiSymPAE
from model.DetGeo_hisym_pae_inline import DetGeoHiSymPAEInline
from dataset.adaptive_gaussian_field_loader import AdaptiveGaussianFieldDataset
from model.DetGeo_adaptive_gaussian_field import DetGeoAdaptiveGaussianField
from model.DetGeo_b import DetGeoB
from model.DetGeo_backbone_ablation import DetGeoBackboneAblation
from model.DetGeo_single_scale_ca import DetGeoSingleScaleCA
from model.TROGeo_wo_ost import TROGeoWoOST
from model.TROGeo_ms_direct_ca_sh import TROGeoMSDirectCASH
from model.TROGeo_ms_detection_ablation import TROGeoMSDetectionAblation
from model.loss import yolo_loss, build_target, adjust_learning_rate
from model.multiscale_detection_loss import (multigrid_yolo_loss, two_head_yolo_loss, three_head_yolo_loss,
                                             three_head_yolo_loss_stage2_cls_half, coarse_heatmap_loss,
                                             rccd_consensus_loss, two_head_yolo_threshold_reg_loss)
from utils.utils import AverageMeter, eval_iou_acc, bbox_iou
from utils.multiscale_detection import (decode_multigrid_top1, decode_top1, select_two_heads, select_two_heads_hqs,
                                        select_two_heads_hqs_v2a, select_three_heads,
                                        eval_decoded_boxes, analyze_two_head_oracle, build_qcc_state,
                                        select_qcc_a, select_qcc_af, select_qcc_ranked)
from utils.checkpoint import save_checkpoint, load_pretrain


def seed_global_rng(seed):
    """Seed process-level RNGs for a reproducible model/runtime state."""
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)


def seed_worker(worker_id):
    """Synchronize all per-worker augmentation RNGs from the loader generator."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    cv2.setRNGSeed(int(worker_seed % (2 ** 31 - 1)))

    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if hasattr(dataset, 'rs_transform') and hasattr(dataset.rs_transform, 'set_random_seed'):
        dataset.rs_transform.set_random_seed(worker_seed)
    if hasattr(dataset, 'myaugment'):
        transform = getattr(dataset.myaugment, 'transform', None)
        if transform is not None and hasattr(transform, 'set_random_seed'):
            transform.set_random_seed((worker_seed + 1) % (2 ** 32))


def main():
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--prompt_lr', default=1e-4, type=float, help='learning rate for the newly initialized PromptFusion')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='./data', help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial', type=str, help='CVOGL_DroneAerial/CVOGL_SVI')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume model, optimizer and epoch from a training checkpoint')
    parser.add_argument('--resume_best_accu', default=None, type=float,
                        help='best validation Acc@0.50 before resume; required to preserve historical best selection')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--loader_seed', default=None, type=int, help='independent seed for DataLoader shuffle and worker seeds')
    parser.add_argument('--runtime_seed', default=None, type=int, help='reset global training RNG after model construction')
    parser.add_argument('--rng_probe', action='store_true', help='print first batches to verify matched data/augmentation trajectories')
    parser.add_argument('--original_rng_matched', action='store_true', help='P10: retain original DataLoader RNG behavior and isolate only SAM PromptFusion initialization RNG')
    parser.add_argument('--standard_rng', action='store_true', help='ordinary RNG: seed once, no module isolation, loader generator, worker seeding, or post-model reset')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--sam_prompt', action='store_true', help='use offline SAM/Gaussian prompt residual with the original YOLO head')
    parser.add_argument('--gaussian_only', action='store_true', help='replace the square click map with Gaussian encoding only; no SAM or PromptFusion')
    parser.add_argument('--adaptive_sam_prompt', action='store_true', help='use adaptive multi-mask SAM-Gaussian positional encoding')
    parser.add_argument('--sam_refined_pe', action='store_true', help='refine original DetGeo P0 with offline SAM multi-mask candidates')
    parser.add_argument('--rgbp_interaction', action='store_true', help='enable zero-residual bidirectional RGB-position interaction')
    parser.add_argument('--hisym_pae', action='store_true', help='use HiSymGeo-style residual Conv3x3 RGB-position fusion')
    parser.add_argument('--hisym_pae_inline_init', action='store_true', help='standalone P0+PAE with PAE initialized at the original click-fusion slot')
    parser.add_argument('--adaptive_gaussian_field', action='store_true',
                        help='content-adaptive Gaussian-only position field; no SAM or RGB PAE')
    parser.add_argument('--gaussian_field_mode', choices=('msg', 'msg_ag', 'full', 'g25_ag', 'g25_cc', 'g25_ag_cc'), default='msg',
                        help='adaptive Gaussian field: multi-scale, anisotropic, or core/context')
    parser.add_argument('--gaussian_bank', default='12,20,25,35,50',
                        help='comma-separated feature-map-scale sigma bank; must include --gaussian_sigma')
    parser.add_argument('--gaussian_context_scale', default=2.0, type=float,
                        help='wide/core sigma ratio for Gaussian field full mode')
    parser.add_argument('--gaussian_gamma_init', default=0.05, type=float,
                        help='initial adaptive-field residual gate value in (0,1)')
    parser.add_argument('--gaussian_beta_init', default=0.05, type=float,
                        help='initial full-mode context gate value in (0,1)')
    parser.add_argument('--b_variant', choices=('none', 'msst', 'core', 'b1', 'b2'), default='none',
                        help='original-P0 B framework variant; none retains the unmodified DetGeo baseline')
    parser.add_argument('--b_num_tokens', type=int, default=8, help='shared MSST token count')
    parser.add_argument('--b_num_heads', type=int, default=8, help='attention head count for DetGeoB')
    parser.add_argument('--b_ffn_dim', type=int, default=1024, help='attention FFN width for DetGeoB')
    parser.add_argument('--backbone_exp', choices=('baseline', 'darknet53_noshare', 'darknet53_shared', 'resnet50_shared'),
                        default='baseline', help='pure original-P0 DetGeo backbone ablation')
    parser.add_argument('--single_scale_ca', action='store_true',
                        help='replace only original QACVFM with one spatial multi-head cross-attention block')
    parser.add_argument('--trogeo_wo_ost', action='store_true',
                        help='standalone TROGeo-style shared Swin-S + CVOPM detection reproduction without OST')
    parser.add_argument('--trogeo_direct_ca', action='store_true',
                        help='TROGeo w/o OST with satellite self-attention removed; direct satellite-query cross-attention only')
    parser.add_argument('--trogeo_ms_direct_ca_sh', action='store_true',
                        help='Swin-T stage3/stage4 Direct-CA with separate 6/3-anchor heads and no feature fusion')
    parser.add_argument('--trogeo_ms_det_variant', choices=(
        'none', 'correct63', 'b_multigrid', 'h2_shared', 'h2_ind', 'h3_ind', 'h3_adaptive',
        'h2_ind_3scale', 'h2_ind_3scale_stage2cls05', 'h2_ind_3scale_pe_ln_amp',
        'h2_ind_3scale_pe_all_add', 'h2_ind_3scale_pe_all_key', 'h2_ind_3scale_le_ind_res',
        'h2_ind_3scale_le_stage2_res',
        'h2_ind_pe_ln_amp_kv', 'h2_ind_pe_ln_amp_key', 'h2_ind_pe_all_add',
        'h2_ind_pe_all_key', 'h2_ind_le_stage2_res',
        'h2_ind_pgca_c_direct', 'h2_ind_pgca_a_conv', 'h2_ind_pgca_b_dynamic',
        'h2_ind_csfi', 'h2_ind_cg', 'h2_ind_csfi_cg', 'h2_ind_hier',
        'h2_ind_cg_habr_prior',
        'h2_ind_csfi_fg', 'h2_ind_csfi_bi', 'h2_ind_mhcsfi_bi', 'h2_ind_amhcsfi_bi', 'h2_ind_amhcsfi_res_bi',
        'h2_ind_amhcsfi_res_bi_afuse',
        'h2_ind_amhcsfi_res_bi_qcc_a', 'h2_ind_amhcsfi_res_bi_qcc_af', 'h2_ind_amhcsfi_res_bi_qcc_b',
        'h2_ind_amhcsfi_res_bi_qcc_full',
        'h2_ind_cg_amhcsfi_res', 'h2_ind_fg_amhcsfi_res',
        'h2_ind_fg_amhcsfi_res_s3', 'h2_ind_fg_amhcsfi_res_s4', 'h2_ind_fg_amhcsfi_res_afuse',
        'h2_ind_fg_nocsfi', 'h2_ind_bi_nocsfi',
        'h2_ind_csfi_cg_channel', 'h2_ind_csfi_cg_dir', 'h2_ind_csfi_cg_ar',
        'h2_ind_pqra',
        'h2_ind_habr_core', 'h2_ind_habr_prior', 'h2_ind_habr_adapt', 'h2_ind_habr'),
                        default='none', help='controlled Swin-T multi-scale detection ablation')
    parser.add_argument('--coarse_loss_weight', default=0.2, type=float,
                        help='weight of coarse Stage4 heatmap supervision for E4 collaboration variants')
    parser.add_argument('--coarse_sigma', default=1.5, type=float,
                        help='Gaussian sigma in cells for E4 Stage4 coarse heatmap supervision')
    parser.add_argument('--fine_sigma', default=3.0, type=float,
                        help='Gaussian sigma in Stage3 64x64 cells for fine-guidance supervision')
    parser.add_argument('--qcc_quality_weight', default=0.2, type=float,
                        help='QCC localization-quality loss weight')
    parser.add_argument('--qcc_rank_weight', default=0.1, type=float,
                        help='QCC detached pairwise-ranker loss weight')
    parser.add_argument('--qcc_rank_epsilon', default=0.03, type=float,
                        help='QCC ignore band for the two heads GT-IoU gap')
    parser.add_argument('--qcc_fusion_iou', default=0.5, type=float,
                        help='QCC-AF/QCC-Full predicted-box agreement IoU threshold')
    parser.add_argument('--bbox_threshold_reg', action='store_true',
                        help='enable Acc@0.25/0.50-oriented positive-box threshold regularization')
    parser.add_argument('--bbox_threshold_reg_weight', default=0.2, type=float,
                        help='overall threshold-regularization weight')
    parser.add_argument('--bbox_threshold_temperature', default=0.05, type=float,
                        help='soft IoU-threshold temperature')
    parser.add_argument('--bbox_threshold_weight25', default=0.5, type=float,
                        help='relative IoU=0.25 threshold weight')
    parser.add_argument('--bbox_threshold_weight50', default=1.0, type=float,
                        help='relative IoU=0.50 threshold weight')
    parser.add_argument('--rccd', action='store_true',
                        help='training-only reliability-aware confidence consensus for h2_ind_csfi_bi')
    parser.add_argument('--rccd_weight', default=0.1, type=float, help='maximum RCCD loss weight')
    parser.add_argument('--rccd_temperature', default=2.0, type=float, help='RCCD distillation temperature')
    parser.add_argument('--rccd_tau', default=0.2, type=float, help='RCCD reliability-weight temperature')
    parser.add_argument('--rccd_warmup_epochs', default=3, type=int, help='initial zero-weight RCCD epochs')
    parser.add_argument('--rccd_ramp_epochs', default=3, type=int, help='RCCD ramp epochs before full weight')
    parser.add_argument('--h3_iou_threshold', default=0.5, type=float,
                        help='H3: fuse two Top-1 boxes only when their pair IoU reaches this threshold')
    parser.add_argument('--trogeo_backbone', choices=('swin_s', 'swin_t', 'resnet50', 'vit_t', 'vit_s'), default='swin_s',
                        help='shared ImageNet backbone for TROGeo modes')
    parser.add_argument('--trogeo_aug_mode', choices=('current', 'detgeo'), default='current',
                        help='TROGeo train augmentation: current or original DetGeo RSDataset recipe')
    parser.add_argument('--trogeo_position_mode', choices=('current', 'detgeo', 'dg', 'ddg', 'rdg'), default='current',
                        help='TROGeo front-end position encoder: current / DetGeo / DG-PE / DDG-PE / RDG-PE')
    parser.add_argument('--trogeo_click_map_mode', choices=('distance', 'gaussian'), default='distance',
                        help='TROGeo click map: original distance-decay map or fixed Gaussian map')
    parser.add_argument('--dadpe_mode', choices=('none', 'input', 'multiscale'), default='none',
                        help='direction-aware DetGeo PE: none=A, input=B, multiscale=D')
    parser.add_argument('--amr_pe_mode', choices=('none', 'fixed', 'adaptive'), default='none',
                        help='AMR-PE front-end field: none=DetGeo PE, fixed=uniform ranges, adaptive=query ranges')
    parser.add_argument('--hqs_oracle_diag', action='store_true',
                        help='validation-only Oracle head-selection diagnostic for supported AMHCSFI-Res variants')
    parser.add_argument('--hqs_oracle_csv', default='', type=str,
                        help='optional CSV path for per-sample HQS Oracle diagnostics')
    parser.add_argument('--hqs_v1', action='store_true', help='train a frozen FG-Res Head Quality Selector')
    parser.add_argument('--hqs_lr', default=1e-3, type=float)
    parser.add_argument('--hqs_temperature', default=0.15, type=float)
    parser.add_argument('--hqs_gap_delta', default=0.30, type=float)
    parser.add_argument('--hqs_v2a', action='store_true',
                        help='HQS-v2a: frozen FG-Res with balanced pairwise head ranking')
    parser.add_argument('--hqs_v2b', action='store_true',
                        help='HQS-v2b: prediction-quality-aware pairwise head ranking')
    parser.add_argument('--hqs_rank_epsilon', default=0.03, type=float)
    parser.add_argument('--hqs_rank_threshold', default=0.0, type=float,
                        help='HQS-v2a inference threshold: rank_logit >= threshold selects Head3')
    parser.add_argument('--hqs_v2a_rank_diag', action='store_true',
                        help='validation-only HQS-v2a rank separability and threshold-sweep diagnostic')
    parser.add_argument('--hqs_v2a_rank_diag_prefix', default='results/hqs_v2a_rank_diag', type=str,
                        help='output prefix for HQS-v2a rank diagnostic CSV files')
    parser.add_argument('--hqs_v2b_rank_diag', action='store_true',
                        help='validation-only HQS-v2b rank separability and threshold-sweep diagnostic')
    parser.add_argument('--sam_mask_root', default='', help='optional root of split-indexed SAM masks')
    parser.add_argument('--sam_multimask_root', default='', help='optional root of split-indexed SAM multi-mask npz files')
    parser.add_argument('--gaussian_sigma', default=25.0, type=float, help='Gaussian click sigma at the query feature-map scale')
    parser.add_argument('--sigma_min', default=8.0, type=float)
    parser.add_argument('--sigma_max', default=50.0, type=float)
    parser.add_argument('--freeze_prompt_only', action='store_true', help='train only the zero-init prompt_fusion module')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    
    global args, anchors_full
    args = parser.parse_args()
    if args.pretrain and args.resume:
        parser.error('--pretrain and --resume are mutually exclusive')
    if args.rccd and args.trogeo_ms_det_variant != 'h2_ind_csfi_bi':
        parser.error('--rccd is restricted to --trogeo_ms_det_variant h2_ind_csfi_bi')
    if args.rccd_weight < 0.0 or args.rccd_temperature <= 0.0 or args.rccd_tau <= 0.0 or \
            args.rccd_warmup_epochs < 0 or args.rccd_ramp_epochs < 0:
        parser.error('RCCD requires nonnegative weight/epochs and positive temperatures')
    if not 0.0 <= args.qcc_fusion_iou <= 1.0:
        parser.error('--qcc_fusion_iou must be in [0,1]')
    if args.loader_seed is None:
        args.loader_seed = args.seed
    if args.runtime_seed is None:
        args.runtime_seed = args.seed
    try:
        args.gaussian_bank_values = tuple(float(value.strip()) for value in args.gaussian_bank.split(',') if value.strip())
    except ValueError:
        parser.error('--gaussian_bank must be a comma-separated list of numbers')
    if not args.gaussian_bank_values or args.gaussian_sigma not in args.gaussian_bank_values:
        parser.error('--gaussian_bank must include --gaussian_sigma')
    if not (0.0 < args.gaussian_gamma_init < 1.0 and 0.0 < args.gaussian_beta_init < 1.0):
        parser.error('--gaussian_gamma_init and --gaussian_beta_init must be in (0,1)')
    if sum((args.sam_prompt, args.gaussian_only, args.adaptive_sam_prompt, args.sam_refined_pe)) > 1:
        parser.error('only one positional-encoding mode can be selected')
    if args.rgbp_interaction and (args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt):
        parser.error('--rgbp_interaction supports only original P0 or --sam_refined_pe')
    if args.hisym_pae and (args.rgbp_interaction or args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt):
        parser.error('--hisym_pae supports only original P0 or --sam_refined_pe')
    if args.standard_rng and args.original_rng_matched:
        parser.error('--standard_rng and --original_rng_matched are mutually exclusive')
    if args.hisym_pae_inline_init and (args.hisym_pae or args.sam_refined_pe or args.rgbp_interaction
                                       or args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt):
        parser.error('--hisym_pae_inline_init is a standalone P0+PAE confirmation experiment')
    if args.hisym_pae_inline_init and not args.standard_rng:
        parser.error('--hisym_pae_inline_init must use --standard_rng')
    if args.adaptive_gaussian_field and (args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt
                                         or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae
                                         or args.hisym_pae_inline_init):
        parser.error('--adaptive_gaussian_field is a standalone Gaussian-only experiment')
    if args.adaptive_gaussian_field and not args.standard_rng:
        parser.error('--adaptive_gaussian_field must use --standard_rng (P08-style ordinary RNG)')
    if args.b_variant != 'none' and (args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt
                                     or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae
                                     or args.hisym_pae_inline_init or args.adaptive_gaussian_field):
        parser.error('--b_variant must use original DetGeo positional encoding only')
    if args.backbone_exp != 'baseline' and (args.b_variant != 'none' or args.sam_prompt or args.gaussian_only
                                             or args.adaptive_sam_prompt or args.sam_refined_pe or args.rgbp_interaction
                                             or args.hisym_pae or args.hisym_pae_inline_init or args.adaptive_gaussian_field):
        parser.error('--backbone_exp must use only original DetGeo square positional encoding')
    if args.single_scale_ca and (args.backbone_exp != 'baseline' or args.b_variant != 'none'
                                 or args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt
                                 or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae
                                 or args.hisym_pae_inline_init or args.adaptive_gaussian_field
                                 or args.freeze_prompt_only):
        parser.error('--single_scale_ca is a standalone original-DetGeo square-position experiment')
    if args.single_scale_ca and not args.standard_rng:
        parser.error('--single_scale_ca must use --standard_rng (ordinary RNG protocol)')
    trogeo_experiments = (args.trogeo_wo_ost, args.trogeo_direct_ca, args.trogeo_ms_direct_ca_sh,
                          args.trogeo_ms_det_variant != 'none')
    if sum(trogeo_experiments) > 1:
        parser.error('TROGeo experiment modes are mutually exclusive')
    trogeo_mode = any(trogeo_experiments)
    if args.trogeo_ms_direct_ca_sh and args.trogeo_backbone != 'swin_t':
        parser.error('--trogeo_ms_direct_ca_sh currently requires --trogeo_backbone swin_t')
    if args.trogeo_ms_det_variant != 'none' and args.trogeo_backbone not in ('swin_t', 'vit_t', 'vit_s'):
        parser.error('--trogeo_ms_det_variant requires --trogeo_backbone swin_t/vit_t/vit_s')
    if args.trogeo_backbone in ('vit_t', 'vit_s') and args.trogeo_ms_det_variant != 'h2_ind':
        parser.error('ViT backbones are restricted to the strict two-scale E4 h2_ind experiment')
    if args.dadpe_mode != 'none':
        if args.trogeo_ms_det_variant not in ('h2_ind_csfi_bi', 'h2_ind_fg_amhcsfi_res',
                                               'h2_ind_amhcsfi_res_bi'):
            parser.error('--dadpe_mode is restricted to A3-Bi, FG-Res, or Bi-Res')
        if args.trogeo_position_mode != 'detgeo':
            parser.error('--dadpe_mode requires --trogeo_position_mode detgeo')
        if args.trogeo_click_map_mode != 'distance':
            parser.error('--dadpe_mode requires --trogeo_click_map_mode distance')
        if args.trogeo_backbone != 'swin_t':
            parser.error('--dadpe_mode currently requires --trogeo_backbone swin_t')
    if args.amr_pe_mode != 'none':
        if args.trogeo_ms_det_variant != 'h2_ind_amhcsfi_res_bi':
            parser.error('--amr_pe_mode is restricted to Bi-Res h2_ind_amhcsfi_res_bi')
        if args.trogeo_position_mode != 'detgeo':
            parser.error('--amr_pe_mode requires --trogeo_position_mode detgeo')
        if args.trogeo_click_map_mode != 'distance':
            parser.error('--amr_pe_mode requires --trogeo_click_map_mode distance')
        if args.trogeo_backbone != 'swin_t':
            parser.error('--amr_pe_mode currently requires --trogeo_backbone swin_t')
        if args.dadpe_mode != 'none':
            parser.error('--amr_pe_mode requires --dadpe_mode none for a front-end-only ablation')
    if args.trogeo_position_mode in ('dg', 'ddg', 'rdg'):
        if args.trogeo_ms_det_variant != 'h2_ind_amhcsfi_res_bi' or args.trogeo_backbone != 'swin_t':
            parser.error('DG/DDG/RDG-PE requires Bi-Res h2_ind_amhcsfi_res_bi with Swin-T')
        if args.trogeo_click_map_mode != 'gaussian' or args.gaussian_sigma <= 0.0:
            parser.error('DG/DDG/RDG-PE requires --trogeo_click_map_mode gaussian and positive --gaussian_sigma')
        if args.dadpe_mode != 'none' or args.amr_pe_mode != 'none':
            parser.error('DG/DDG/RDG-PE requires --dadpe_mode none and --amr_pe_mode none')
    if args.bbox_threshold_reg:
        if args.trogeo_ms_det_variant != 'h2_ind_amhcsfi_res_bi':
            parser.error('--bbox_threshold_reg is restricted to Bi-Res h2_ind_amhcsfi_res_bi')
        if args.trogeo_backbone != 'swin_t' or args.trogeo_position_mode != 'detgeo':
            parser.error('--bbox_threshold_reg requires Swin-T and original DetGeo PE')
        if args.trogeo_click_map_mode != 'distance' or args.dadpe_mode != 'none' or args.amr_pe_mode != 'none':
            parser.error('--bbox_threshold_reg requires distance map, dadpe_mode=none, and amr_pe_mode=none')
        if args.rccd:
            parser.error('--bbox_threshold_reg must not be combined with RCCD')
        if args.bbox_threshold_reg_weight < 0.0 or args.bbox_threshold_weight25 < 0.0 or args.bbox_threshold_weight50 < 0.0:
            parser.error('threshold-regularization weights must be >= 0')
        if args.bbox_threshold_temperature <= 0.0:
            parser.error('--bbox_threshold_temperature must be > 0')
    if args.hqs_oracle_diag:
        if args.trogeo_ms_det_variant not in ('h2_ind_fg_amhcsfi_res', 'h2_ind_amhcsfi_res_bi'):
            parser.error('--hqs_oracle_diag is restricted to '
                         '--trogeo_ms_det_variant h2_ind_fg_amhcsfi_res or h2_ind_amhcsfi_res_bi')
        if not args.val or args.test:
            parser.error('--hqs_oracle_diag is validation-only; use --val and do not use --test')
        if args.rccd:
            parser.error('--hqs_oracle_diag must not be combined with RCCD')
    if args.hqs_v1:
        if args.trogeo_ms_det_variant != 'h2_ind_fg_amhcsfi_res':
            parser.error('--hqs_v1 requires --trogeo_ms_det_variant h2_ind_fg_amhcsfi_res')
        if args.rccd or args.hqs_oracle_diag:
            parser.error('--hqs_v1 cannot be combined with RCCD or --hqs_oracle_diag')
        if args.hqs_temperature <= 0 or args.hqs_gap_delta <= 0:
            parser.error('--hqs_temperature and --hqs_gap_delta must be positive')
        if not (args.test or args.val) and not (args.pretrain or args.resume):
            parser.error('HQS-v1 training requires the completed FG-Res checkpoint')
    if sum((args.hqs_v1, args.hqs_v2a, args.hqs_v2b)) > 1:
        parser.error('--hqs_v1, --hqs_v2a and --hqs_v2b are mutually exclusive')
    if args.hqs_v2a:
        if args.trogeo_ms_det_variant != 'h2_ind_fg_amhcsfi_res':
            parser.error('--hqs_v2a requires --trogeo_ms_det_variant h2_ind_fg_amhcsfi_res')
        if args.rccd or args.hqs_oracle_diag:
            parser.error('--hqs_v2a cannot be combined with RCCD or --hqs_oracle_diag')
        if not 0.0 <= args.hqs_rank_epsilon < 1.0:
            parser.error('--hqs_rank_epsilon must be in [0,1)')
        if not (args.test or args.val) and not (args.pretrain or args.resume):
            parser.error('HQS-v2a training requires the completed FG-Res checkpoint')
    if args.hqs_v2a_rank_diag:
        if not args.hqs_v2a:
            parser.error('--hqs_v2a_rank_diag requires --hqs_v2a')
        if args.trogeo_ms_det_variant != 'h2_ind_fg_amhcsfi_res':
            parser.error('--hqs_v2a_rank_diag requires --trogeo_ms_det_variant h2_ind_fg_amhcsfi_res')
        if not args.val or args.test:
            parser.error('--hqs_v2a_rank_diag is validation-only; use --val and do not use --test')
        if args.hqs_oracle_diag:
            parser.error('--hqs_v2a_rank_diag and --hqs_oracle_diag are separate diagnostics')
    if args.hqs_v2b:
        if args.trogeo_ms_det_variant != 'h2_ind_fg_amhcsfi_res':
            parser.error('--hqs_v2b requires --trogeo_ms_det_variant h2_ind_fg_amhcsfi_res')
        if args.rccd or args.hqs_oracle_diag:
            parser.error('--hqs_v2b cannot be combined with RCCD or --hqs_oracle_diag')
        if not 0.0 <= args.hqs_rank_epsilon < 1.0:
            parser.error('--hqs_rank_epsilon must be in [0,1)')
        if not (args.test or args.val) and not (args.pretrain or args.resume):
            parser.error('HQS-v2b training requires the completed FG-Res checkpoint')
    if args.hqs_v2b_rank_diag:
        if not args.hqs_v2b:
            parser.error('--hqs_v2b_rank_diag requires --hqs_v2b')
        if not args.val or args.test:
            parser.error('--hqs_v2b_rank_diag is validation-only; use --val and do not use --test')
        if args.hqs_oracle_diag:
            parser.error('--hqs_v2b_rank_diag and --hqs_oracle_diag are separate diagnostics')
    if args.trogeo_ms_det_variant in ('h3_ind', 'h3_adaptive') and not (args.test or args.val):
        parser.error('H3 variants are inference-only: train h2_ind then evaluate its best checkpoint')
    if not 0.0 <= args.h3_iou_threshold <= 1.0:
        parser.error('--h3_iou_threshold must be in [0, 1]')
    if not trogeo_mode and args.trogeo_backbone != 'swin_s':
        parser.error('--trogeo_backbone is only valid for a TROGeo mode')
    if not trogeo_mode and (args.trogeo_aug_mode != 'current' or args.trogeo_position_mode != 'current'
                            or args.trogeo_click_map_mode != 'distance'):
        parser.error('--trogeo_aug_mode, --trogeo_position_mode and --trogeo_click_map_mode are only valid for TROGeo modes')
    if trogeo_mode and (args.backbone_exp != 'baseline' or args.single_scale_ca or args.b_variant != 'none'
                               or args.sam_prompt or args.gaussian_only or args.adaptive_sam_prompt
                               or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae
                               or args.hisym_pae_inline_init or args.adaptive_gaussian_field
                               or args.freeze_prompt_only):
        parser.error('TROGeo modes are standalone reproduction experiments')
    if trogeo_mode and not args.standard_rng:
        parser.error('TROGeo modes must use --standard_rng (ordinary RNG protocol)')
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed_global_rng(args.seed)

    eps=1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=1024
    if args.data_name == 'CVOGL_DroneAerial':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    elif args.data_name == 'CVOGL_SVI':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    else:
        assert(False)
    args.anchors = anchors

    ## save logs
    if args.savename=='default':
        args.savename = '%s_batch%d' % (args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    if trogeo_mode:
        dataset_class = TROGeoRSDataset
        prompt_kwargs = {'click_map_mode': args.trogeo_click_map_mode, 'gaussian_sigma': args.gaussian_sigma}
    elif args.backbone_exp != 'baseline':
        # Backbone ablations retain the original square-position RSDataset.
        dataset_class = RSDataset
        prompt_kwargs = {}
    elif args.b_variant != 'none':
        dataset_class = RSDataset
        prompt_kwargs = {}
    elif args.adaptive_gaussian_field:
        dataset_class = AdaptiveGaussianFieldDataset
        prompt_kwargs = {}
    elif args.sam_refined_pe:
        dataset_class = SAMMultiMaskDataset
        prompt_kwargs = {'sam_multimask_root': args.sam_multimask_root or None}
    elif args.adaptive_sam_prompt:
        dataset_class = AdaptiveSAMPromptDataset
        prompt_kwargs = {'sam_multimask_root': args.sam_multimask_root or None}
    elif args.sam_prompt:
        dataset_class = SAMPromptDataset
        prompt_kwargs = {'sam_mask_root': args.sam_mask_root or None, 'gaussian_sigma': args.gaussian_sigma}
    elif args.gaussian_only:
        dataset_class = GaussianPromptDataset
        prompt_kwargs = {'gaussian_sigma': args.gaussian_sigma}
    else:
        dataset_class = RSDataset
        prompt_kwargs = {}
    train_dataset = dataset_class(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='train',
                         img_size=args.img_size,
                         transform=input_transform,
                         augment=True,
                         **({'aug_mode': args.trogeo_aug_mode} if trogeo_mode else {}), **prompt_kwargs)
    val_dataset = dataset_class(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='val',
                         img_size = args.img_size,
                         transform=input_transform,
                         **({'aug_mode': args.trogeo_aug_mode} if trogeo_mode else {}), **prompt_kwargs)
    test_dataset = dataset_class(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='test',
                         img_size = args.img_size,
                         transform=input_transform,
                         **({'aug_mode': args.trogeo_aug_mode} if trogeo_mode else {}), **prompt_kwargs)
    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=True,
                         drop_last=False, num_workers=args.num_workers)
    if args.original_rng_matched or args.standard_rng:
        # P10 keeps its model-init RNG isolation; standard mode deliberately does not.
        # Both retain DetGeo's ordinary DataLoader construction.
        # P10 deliberately reproduces DetGeo's original default DataLoader RNG.
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        eval_loader_kwargs = dict(loader_kwargs, batch_size=args.batch_size * 2) if trogeo_mode else loader_kwargs
        val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)
    else:
        train_generator = torch.Generator()
        train_generator.manual_seed(args.loader_seed)
        val_generator = torch.Generator()
        val_generator.manual_seed(args.loader_seed + 1)
        test_generator = torch.Generator()
        test_generator.manual_seed(args.loader_seed + 2)
        train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator,
                                  worker_init_fn=seed_worker, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, generator=val_generator,
                                worker_init_fn=seed_worker, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, generator=test_generator,
                                 worker_init_fn=seed_worker, **loader_kwargs)
    
    ## Model
    if args.trogeo_wo_ost:
        model = TROGeoWoOST(emb_size=args.emb_size, use_satellite_self_attention=True,
                             backbone=args.trogeo_backbone)
    elif args.trogeo_direct_ca:
        model = TROGeoWoOST(emb_size=args.emb_size, use_satellite_self_attention=False,
                             backbone=args.trogeo_backbone)
    elif args.trogeo_ms_direct_ca_sh:
        model = TROGeoMSDirectCASH(emb_size=args.emb_size, backbone=args.trogeo_backbone)
    elif args.trogeo_ms_det_variant != 'none':
        model = TROGeoMSDetectionAblation(emb_size=args.emb_size, backbone=args.trogeo_backbone,
                                           variant=args.trogeo_ms_det_variant,
                                           position_mode=args.trogeo_position_mode,
                                           dadpe_mode=args.dadpe_mode, amr_pe_mode=args.amr_pe_mode,
                                           gaussian_sigma=args.gaussian_sigma,
                                           enable_hqs=args.hqs_v1,
                                           enable_hqs_v2a=args.hqs_v2a, enable_hqs_v2b=args.hqs_v2b)
    elif args.backbone_exp != 'baseline':
        model = DetGeoBackboneAblation(emb_size=args.emb_size, leaky=True, backbone_exp=args.backbone_exp)
    elif args.single_scale_ca:
        model = DetGeoSingleScaleCA(emb_size=args.emb_size, leaky=True)
    elif args.b_variant != 'none':
        model = DetGeoB(emb_size=args.emb_size, leaky=True, variant=args.b_variant,
                         num_tokens=args.b_num_tokens, num_heads=args.b_num_heads, ffn_dim=args.b_ffn_dim)
    elif args.adaptive_gaussian_field:
        model = DetGeoAdaptiveGaussianField(emb_size=args.emb_size, leaky=True,
                                            mode=args.gaussian_field_mode,
                                            sigma_bank=args.gaussian_bank_values,
                                            base_sigma=args.gaussian_sigma,
                                            context_scale=args.gaussian_context_scale,
                                            gamma_init=args.gaussian_gamma_init,
                                            beta_init=args.gaussian_beta_init)
    elif args.hisym_pae_inline_init:
        model = DetGeoHiSymPAEInline(emb_size=args.emb_size, leaky=True)
    elif args.hisym_pae:
        model = DetGeoHiSymPAE(use_sam_refinement=args.sam_refined_pe,
                               preserve_downstream_rng=args.original_rng_matched)
    elif args.sam_refined_pe or args.rgbp_interaction:
        model = DetGeoPromptInteraction(use_sam_refinement=args.sam_refined_pe,
                                        use_rgbp_interaction=args.rgbp_interaction,
                                        preserve_downstream_rng=args.original_rng_matched)
    elif args.adaptive_sam_prompt:
        model = DetGeoAdaptiveSAM(preserve_downstream_rng=args.original_rng_matched,
                                  base_sigma=args.gaussian_sigma, sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    elif args.sam_prompt:
        model = DetGeoSAMPrompt(preserve_downstream_rng=args.original_rng_matched)
    elif args.gaussian_only:
        model = DetGeoGaussian()
    else:
        model = DetGeo()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)

    if args.hqs_v1:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.module.head_quality_selector.parameters():
            parameter.requires_grad = True
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or not all(name.startswith('module.head_quality_selector.') for name in trainable_names):
            raise RuntimeError('HQS-v1 unexpectedly unfroze FG-Res parameters: {}'.format(trainable_names))
        print('[HQS-v1] frozen FG-Res; trainable parameters={}'.format(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)))
    if args.hqs_v2a:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.module.head_pairwise_ranker.parameters():
            parameter.requires_grad = True
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or not all(name.startswith('module.head_pairwise_ranker.') for name in trainable_names):
            raise RuntimeError('HQS-v2a unexpectedly unfroze FG-Res parameters: {}'.format(trainable_names))
        print('[HQS-v2a] frozen FG-Res; trainable parameters={}'.format(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)))
    if args.hqs_v2b:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.module.head_prediction_quality_ranker.parameters():
            parameter.requires_grad = True
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or not all(name.startswith('module.head_prediction_quality_ranker.')
                                          for name in trainable_names):
            raise RuntimeError('HQS-v2b unexpectedly unfroze FG-Res parameters: {}'.format(trainable_names))
        print('[HQS-v2b] frozen FG-Res; trainable parameters={}'.format(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)))

    if args.freeze_prompt_only:
        if not (args.sam_prompt or args.adaptive_sam_prompt or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae):
            raise ValueError('--freeze_prompt_only requires a prompt mode')
        if args.sam_prompt:
            prefixes = ('module.prompt_fusion.',)
        elif args.adaptive_sam_prompt:
            prefixes = ('module.adaptive_prompt.',)
        elif args.hisym_pae:
            prefixes = tuple('module.' + p for p, active in
                             (('sam_refiner.', args.sam_refined_pe), ('pae_fusion.', True)) if active)
        else:
            prefixes = tuple('module.' + p for p, active in
                             (('sam_refiner.', args.sam_refined_pe), ('rgbp_interaction.', args.rgbp_interaction)) if active)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
        trainable = sum(parameter.nelement() for parameter in model.parameters() if parameter.requires_grad)
        print('Frozen original DetGeo; trainable PromptFusion parameters:', trainable)
    
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    if args.hqs_v1:
        optimizer = torch.optim.Adam(model.module.head_quality_selector.parameters(), lr=args.hqs_lr, betas=(0.9, 0.999))
    elif args.hqs_v2a:
        optimizer = torch.optim.Adam(model.module.head_pairwise_ranker.parameters(), lr=args.hqs_lr, betas=(0.9, 0.999))
    elif args.hqs_v2b:
        optimizer = torch.optim.Adam(model.module.head_prediction_quality_ranker.parameters(), lr=args.hqs_lr,
                                     betas=(0.9, 0.999))
    elif trogeo_mode:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.sam_prompt or args.adaptive_sam_prompt or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae or args.adaptive_gaussian_field:
        prompt_params, base_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if args.adaptive_gaussian_field:
                is_new = 'gaussian_field.' in name
            elif args.sam_prompt:
                is_new = 'prompt_fusion.' in name
            elif args.adaptive_sam_prompt:
                is_new = 'adaptive_prompt.' in name
            elif args.hisym_pae:
                is_new = 'sam_refiner.' in name or 'pae_fusion.' in name
            else:
                is_new = 'sam_refiner.' in name or 'rgbp_interaction.' in name
            (prompt_params if is_new else base_params).append(parameter)
        optimizer_groups = []
        if base_params:
            optimizer_groups.append({'params': base_params, 'lr': args.lr, 'base_lr': args.lr})
        if prompt_params:
            optimizer_groups.append({'params': prompt_params, 'lr': args.prompt_lr, 'base_lr': args.prompt_lr})
    else:
        optimizer_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr, 'base_lr': args.lr}]
    if not trogeo_mode:
        optimizer = torch.optim.RMSprop(optimizer_groups, weight_decay=0.0005)

    if not args.original_rng_matched and not args.standard_rng:
        # P09: SAM creates extra PromptFusion parameters. Reset runtime RNG
        # after model/optimizer construction so later training randomness matches.
        seed_global_rng(args.runtime_seed)
    
    ## training and testing
    start_epoch = 0
    best_accu = -float('Inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint['epoch'])
        best_accu = float(args.resume_best_accu) if args.resume_best_accu is not None else float(checkpoint['best_loss'])
        message = '=> resumed checkpoint {} at epoch {}, retained best Accu {:.6f}'.format(args.resume, start_epoch, best_accu)
        print(message)
        logging.info(message)
    if args.rccd and not (args.test or args.val):
        message = ('[RCCD config] enabled=True variant=h2_ind_csfi_bi temperature={} tau={} max_weight={} '
                   'warmup_epochs={} ramp_epochs={} bbox_distillation=False inference_rccd=False').format(
                       args.rccd_temperature, args.rccd_tau, args.rccd_weight,
                       args.rccd_warmup_epochs, args.rccd_ramp_epochs)
        print(message, flush=True)
        logging.info(message)
    
    if args.test:
        _ = test_epoch(test_loader, model, args)
    elif args.val:
        _ = test_epoch(val_loader, model, args)
    else:
        for epoch in range(start_epoch, args.max_epoch):
            gc.collect()
            if args.hqs_v2b:
                train_hqs_v2b_epoch(train_loader, model, optimizer, epoch, args)
            elif args.hqs_v2a:
                train_hqs_v2a_epoch(train_loader, model, optimizer, epoch, args)
            elif args.hqs_v1:
                train_hqs_epoch(train_loader, model, optimizer, epoch, args)
            else:
                adjust_learning_rate(args, optimizer, epoch)
                train_epoch(train_loader, model, optimizer, epoch, args)
            accu_new = test_epoch(val_loader, model, args)
            ## remember best accu and save checkpoint
            is_best = accu_new > best_accu
            best_accu = max(accu_new, best_accu)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': accu_new,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
        print('\nBest Accu: %f\n'%best_accu)
        logging.info('\nBest Accu: %f\n'%best_accu)

def unpack_batch(batch, args):
    if args.adaptive_gaussian_field:
        query_imgs, rs_imgs, original_click_map, click_xy, ori_gt_bbox, sample_index = batch
        return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, (click_xy,)
    if args.sam_refined_pe:
        query_imgs, rs_imgs, original_click_map, sam_masks, sam_scores, ori_gt_bbox, sample_index = batch
        return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, (sam_masks, sam_scores)
    if args.adaptive_sam_prompt:
        query_imgs, rs_imgs, original_click_map, click_xy, sam_masks, sam_scores, ori_gt_bbox, sample_index = batch
        return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, (click_xy, sam_masks, sam_scores)
    if args.sam_prompt:
        query_imgs, rs_imgs, original_click_map, gaussian_map, sam_mask, masked_gaussian, ori_gt_bbox, sample_index = batch
        return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, (gaussian_map, sam_mask, masked_gaussian)
    if args.gaussian_only:
        query_imgs, rs_imgs, original_click_map, gaussian_map, ori_gt_bbox, sample_index = batch
        return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, (gaussian_map,)
    query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index = batch
    return query_imgs, rs_imgs, original_click_map, ori_gt_bbox, sample_index, ()


def forward_model(model, query_imgs, rs_imgs, original_click_map, prompt_maps):
    if prompt_maps:
        return model(query_imgs, rs_imgs, original_click_map, *prompt_maps)
    return model(query_imgs, rs_imgs, original_click_map)


def is_ms_detection_variant(args):
    return args.trogeo_ms_det_variant != 'none'


def get_rccd_weight(epoch, args):
    """Zero-based warm-up and linear ramp for the training-only RCCD term."""
    if not args.rccd or epoch < args.rccd_warmup_epochs:
        return 0.0
    if args.rccd_ramp_epochs == 0 or epoch >= args.rccd_warmup_epochs + args.rccd_ramp_epochs:
        return args.rccd_weight
    progress = (epoch - args.rccd_warmup_epochs + 1) / float(args.rccd_ramp_epochs + 1)
    return args.rccd_weight * progress


def _ms_predictions_and_loss(predictions, ori_gt_bbox, anchors_full, args, include_loss=True):
    """Return decoded final boxes and, during training, the matching loss terms."""
    variant = args.trogeo_ms_det_variant
    coarse_variants = ('h2_ind_cg', 'h2_ind_csfi_cg', 'h2_ind_hier', 'h2_ind_cg_amhcsfi_res',
                       'h2_ind_csfi_cg_channel', 'h2_ind_csfi_cg_dir', 'h2_ind_csfi_cg_ar',
                       'h2_ind_cg_habr_prior')
    habr_prior_variants = ('h2_ind_habr_prior', 'h2_ind_habr_adapt', 'h2_ind_habr')
    fg_amhcsfi_single_variants = (
        'h2_ind_fg_amhcsfi_res_s3', 'h2_ind_fg_amhcsfi_res_s4', 'h2_ind_fg_amhcsfi_res_afuse')
    bi_amhcsfi_single_variants = ('h2_ind_amhcsfi_res_bi_afuse',)
    qcc_a_variants = ('h2_ind_amhcsfi_res_bi_qcc_a',)
    qcc_af_variants = ('h2_ind_amhcsfi_res_bi_qcc_af',)
    qcc_rank_variants = ('h2_ind_amhcsfi_res_bi_qcc_b', 'h2_ind_amhcsfi_res_bi_qcc_full')
    qcc_variants = qcc_a_variants + qcc_af_variants + qcc_rank_variants
    loss_aux = None
    qcc_losses = None
    three_scale_variants = (
        'h2_ind_3scale', 'h2_ind_3scale_stage2cls05', 'h2_ind_3scale_pe_ln_amp',
        'h2_ind_3scale_pe_all_add', 'h2_ind_3scale_pe_all_key', 'h2_ind_3scale_le_ind_res',
        'h2_ind_3scale_le_stage2_res')
    if variant in three_scale_variants:
        p2 = predictions['stage2'].view(predictions['stage2'].shape[0], 9, 5, 64, 64)
        p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
        p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
        if include_loss:
            if variant == 'h2_ind_3scale_stage2cls05':
                loss_geo, loss_cls = three_head_yolo_loss_stage2_cls_half(
                    p2, p3, p4, ori_gt_bbox, anchors_full, args.img_size)
            else:
                loss_geo, loss_cls = three_head_yolo_loss(p2, p3, p4, ori_gt_bbox, anchors_full, args.img_size)
        else:
            loss_geo = loss_cls = None
        final_box, diagnostics = select_three_heads(p2, p3, p4, anchors_full, args.img_size)
    elif variant == 'correct63':
        joint = predictions['joint'].view(predictions['joint'].shape[0], 9, 5, 64, 64)
        target, best = build_target(ori_gt_bbox, anchors_full, args.img_size, 64)
        if include_loss:
            loss_geo, loss_cls = yolo_loss(joint, target, anchors_full, best, args.img_size)
        else:
            loss_geo = loss_cls = None
        _, _, _, _, final_box, _ = eval_iou_acc(joint, ori_gt_bbox, anchors_full, best[:, 1], best[:, 2],
                                                 args.img_size, iou_threshold_list=[0.5])
        diagnostics = {}
    elif variant == 'b_multigrid':
        p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 6, 5, 64, 64)
        p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 3, 5, 32, 32)
        if include_loss:
            loss_geo, loss_cls = multigrid_yolo_loss(p3, p4, ori_gt_bbox, anchors_full, args.img_size)
        else:
            loss_geo = loss_cls = None
        final_box = decode_multigrid_top1(p3, p4, anchors_full, args.img_size)
        diagnostics = {}
    elif variant in fg_amhcsfi_single_variants:
        p = predictions['single'].view(predictions['single'].shape[0], 9, 5, 64, 64)
        target, best = build_target(ori_gt_bbox, anchors_full, args.img_size, 64)
        if include_loss:
            loss_geo, loss_cls = yolo_loss(p, target, anchors_full, best, args.img_size)
            loss_aux = coarse_heatmap_loss(predictions['fine_logits'], ori_gt_bbox,
                                           args.img_size, args.fine_sigma)
        else:
            loss_geo = loss_cls = None
        final_box, _ = decode_top1(p, anchors_full, args.img_size)
        diagnostics = {}
        if 'fusion_weights' in predictions:
            diagnostics['fusion_w3'] = predictions['fusion_weights'][:, 0].mean()
            diagnostics['fusion_w4'] = predictions['fusion_weights'][:, 1].mean()
    elif variant in bi_amhcsfi_single_variants:
        p = predictions['single'].view(predictions['single'].shape[0], 9, 5, 64, 64)
        target, best = build_target(ori_gt_bbox, anchors_full, args.img_size, 64)
        if include_loss:
            loss_geo, loss_cls = yolo_loss(p, target, anchors_full, best, args.img_size)
            loss_coarse = coarse_heatmap_loss(predictions['coarse_logits'], ori_gt_bbox,
                                              args.img_size, args.coarse_sigma)
            loss_fine = coarse_heatmap_loss(predictions['fine_logits'], ori_gt_bbox,
                                            args.img_size, args.fine_sigma)
            loss_aux = 0.5 * (loss_coarse + loss_fine)
        else:
            loss_geo = loss_cls = None
        final_box, _ = decode_top1(p, anchors_full, args.img_size)
        diagnostics = {
            'fusion_w3': predictions['fusion_weights'][:, 0].mean(),
            'fusion_w4': predictions['fusion_weights'][:, 1].mean(),
        }
    elif variant in qcc_variants:
        p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
        p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
        if include_loss:
            loss_geo, loss_cls = two_head_yolo_loss(p3, p4, ori_gt_bbox, anchors_full, args.img_size)
            loss_coarse = coarse_heatmap_loss(predictions['coarse_logits'], ori_gt_bbox,
                                              args.img_size, args.coarse_sigma)
            loss_fine = coarse_heatmap_loss(predictions['fine_logits'], ori_gt_bbox,
                                            args.img_size, args.fine_sigma)
            loss_aux = 0.5 * (loss_coarse + loss_fine)
        else:
            loss_geo = loss_cls = None
        state = build_qcc_state(p3, p4, predictions['qcc_quality3'], predictions['qcc_quality4'],
                                anchors_full, args.img_size)
        if variant in qcc_a_variants:
            final_box, diagnostics = select_qcc_a(state)
        elif variant in qcc_af_variants:
            final_box, diagnostics = select_qcc_af(state, fusion_iou=args.qcc_fusion_iou)
        else:
            if 'qcc_rank_logit' not in predictions:
                raise RuntimeError('QCC-B/Full requires detached qcc_rank_logit before decoding')
            final_box, diagnostics = select_qcc_ranked(
                state, predictions['qcc_rank_logit'],
                fusion_iou=args.qcc_fusion_iou if variant == 'h2_ind_amhcsfi_res_bi_qcc_full' else None)
        if include_loss:
            quality_loss, iou3, iou4 = qcc_quality_loss(state, ori_gt_bbox)
            rank_loss = None
            if variant in qcc_rank_variants:
                rank_loss, rank_diag = qcc_pairwise_ranking_loss(
                    predictions['qcc_rank_logit'], iou3, iou4, args.qcc_rank_epsilon)
                diagnostics['qcc_rank_valid_ratio'] = rank_diag['valid_ratio']
            qcc_losses = {'quality': quality_loss, 'rank': rank_loss}
    elif variant == 'h2_ind_hier':
        p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
        target, best = build_target(ori_gt_bbox, anchors_full, args.img_size, 64)
        if include_loss:
            loss_geo, loss_cls = yolo_loss(p3, target, anchors_full, best, args.img_size)
            loss_aux = coarse_heatmap_loss(predictions['coarse_logits'], ori_gt_bbox,
                                           args.img_size, args.coarse_sigma)
        else:
            loss_geo = loss_cls = None
        final_box, _ = decode_top1(p3, anchors_full, args.img_size)
        diagnostics = {}
    else:
        p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
        p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
        if include_loss:
            if args.bbox_threshold_reg:
                loss_geo, loss_cls = two_head_yolo_threshold_reg_loss(
                    p3, p4, ori_gt_bbox, anchors_full, args.img_size,
                    reg_weight=args.bbox_threshold_reg_weight,
                    temperature=args.bbox_threshold_temperature,
                    weight25=args.bbox_threshold_weight25,
                    weight50=args.bbox_threshold_weight50)
            else:
                loss_geo, loss_cls = two_head_yolo_loss(p3, p4, ori_gt_bbox, anchors_full, args.img_size)
            if variant in habr_prior_variants:
                loss_prior3 = coarse_heatmap_loss(predictions['habr_prior3_logits'], ori_gt_bbox,
                                                   args.img_size, args.fine_sigma)
                loss_prior4 = coarse_heatmap_loss(predictions['habr_prior4_logits'], ori_gt_bbox,
                                                   args.img_size, args.coarse_sigma)
                loss_aux = 0.5 * (loss_prior3 + loss_prior4)
            elif variant in ('h2_ind_csfi_fg', 'h2_ind_fg_nocsfi', 'h2_ind_fg_amhcsfi_res'):
                loss_aux = coarse_heatmap_loss(predictions['fine_logits'], ori_gt_bbox,
                                               args.img_size, args.fine_sigma)
            elif variant in ('h2_ind_csfi_bi', 'h2_ind_bi_nocsfi', 'h2_ind_mhcsfi_bi', 'h2_ind_amhcsfi_bi',
                             'h2_ind_amhcsfi_res_bi'):
                loss_coarse = coarse_heatmap_loss(predictions['coarse_logits'], ori_gt_bbox,
                                                  args.img_size, args.coarse_sigma)
                loss_fine = coarse_heatmap_loss(predictions['fine_logits'], ori_gt_bbox,
                                                args.img_size, args.fine_sigma)
                loss_aux = 0.5 * (loss_coarse + loss_fine)
            elif variant in coarse_variants:
                loss_aux = coarse_heatmap_loss(predictions['coarse_logits'], ori_gt_bbox,
                                               args.img_size, args.coarse_sigma)
        else:
            loss_geo = loss_cls = None
        if args.hqs_v2a or args.hqs_v2b:
            final_box, diagnostics = select_two_heads_hqs_v2a(
                p3, p4, predictions['hqs_rank_logit'], anchors_full, args.img_size,
                threshold=args.hqs_rank_threshold)
        elif args.hqs_v1:
            final_box, diagnostics = select_two_heads_hqs(
                p3, p4, predictions['hqs_logits'], anchors_full, args.img_size)
        else:
            final_box, diagnostics = select_two_heads(
                p3, p4, anchors_full, args.img_size,
                fusion=variant in ('h3_ind', 'h3_adaptive'),
                iou_threshold=args.h3_iou_threshold,
                adaptive=(variant == 'h3_adaptive'))
    return loss_geo, loss_cls, loss_aux, qcc_losses, final_box, diagnostics


def hqs_quality_loss(predictions, ori_gt_bbox, anchors_full, args):
    p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
    p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
    with torch.no_grad():
        box3, _ = decode_top1(p3, anchors_full, args.img_size)
        box4, _ = decode_top1(p4, anchors_full, args.img_size)
        iou3 = bbox_iou(box3, ori_gt_bbox, x1y1x2y2=True)
        iou4 = bbox_iou(box4, ori_gt_bbox, x1y1x2y2=True)
        qualities = torch.stack((iou3, iou4), dim=1)
        target = torch.softmax(qualities / args.hqs_temperature, dim=1)
        gap = (iou3 - iou4).abs()
        weight = torch.clamp(gap / args.hqs_gap_delta, max=1.0)
        oracle_use3 = iou3 >= iou4
    per_sample = F.kl_div(F.log_softmax(predictions['hqs_logits'], dim=1), target, reduction='none').sum(dim=1)
    loss = (weight * per_sample).sum() / weight.sum().clamp_min(1e-6)
    hqs_use3 = predictions['hqs_logits'][:, 0] >= predictions['hqs_logits'][:, 1]
    return loss, {'agreement': (hqs_use3 == oracle_use3).float().mean(), 'gap': gap}


@torch.no_grad()
def build_hqs_v2b_quality_stats(predictions, anchors_full, image_wh):
    """Return the seven detached prediction-quality statistics for HQS-v2b."""
    p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
    p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
    batch_size = p3.shape[0]
    conf3, conf4 = p3[:, :, 4].reshape(batch_size, -1), p4[:, :, 4].reshape(batch_size, -1)
    prob3, prob4 = torch.softmax(conf3, dim=1), torch.softmax(conf4, dim=1)
    top2_3, top2_4 = torch.topk(prob3, k=2, dim=1).values, torch.topk(prob4, k=2, dim=1).values
    score3, score4 = top2_3[:, 0], top2_4[:, 0]
    margin3, margin4 = top2_3[:, 0] - top2_3[:, 1], top2_4[:, 0] - top2_4[:, 1]
    norm = math.log(float(prob3.shape[1]))
    entropy3 = -(prob3 * torch.log(prob3.clamp_min(1e-12))).sum(dim=1) / norm
    entropy4 = -(prob4 * torch.log(prob4.clamp_min(1e-12))).sum(dim=1) / norm
    box3, _ = decode_top1(p3, anchors_full, image_wh)
    box4, _ = decode_top1(p4, anchors_full, image_wh)
    pair_iou = bbox_iou(box3, box4, x1y1x2y2=True).clamp(0.0, 1.0)
    quality_stats = torch.stack((score3, score4, margin3, margin4, entropy3, entropy4, pair_iou), dim=1)
    if quality_stats.shape != (batch_size, 7):
        raise RuntimeError('Unexpected HQS-v2b statistics shape {}'.format(tuple(quality_stats.shape)))
    return quality_stats, {
        'score3': score3, 'score4': score4, 'margin3': margin3, 'margin4': margin4,
        'entropy3': entropy3, 'entropy4': entropy4, 'pair_iou': pair_iou,
    }


def attach_hqs_v2b_rank_logit(model, predictions, anchors_full, args):
    """Attach v2b's trainable rank logit after anchor-aware quality extraction."""
    quality_stats, diagnostics = build_hqs_v2b_quality_stats(predictions, anchors_full, args.img_size)
    predictions['hqs_rank_logit'] = model.module.head_prediction_quality_ranker(
        predictions['hqs_feat3'].detach(), predictions['hqs_feat4'].detach(), quality_stats.detach())
    return diagnostics


def attach_qcc_rank_logit(model, predictions, anchors_full, args):
    """QCC ranker receives only detached anchor-aware prediction state."""
    p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
    p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
    state = build_qcc_state(p3, p4, predictions['qcc_quality3'], predictions['qcc_quality4'],
                            anchors_full, args.img_size)
    predictions['qcc_rank_logit'] = model.module.qcc_ranker(state['rank_input'].detach())


def qcc_quality_loss(state, gt_bbox):
    """Train quality heads against detached Top-1 localization IoU targets."""
    with torch.no_grad():
        iou3 = bbox_iou(state['box3'].detach(), gt_bbox, x1y1x2y2=True).clamp(0.0, 1.0)
        iou4 = bbox_iou(state['box4'].detach(), gt_bbox, x1y1x2y2=True).clamp(0.0, 1.0)
    loss = 0.5 * (F.smooth_l1_loss(state['quality3'], iou3) + F.smooth_l1_loss(state['quality4'], iou4))
    return loss, iou3, iou4


def qcc_pairwise_ranking_loss(rank_logit, iou3, iou4, epsilon):
    """Balanced BCE only where one QCC head has a clear GT-IoU advantage."""
    gap = iou3.detach() - iou4.detach()
    valid = gap.abs() > float(epsilon)
    if not bool(valid.any()):
        return rank_logit.sum() * 0.0, {'valid_ratio': valid.float().mean()}
    target = (gap[valid] > 0).float()
    per_sample = F.binary_cross_entropy_with_logits(rank_logit[valid], target, reduction='none')
    mask3, mask4 = target.eq(1.0), target.eq(0.0)
    if bool(mask3.any()) and bool(mask4.any()):
        weights = torch.zeros_like(per_sample)
        weights[mask3] = 0.5 / mask3.sum()
        weights[mask4] = 0.5 / mask4.sum()
        loss = (weights * per_sample).sum()
    else:
        loss = per_sample.mean()
    return loss, {'valid_ratio': valid.float().mean()}


def hqs_pairwise_ranking_loss(predictions, ori_gt_bbox, anchors_full, args):
    p3 = predictions['stage3'].view(predictions['stage3'].shape[0], 9, 5, 64, 64)
    p4 = predictions['stage4'].view(predictions['stage4'].shape[0], 9, 5, 64, 64)
    rank_logit = predictions['hqs_rank_logit']
    with torch.no_grad():
        box3, _ = decode_top1(p3, anchors_full, args.img_size)
        box4, _ = decode_top1(p4, anchors_full, args.img_size)
        signed_gap = bbox_iou(box3, ori_gt_bbox, x1y1x2y2=True) - bbox_iou(
            box4, ori_gt_bbox, x1y1x2y2=True)
        valid = signed_gap.abs() > args.hqs_rank_epsilon
        target = (signed_gap > 0).float()
    valid_count = int(valid.sum().item())
    hqs_use3 = rank_logit >= 0.0
    if valid_count == 0:
        return rank_logit.sum() * 0.0, {
            'valid_count': 0, 'target_head3_count': 0, 'target_head4_count': 0,
            'agreement_correct': 0, 'selected_head3_count': int(hqs_use3.sum().item()),
            'mixed_class_batch': 0,
        }
    logits, targets = rank_logit[valid], target[valid]
    mask3, mask4 = targets.eq(1.0), targets.eq(0.0)
    n3, n4 = int(mask3.sum().item()), int(mask4.sum().item())
    per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if n3 > 0 and n4 > 0:
        weights = torch.zeros_like(per_sample)
        weights[mask3], weights[mask4] = 0.5 / n3, 0.5 / n4
        loss, mixed = (weights * per_sample).sum(), 1
    else:
        loss, mixed = per_sample.mean(), 0
    agreement_correct = int((hqs_use3[valid] == (signed_gap[valid] > 0)).sum().item())
    return loss, {
        'valid_count': valid_count, 'target_head3_count': n3, 'target_head4_count': n4,
        'agreement_correct': agreement_correct,
        'selected_head3_count': int(hqs_use3.sum().item()), 'mixed_class_batch': mixed,
    }


def train_hqs_v2a_epoch(train_loader, model, optimizer, epoch, args):
    model.eval()
    model.module.head_pairwise_ranker.train()
    anchors = np.array([float(x.strip()) for x in args.anchors.split(',')]).reshape(-1, 2)[::-1].copy()
    anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
    losses = AverageMeter()
    total = valid = target3 = target4 = selected3 = correct = mixed = nonempty = 0
    for batch_idx, batch in enumerate(train_loader):
        query, rs, click, bbox, _, prompts = unpack_batch(batch, args)
        query, rs, click = query.cuda(), rs.cuda(), click.cuda()
        prompts = tuple(item.cuda() for item in prompts)
        bbox = torch.clamp(bbox.cuda(), min=0, max=args.img_size - 1)
        predictions, _ = forward_model(model, query, rs, click, prompts)
        loss, diag = hqs_pairwise_ranking_loss(predictions, bbox, anchors, args)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_size = query.shape[0]; losses.update(loss.item(), batch_size); total += batch_size
        valid += diag['valid_count']; target3 += diag['target_head3_count']; target4 += diag['target_head4_count']
        selected3 += diag['selected_head3_count']; correct += diag['agreement_correct']
        if diag['valid_count']:
            nonempty += 1; mixed += diag['mixed_class_batch']
        if batch_idx % args.print_freq == 0:
            text = ('[HQS-v2a] Epoch [{}/{}] batch {}/{} Loss {:.5f} SelectH3 {:.2f}% '
                    'SelectH4 {:.2f}% OracleAgreeValid {:.2f}% Valid {:.2f}% TargetH3={} TargetH4={}').format(
                epoch, args.max_epoch, batch_idx, len(train_loader), losses.avg, 100 * selected3 / max(total, 1),
                100 * (1 - selected3 / max(total, 1)), 100 * correct / max(valid, 1),
                100 * valid / max(total, 1), target3, target4)
            print(text); logging.info(text)
    summary = ('[HQS-v2a epoch summary] epoch={} Loss={:.6f} SelectH3={:.2f}% SelectH4={:.2f}% '
               'OracleAgreeValid={:.2f}% Valid={:.2f}% TargetH3={} TargetH4={} MixedClassBatch={:.2f}%').format(
        epoch, losses.avg, 100 * selected3 / max(total, 1), 100 * (1 - selected3 / max(total, 1)),
        100 * correct / max(valid, 1), 100 * valid / max(total, 1), target3, target4,
        100 * mixed / max(nonempty, 1))
    print(summary); logging.info(summary)


def train_hqs_v2b_epoch(train_loader, model, optimizer, epoch, args):
    """Keep FG-Res frozen/eval; v2b changes only the ranker's input statistics."""
    model.eval()
    model.module.head_prediction_quality_ranker.train()
    anchors = np.array([float(x.strip()) for x in args.anchors.split(',')]).reshape(-1, 2)[::-1].copy()
    anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
    losses = AverageMeter()
    total = valid = target3 = target4 = selected3 = correct = mixed = nonempty = 0
    quality_totals = {'margin3': 0.0, 'margin4': 0.0, 'entropy3': 0.0, 'entropy4': 0.0, 'pair_iou': 0.0}
    for batch_idx, batch in enumerate(train_loader):
        query, rs, click, bbox, _, prompts = unpack_batch(batch, args)
        query, rs, click = query.cuda(), rs.cuda(), click.cuda()
        prompts = tuple(item.cuda() for item in prompts)
        bbox = torch.clamp(bbox.cuda(), min=0, max=args.img_size - 1)
        predictions, _ = forward_model(model, query, rs, click, prompts)
        quality_diag = attach_hqs_v2b_rank_logit(model, predictions, anchors, args)
        loss, diag = hqs_pairwise_ranking_loss(predictions, bbox, anchors, args)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_size = query.shape[0]; losses.update(loss.item(), batch_size); total += batch_size
        valid += diag['valid_count']; target3 += diag['target_head3_count']; target4 += diag['target_head4_count']
        selected3 += diag['selected_head3_count']; correct += diag['agreement_correct']
        for name in quality_totals:
            quality_totals[name] += quality_diag[name].mean().item() * batch_size
        if diag['valid_count']:
            nonempty += 1; mixed += diag['mixed_class_batch']
        if batch_idx % args.print_freq == 0:
            text = ('[HQS-v2b] Epoch [{}/{}] batch {}/{} Loss {:.5f} SelectH3 {:.2f}% SelectH4 {:.2f}% '
                    'OracleAgreeValid {:.2f}% Valid {:.2f}% TargetH3={} TargetH4={} '
                    'M3={:.6f} M4={:.6f} H3={:.6f} H4={:.6f} PairIoU={:.4f}').format(
                epoch, args.max_epoch, batch_idx, len(train_loader), losses.avg,
                100 * selected3 / max(total, 1), 100 * (1 - selected3 / max(total, 1)),
                100 * correct / max(valid, 1), 100 * valid / max(total, 1), target3, target4,
                quality_diag['margin3'].mean().item(), quality_diag['margin4'].mean().item(),
                quality_diag['entropy3'].mean().item(), quality_diag['entropy4'].mean().item(),
                quality_diag['pair_iou'].mean().item())
            print(text); logging.info(text)
    summary = ('[HQS-v2b epoch summary] epoch={} Loss={:.6f} SelectH3={:.2f}% SelectH4={:.2f}% '
               'OracleAgreeValid={:.2f}% Valid={:.2f}% TargetH3={} TargetH4={} MixedClassBatch={:.2f}% '
               'M3={:.6f} M4={:.6f} H3={:.6f} H4={:.6f} PairIoU={:.4f}').format(
        epoch, losses.avg, 100 * selected3 / max(total, 1), 100 * (1 - selected3 / max(total, 1)),
        100 * correct / max(valid, 1), 100 * valid / max(total, 1), target3, target4,
        100 * mixed / max(nonempty, 1), *(quality_totals[name] / max(total, 1) for name in quality_totals))
    print(summary); logging.info(summary)


def train_hqs_epoch(train_loader, model, optimizer, epoch, args):
    """Keep FG-Res fully frozen/eval; optimize only the selector MLP."""
    model.eval()
    model.module.head_quality_selector.train()
    anchors = np.array([float(x.strip()) for x in args.anchors.split(',')]).reshape(-1, 2)[::-1].copy()
    anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
    losses, agreements = AverageMeter(), AverageMeter()
    for batch_idx, batch in enumerate(train_loader):
        query, rs, click, bbox, _, prompts = unpack_batch(batch, args)
        query, rs, click = query.cuda(), rs.cuda(), click.cuda()
        prompts = tuple(item.cuda() for item in prompts)
        bbox = torch.clamp(bbox.cuda(), min=0, max=args.img_size - 1)
        predictions, _ = forward_model(model, query, rs, click, prompts)
        loss, diag = hqs_quality_loss(predictions, bbox, anchors, args)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.update(loss.item(), query.shape[0]); agreements.update(diag['agreement'].item(), query.shape[0])
        if batch_idx % args.print_freq == 0:
            text = '[HQS] Epoch [{}/{}] batch {}/{} Loss {:.5f} OracleAgree {:.4f}'.format(
                epoch, args.max_epoch, batch_idx, len(train_loader), losses.avg, agreements.avg)
            print(text); logging.info(text)


def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_cls_losses = AverageMeter()
    avg_geo_losses = AverageMeter()
    avg_rccd_losses = AverageMeter()
    avg_rccd_weight3 = AverageMeter()
    avg_rccd_weight4 = AverageMeter()
    avg_rccd_rel3 = AverageMeter()
    avg_rccd_rel4 = AverageMeter()
    avg_accu = AverageMeter()
    avg_accu_center = AverageMeter()
    avg_iou = AverageMeter()

    model.train()
    if args.freeze_prompt_only:
        if args.sam_prompt:
            prompt_module_prefixes = ('prompt_fusion',)
        elif args.adaptive_sam_prompt:
            prompt_module_prefixes = ('adaptive_prompt',)
        elif args.hisym_pae:
            prompt_module_prefixes = tuple(p for p, active in
                                           (('sam_refiner', args.sam_refined_pe), ('pae_fusion', True)) if active)
        else:
            prompt_module_prefixes = tuple(p for p, active in
                                           (('sam_refiner', args.sam_refined_pe), ('rgbp_interaction', args.rgbp_interaction)) if active)
        for name, module in model.module.named_modules():
            if name and not name.startswith(prompt_module_prefixes):
                module.eval()
    end = time.time()
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()
    
    for batch_idx, batch in enumerate(train_loader):
        query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, sample_index, prompt_maps = unpack_batch(batch, args)
        if args.rng_probe and epoch == 0 and batch_idx < 3:
            print(
                'RNG_PROBE batch={} indices={} bbox_head={} rs_sum={:.6f}'.format(
                    batch_idx, sample_index.tolist(), ori_gt_bbox[:2].tolist(), float(rs_imgs.sum())
                ),
                flush=True,
            )
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        prompt_maps = tuple(prompt_map.cuda() for prompt_map in prompt_maps)
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        prediction_output, _ = forward_model(model, query_imgs, rs_imgs, mat_clickxy, prompt_maps)
        loss_rccd = None
        rccd_diagnostics = None
        current_rccd_weight = 0.0
        qcc_losses = None
        if is_ms_detection_variant(args):
            if args.trogeo_ms_det_variant in ('h2_ind_amhcsfi_res_bi_qcc_b',
                                               'h2_ind_amhcsfi_res_bi_qcc_full'):
                attach_qcc_rank_logit(model, prediction_output, anchors_full, args)
            loss_geo, loss_cls, loss_aux, qcc_losses, final_box, _ = _ms_predictions_and_loss(
                prediction_output, ori_gt_bbox, anchors_full, args, include_loss=True)
            accu, _, iou, accu_center = eval_decoded_boxes(final_box, ori_gt_bbox, args.img_size)
            if args.rccd:
                current_rccd_weight = get_rccd_weight(epoch, args)
                if current_rccd_weight > 0.0:
                    p3_rccd = prediction_output['stage3'].view(prediction_output['stage3'].shape[0], 9, 5, 64, 64)
                    p4_rccd = prediction_output['stage4'].view(prediction_output['stage4'].shape[0], 9, 5, 64, 64)
                    loss_rccd, rccd_diagnostics = rccd_consensus_loss(
                        p3_rccd, p4_rccd, temperature=args.rccd_temperature,
                        weight_temperature=args.rccd_tau)
        else:
            pred_anchor = prediction_output.view(prediction_output.shape[0], 9, 5,
                                                 prediction_output.shape[2], prediction_output.shape[3])
            new_gt_bbox, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
            loss_geo, loss_cls = yolo_loss(pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, args.img_size)
            accu_list, accu_center, iou, _, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full,
                best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5])
            accu = accu_list[0]
        loss = loss_cls + loss_geo * args.beta
        if loss_aux is not None:
            loss = loss + args.coarse_loss_weight * loss_aux
        if qcc_losses is not None:
            loss = loss + args.qcc_quality_weight * qcc_losses['quality']
            if qcc_losses['rank'] is not None:
                loss = loss + args.qcc_rank_weight * qcc_losses['rank']
        if loss_rccd is not None:
            loss = loss + current_rccd_weight * loss_rccd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_losses.update(loss.item(), query_imgs.shape[0])
        avg_geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        avg_cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        if loss_rccd is not None:
            avg_rccd_losses.update(loss_rccd.item(), query_imgs.shape[0])
            avg_rccd_weight3.update(rccd_diagnostics['rccd_weight3'].item(), query_imgs.shape[0])
            avg_rccd_weight4.update(rccd_diagnostics['rccd_weight4'].item(), query_imgs.shape[0])
            avg_rccd_rel3.update(rccd_diagnostics['rccd_reliability3'].item(), query_imgs.shape[0])
            avg_rccd_rel4.update(rccd_diagnostics['rccd_reliability4'].item(), query_imgs.shape[0])
        
        ## metrics
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu.update(accu, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Geo Loss {geo.val:.4f} ({geo.avg:.4f})\t' \
                'Cls Loss {cls.val:.4f} ({cls.avg:.4f})\t' \
                'Accu {accu.val:.4f} ({accu.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    epoch, batch_idx, len(train_loader), batch_time=batch_time, \
                    loss=avg_losses, geo=avg_geo_losses, cls=avg_cls_losses, accu=avg_accu, miou=avg_iou, accu_c=avg_accu_center)
            if args.rccd:
                print_str += ('RCCD {rccd:.6f}\tRCCD_lambda {lam:.4f}\tRCCD_w3 {w3:.4f}\t'
                              'RCCD_w4 {w4:.4f}\tRCCD_r3 {r3:.4f}\tRCCD_r4 {r4:.4f}\t').format(
                                  rccd=avg_rccd_losses.avg, lam=current_rccd_weight,
                                  w3=avg_rccd_weight3.avg, w4=avg_rccd_weight4.avg,
                                  r3=avg_rccd_rel3.avg, r4=avg_rccd_rel4.avg)
            print(print_str)
            logging.info(print_str)


def binary_pairwise_auc(positive_scores, negative_scores):
    """ROC-AUC without an sklearn dependency; positive means Head3 is better."""
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return float('nan')
    difference = positive_scores[:, None] - negative_scores[None, :]
    return float(difference.gt(0).float().mean() + 0.5 * difference.eq(0).float().mean())


def describe_rank_logits(name, values):
    """Log a compact distribution report for one HQS-v2a ranking class."""
    if values.numel() == 0:
        text = '{}: empty'.format(name)
    else:
        values = values.float()
        text = ('{}: N={} mean={:.6f} std={:.6f} median={:.6f} q25={:.6f} '
                'q75={:.6f} min={:.6f} max={:.6f}').format(
                    name, values.numel(), values.mean().item(), values.std(unbiased=False).item(),
                    values.median().item(), torch.quantile(values, 0.25).item(),
                    torch.quantile(values, 0.75).item(), values.min().item(), values.max().item())
    print(text)
    logging.info(text)
    return text


def test_epoch(data_loader, model, args):
    batch_time = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()
    diagnostic_meters = {}
    oracle_metric_meters = {}
    oracle_extra_meters = {
        'agreement': AverageMeter(), 'confidence_head3_ratio': AverageMeter(),
        'oracle_head3_ratio': AverageMeter(), 'oracle_iou_gain': AverageMeter(),
        'head3_better_ratio': AverageMeter(), 'head4_better_ratio': AverageMeter(),
        'head_tie_ratio': AverageMeter(), 'recover_acc50': AverageMeter(),
        'recover_acc25': AverageMeter(),
    }
    oracle_rows = []
    rank_diag_logits, rank_diag_box3, rank_diag_box4 = [], [], []
    rank_diag_gt, rank_diag_iou3, rank_diag_iou4, rank_diag_sample_index = [], [], [], []

    def oracle_meters(mode_name):
        if mode_name not in oracle_metric_meters:
            oracle_metric_meters[mode_name] = {
                'acc50': AverageMeter(), 'acc25': AverageMeter(),
                'miou': AverageMeter(), 'center': AverageMeter(),
            }
        return oracle_metric_meters[mode_name]
    
    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    #print(datetime.datetime.now())
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    for batch_idx, batch in enumerate(data_loader):
        query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, sample_index, prompt_maps = unpack_batch(batch, args)
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        prompt_maps = tuple(prompt_map.cuda() for prompt_map in prompt_maps)
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        with torch.no_grad():
            prediction_output, attn_score = forward_model(model, query_imgs, rs_imgs, mat_clickxy, prompt_maps)
            if args.hqs_v2b:
                attach_hqs_v2b_rank_logit(model, prediction_output, anchors_full, args)
            if args.trogeo_ms_det_variant in ('h2_ind_amhcsfi_res_bi_qcc_b',
                                               'h2_ind_amhcsfi_res_bi_qcc_full'):
                attach_qcc_rank_logit(model, prediction_output, anchors_full, args)
            if is_ms_detection_variant(args):
                _, _, _, _, final_box, diagnostics = _ms_predictions_and_loss(
                    prediction_output, ori_gt_bbox, anchors_full, args, include_loss=False)
                accu50, accu25, iou, accu_center = eval_decoded_boxes(final_box, ori_gt_bbox, args.img_size)
                accu_list = [accu50, accu25]
                for name, value in diagnostics.items():
                    diagnostic_meters.setdefault(name, AverageMeter()).update(float(value), query_imgs.shape[0])
                if args.hqs_v2a or args.hqs_v2b:
                    p3_rank = prediction_output['stage3'].view(prediction_output['stage3'].shape[0], 9, 5, 64, 64)
                    p4_rank = prediction_output['stage4'].view(prediction_output['stage4'].shape[0], 9, 5, 64, 64)
                    box3_rank, _ = decode_top1(p3_rank, anchors_full, args.img_size)
                    box4_rank, _ = decode_top1(p4_rank, anchors_full, args.img_size)
                    iou3_rank = bbox_iou(box3_rank, ori_gt_bbox, x1y1x2y2=True)
                    iou4_rank = bbox_iou(box4_rank, ori_gt_bbox, x1y1x2y2=True)
                    signed_gap = iou3_rank - iou4_rank
                    valid_rank = signed_gap.abs() > args.hqs_rank_epsilon
                    valid_count = int(valid_rank.sum().item())
                    if valid_count:
                        agreement = ((prediction_output['hqs_rank_logit'][valid_rank] >= 0) ==
                                     (signed_gap[valid_rank] > 0)).float().mean()
                        diagnostic_meters.setdefault('hqs_v2a_oracle_agreement_valid', AverageMeter()).update(
                            float(agreement), valid_count)
                    diagnostic_meters.setdefault('hqs_v2a_valid_ratio', AverageMeter()).update(
                        float(valid_rank.float().mean()), query_imgs.shape[0])
                    if args.hqs_v2a_rank_diag or args.hqs_v2b_rank_diag:
                        rank_logit = prediction_output['hqs_rank_logit'].detach()
                        zero_box = torch.where((rank_logit >= 0.0)[:, None], box3_rank, box4_rank)
                        if abs(args.hqs_rank_threshold) < 1e-12:
                            max_diff = (zero_box - final_box).abs().max().item()
                            if max_diff > 1e-4:
                                raise RuntimeError('HQS-v2a rank diagnostic failed to reproduce threshold=0 '
                                                   'decoder: max_diff={:.8f}'.format(max_diff))
                        rank_diag_logits.append(rank_logit.cpu())
                        rank_diag_box3.append(box3_rank.detach().cpu())
                        rank_diag_box4.append(box4_rank.detach().cpu())
                        rank_diag_gt.append(ori_gt_bbox.detach().cpu())
                        rank_diag_iou3.append(iou3_rank.detach().cpu())
                        rank_diag_iou4.append(iou4_rank.detach().cpu())
                        rank_diag_sample_index.append(sample_index.detach().cpu())
                if args.hqs_oracle_diag:
                    p3 = prediction_output['stage3'].view(prediction_output['stage3'].shape[0], 9, 5, 64, 64)
                    p4 = prediction_output['stage4'].view(prediction_output['stage4'].shape[0], 9, 5, 64, 64)
                    oracle_diag = analyze_two_head_oracle(
                        p3, p4, ori_gt_bbox, anchors_full, args.img_size)
                    max_selector_diff = (oracle_diag['confidence_box'] - final_box).abs().max().item()
                    if max_selector_diff > 1e-4:
                        raise RuntimeError('HQS Oracle diagnostic changed baseline decoding: '
                                           'max box difference = {:.8f}'.format(max_selector_diff))
                    batch_size = query_imgs.shape[0]
                    for mode_name, mode_box in {
                            'head3_only': oracle_diag['box3'], 'head4_only': oracle_diag['box4'],
                            'confidence': oracle_diag['confidence_box'], 'oracle': oracle_diag['oracle_box'],
                    }.items():
                        a50, a25, miou, center = eval_decoded_boxes(mode_box, ori_gt_bbox, args.img_size)
                        meters = oracle_meters(mode_name)
                        meters['acc50'].update(float(a50), batch_size)
                        meters['acc25'].update(float(a25), batch_size)
                        meters['miou'].update(float(miou), batch_size)
                        meters['center'].update(float(center), batch_size)
                    extras = {
                        'agreement': oracle_diag['agreement'].mean(),
                        'confidence_head3_ratio': oracle_diag['confidence_use3'].float().mean(),
                        'oracle_head3_ratio': oracle_diag['oracle_use3'].float().mean(),
                        'head3_better_ratio': (oracle_diag['iou3'] > oracle_diag['iou4']).float().mean(),
                        'head4_better_ratio': (oracle_diag['iou4'] > oracle_diag['iou3']).float().mean(),
                        'head_tie_ratio': ((oracle_diag['iou3'] - oracle_diag['iou4']).abs() < 1e-6).float().mean(),
                        'oracle_iou_gain': (oracle_diag['oracle_iou'] - oracle_diag['confidence_iou']).mean(),
                        'recover_acc50': ((oracle_diag['oracle_iou'] > 0.50) &
                                          (oracle_diag['confidence_iou'] <= 0.50)).float().mean(),
                        'recover_acc25': ((oracle_diag['oracle_iou'] > 0.25) &
                                          (oracle_diag['confidence_iou'] <= 0.25)).float().mean(),
                    }
                    for name, value in extras.items():
                        oracle_extra_meters[name].update(float(value), batch_size)
                    if args.hqs_oracle_csv:
                        for index in range(batch_size):
                            oracle_rows.append({
                                'sample_index': int(sample_index[index]),
                                'score3': float(oracle_diag['score3'][index]),
                                'score4': float(oracle_diag['score4'][index]),
                                'iou3': float(oracle_diag['iou3'][index]),
                                'iou4': float(oracle_diag['iou4'][index]),
                                'confidence_head': 3 if bool(oracle_diag['confidence_use3'][index]) else 4,
                                'oracle_head': 3 if bool(oracle_diag['oracle_use3'][index]) else 4,
                                'agreement': int(oracle_diag['agreement'][index]),
                                'oracle_iou_gain': float(oracle_diag['oracle_iou'][index] -
                                                         oracle_diag['confidence_iou'][index]),
                            })
            else:
                pred_anchor = prediction_output.view(prediction_output.shape[0], 9, 5,
                    prediction_output.shape[2], prediction_output.shape[3])
                _, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
                accu_list, accu_center, iou, _, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full,
                    best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5, 0.25])
        
        avg_accu50.update(accu_list[0], query_imgs.shape[0])
        avg_accu25.update(accu_list[1], query_imgs.shape[0])
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Accu50 {accu50.val:.4f} ({accu50.avg:.4f})\t' \
                'Accu25 {accu25.val:.4f} ({accu25.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(data_loader), batch_time=batch_time, \
                    accu50=avg_accu50, accu25=avg_accu25, miou=avg_iou, accu_c=avg_accu_center)
            print(print_str)
            logging.info(print_str)
    print(avg_accu50.avg, avg_accu25.avg, avg_iou.avg, avg_accu_center.avg)
    logging.info("%f, %f, %f, %f" % (avg_accu50.avg, avg_accu25.avg, float(avg_iou.avg), avg_accu_center.avg))
    if diagnostic_meters:
        diagnostic_text = 'MS decode diagnostics: ' + ', '.join(
            '{}={:.6f}'.format(name, meter.avg) for name, meter in sorted(diagnostic_meters.items())
        )
        print(diagnostic_text)
        logging.info(diagnostic_text)
    if args.hqs_oracle_diag:
        print('================ HQS ORACLE DIAGNOSTIC ================')
        print('Mode                  Acc@0.50   Acc@0.25   mIoU     Center Acc')
        for mode_name, label in (('head3_only', 'Head3 only'), ('head4_only', 'Head4 only'),
                                 ('confidence', 'Confidence Selector'), ('oracle', 'Oracle Selector')):
            meters = oracle_metric_meters[mode_name]
            print('{:<22} {:>8.2f}   {:>8.2f}   {:>6.2f}   {:>8.2f}'.format(
                label, 100.0 * meters['acc50'].avg, 100.0 * meters['acc25'].avg,
                100.0 * meters['miou'].avg, 100.0 * meters['center'].avg))
        confidence, oracle = oracle_metric_meters['confidence'], oracle_metric_meters['oracle']
        print('Oracle - Confidence: Acc@0.50 {:+.2f} pp, Acc@0.25 {:+.2f} pp, mIoU {:+.2f} pp, Center {:+.2f} pp'.format(
            100.0 * (oracle['acc50'].avg - confidence['acc50'].avg),
            100.0 * (oracle['acc25'].avg - confidence['acc25'].avg),
            100.0 * (oracle['miou'].avg - confidence['miou'].avg),
            100.0 * (oracle['center'].avg - confidence['center'].avg)))
        print('Selector diagnostics: agreement={:.2f}% confidence_head3={:.2f}% confidence_head4={:.2f}% '
              'oracle_head3={:.2f}% oracle_head4={:.2f}%'.format(
                  100.0 * oracle_extra_meters['agreement'].avg,
                  100.0 * oracle_extra_meters['confidence_head3_ratio'].avg,
                  100.0 * (1.0 - oracle_extra_meters['confidence_head3_ratio'].avg),
                  100.0 * oracle_extra_meters['oracle_head3_ratio'].avg,
                  100.0 * (1.0 - oracle_extra_meters['oracle_head3_ratio'].avg)))
        print('Head complementarity: head3_better={:.2f}% head4_better={:.2f}% tie={:.2f}% '
              'mean_oracle_iou_gain={:.4f} recover_acc50={:.2f}% recover_acc25={:.2f}%'.format(
                  100.0 * oracle_extra_meters['head3_better_ratio'].avg,
                  100.0 * oracle_extra_meters['head4_better_ratio'].avg,
                  100.0 * oracle_extra_meters['head_tie_ratio'].avg,
                  oracle_extra_meters['oracle_iou_gain'].avg,
                  100.0 * oracle_extra_meters['recover_acc50'].avg,
                  100.0 * oracle_extra_meters['recover_acc25'].avg))
        print('=======================================================')
        if args.hqs_oracle_csv:
            csv_directory = os.path.dirname(args.hqs_oracle_csv)
            if csv_directory:
                os.makedirs(csv_directory, exist_ok=True)
            with open(args.hqs_oracle_csv, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=(
                    'sample_index', 'score3', 'score4', 'iou3', 'iou4', 'confidence_head',
                    'oracle_head', 'agreement', 'oracle_iou_gain'))
                writer.writeheader()
                writer.writerows(oracle_rows)
            print('HQS Oracle CSV: {}'.format(args.hqs_oracle_csv))

    if args.hqs_v2a_rank_diag or args.hqs_v2b_rank_diag:
        rank_logits = torch.cat(rank_diag_logits, dim=0)
        box3_all, box4_all = torch.cat(rank_diag_box3, dim=0), torch.cat(rank_diag_box4, dim=0)
        gt_all = torch.cat(rank_diag_gt, dim=0)
        iou3_all, iou4_all = torch.cat(rank_diag_iou3, dim=0), torch.cat(rank_diag_iou4, dim=0)
        sample_all = torch.cat(rank_diag_sample_index, dim=0)
        gap = iou3_all - iou4_all
        valid = gap.abs() > args.hqs_rank_epsilon
        head3_better = gap > args.hqs_rank_epsilon
        head4_better = gap < -args.hqs_rank_epsilon
        oracle_use3 = gap > 0.0

        print('================ HQS-v2a RANK DIAGNOSTIC ================')
        logging.info('================ HQS-v2a RANK DIAGNOSTIC ================')
        print('Checkpoint diagnostic: epsilon={:.6f}, valid={:.2f}% ({}/{})'.format(
            args.hqs_rank_epsilon, 100.0 * valid.float().mean().item(), int(valid.sum().item()), len(valid)))
        logging.info('HQS-v2a rank diagnostic epsilon=%f valid=%d/%d',
                     args.hqs_rank_epsilon, int(valid.sum().item()), len(valid))
        describe_rank_logits('Head3-better', rank_logits[head3_better])
        describe_rank_logits('Head4-better', rank_logits[head4_better])
        describe_rank_logits('All', rank_logits)
        auc = binary_pairwise_auc(rank_logits[head3_better], rank_logits[head4_better])
        auc_text = 'ROC-AUC = {:.6f}'.format(auc)
        print(auc_text)
        logging.info(auc_text)

        threshold_min, threshold_max = rank_logits.min().item() - 1e-6, rank_logits.max().item() + 1e-6
        thresholds = torch.cat((torch.linspace(threshold_min, threshold_max, steps=201), torch.tensor([0.0])))
        thresholds, _ = torch.sort(torch.unique(thresholds))
        sweep_rows = []
        for threshold in thresholds:
            threshold_value = float(threshold)
            use3 = rank_logits >= threshold_value
            selected_box = torch.where(use3[:, None], box3_all, box4_all)
            acc50, acc25, miou, center = eval_decoded_boxes(selected_box, gt_all, args.img_size)
            agreement = float((use3[valid] == oracle_use3[valid]).float().mean()) if valid.any() else float('nan')
            sweep_rows.append({
                'threshold': threshold_value,
                'head3_ratio': float(use3.float().mean()),
                'head4_ratio': float((~use3).float().mean()),
                'acc50': float(acc50), 'acc25': float(acc25), 'miou': float(miou),
                'center_acc': float(center), 'oracle_agreement_valid': agreement,
            })
        zero_row = next(row for row in sweep_rows if abs(row['threshold']) < 1e-12)
        online_metrics = (float(avg_accu50.avg), float(avg_accu25.avg), float(avg_iou.avg), float(avg_accu_center.avg))
        zero_metrics = (zero_row['acc50'], zero_row['acc25'], zero_row['miou'], zero_row['center_acc'])
        if abs(args.hqs_rank_threshold) < 1e-12 and max(abs(a - b) for a, b in zip(online_metrics, zero_metrics)) > 1e-6:
            raise RuntimeError('HQS-v2a threshold=0 sweep does not reproduce validation metrics: '
                               'online={} sweep={}'.format(online_metrics, zero_metrics))
        best_row = max(sweep_rows, key=lambda row: (
            round(row['acc50'], 12), round(row['miou'], 12), -abs(row['threshold'])))
        print('Threshold=0: Head3={:.2f}% Head4={:.2f}% Acc@0.50={:.2f}% Acc@0.25={:.2f}% '
              'mIoU={:.2f}% Center={:.2f}% OracleAgree={:.2f}%'.format(
                  100.0 * zero_row['head3_ratio'], 100.0 * zero_row['head4_ratio'],
                  100.0 * zero_row['acc50'], 100.0 * zero_row['acc25'], 100.0 * zero_row['miou'],
                  100.0 * zero_row['center_acc'], 100.0 * zero_row['oracle_agreement_valid']))
        print('BEST VALIDATION THRESHOLD: threshold={:.8f} Head3={:.2f}% Head4={:.2f}% '
              'Acc@0.50={:.2f}% Acc@0.25={:.2f}% mIoU={:.2f}% Center={:.2f}% OracleAgree={:.2f}%'.format(
                  best_row['threshold'], 100.0 * best_row['head3_ratio'], 100.0 * best_row['head4_ratio'],
                  100.0 * best_row['acc50'], 100.0 * best_row['acc25'], 100.0 * best_row['miou'],
                  100.0 * best_row['center_acc'], 100.0 * best_row['oracle_agreement_valid']))
        print('Delta best-threshold vs threshold=0: Acc@0.50={:+.2f} pp mIoU={:+.2f} pp'.format(
            100.0 * (best_row['acc50'] - zero_row['acc50']), 100.0 * (best_row['miou'] - zero_row['miou'])))
        logging.info('HQS-v2a threshold=0 %s', zero_row)
        logging.info('HQS-v2a best threshold %s', best_row)

        for output_path, fieldnames, rows in (
                (args.hqs_v2a_rank_diag_prefix + '_val_samples.csv',
                 ('sample_index', 'rank_logit', 'iou3', 'iou4', 'iou_gap', 'valid', 'label'),
                 ({
                     'sample_index': int(sample_all[index]), 'rank_logit': float(rank_logits[index]),
                     'iou3': float(iou3_all[index]), 'iou4': float(iou4_all[index]), 'iou_gap': float(gap[index]),
                     'valid': int(valid[index]),
                     'label': 'head3_better' if bool(head3_better[index]) else
                              ('head4_better' if bool(head4_better[index]) else 'ignored'),
                 } for index in range(rank_logits.numel()))),
                (args.hqs_v2a_rank_diag_prefix + '_threshold_sweep.csv',
                 ('threshold', 'head3_ratio', 'head4_ratio', 'acc50', 'acc25', 'miou', 'center_acc',
                  'oracle_agreement_valid'), sweep_rows)):
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print('HQS-v2a rank diagnostic CSV: {}'.format(output_path))
        print('==========================================================')

    return avg_accu50.avg


if __name__ == "__main__":
    main()
