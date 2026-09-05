# -*- coding: utf8 -*-

import os
import sys
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2

from torch.autograd import Variable
from torch.utils.data import DataLoader, get_worker_info
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader import RSDataset
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
from model.loss import yolo_loss, build_target, adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc
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

    if args.b_variant != 'none':
        model = DetGeoB(emb_size=args.emb_size, leaky=True, variant=args.b_variant,
                         num_tokens=args.b_num_tokens, num_heads=args.b_num_heads, ffn_dim=args.b_ffn_dim)
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
                         augment=True, **prompt_kwargs)
    val_dataset = dataset_class(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='val',
                         img_size = args.img_size,
                         transform=input_transform, **prompt_kwargs)
    test_dataset = dataset_class(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='test',
                         img_size = args.img_size,
                         transform=input_transform, **prompt_kwargs)
    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=True,
                         drop_last=False, num_workers=args.num_workers)
    if args.original_rng_matched or args.standard_rng:
        # P10 keeps its model-init RNG isolation; standard mode deliberately does not.
        # Both retain DetGeo's ordinary DataLoader construction.
        # P10 deliberately reproduces DetGeo's original default DataLoader RNG.
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
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
    if args.adaptive_gaussian_field:
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

    if args.sam_prompt or args.adaptive_sam_prompt or args.sam_refined_pe or args.rgbp_interaction or args.hisym_pae or args.adaptive_gaussian_field:
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
    optimizer = torch.optim.RMSprop(optimizer_groups, weight_decay=0.0005)

    if not args.original_rng_matched and not args.standard_rng:
        # P09: SAM creates extra PromptFusion parameters. Reset runtime RNG
        # after model/optimizer construction so later training randomness matches.
        seed_global_rng(args.runtime_seed)
    
    ## training and testing
    best_accu = -float('Inf')
    
    if args.test:
        _ = test_epoch(test_loader, model, args)
    elif args.val:
        _ = test_epoch(val_loader, model, args)
    else:
        for epoch in range(args.max_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            gc.collect()
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


def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_cls_losses = AverageMeter()
    avg_geo_losses = AverageMeter()
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

        pred_anchor, _ = forward_model(model, query_imgs, rs_imgs, mat_clickxy, prompt_maps)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        ## convert gt box to center+offset format
        new_gt_bbox, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        # loss
        loss_geo, loss_cls = yolo_loss(pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, args.img_size)
        loss = loss_cls + loss_geo * args.beta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_losses.update(loss.item(), query_imgs.shape[0])
        avg_geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        avg_cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        
        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5])
        accu = accu_list[0]
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
            print(print_str)
            logging.info(print_str)

def test_epoch(data_loader, model, args):
    batch_time = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()
    
    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    #print(datetime.datetime.now())
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    for batch_idx, batch in enumerate(data_loader):
        query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _, prompt_maps = unpack_batch(batch, args)
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        prompt_maps = tuple(prompt_map.cuda() for prompt_map in prompt_maps)
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        with torch.no_grad():
            pred_anchor, attn_score = forward_model(model, query_imgs, rs_imgs, mat_clickxy, prompt_maps)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0],\
            9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        _, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        accu_list, accu_center, iou, each_acc_list, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5, 0.25])
        
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

    return avg_accu50.avg


if __name__ == "__main__":
    main()
