# -*- coding:utf8 -*-

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F


def _unwrap_data_parallel_state_dict(state_dict):
    """Make DataParallel checkpoints loadable by the single-GPU model."""
    if state_dict and all(key.startswith('module.') for key in state_dict):
        return {key[len('module.'):]: value for key, value in state_dict.items()}
    return state_dict

def save_checkpoint(state, is_best, args, filename='default'):
    if filename=='default':
        filename = 'filmconv_nofpn32_%s_batch%d'%(args.dataset,args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        checkpoint_head = checkpoint.get('detector_head')
        if checkpoint_head and checkpoint_head != args.detector_head:
            message = ("=> checkpoint head '{}' differs from requested '{}'; "
                       "only shape-compatible shared weights will be loaded"
                       .format(checkpoint_head, args.detector_head))
            print(message)
            logging.warning(message)
        pretrained_dict = _unwrap_data_parallel_state_dict(checkpoint['state_dict'])
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        logging.info("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model

def load_resume(model, args, logging):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(_unwrap_data_parallel_state_dict(checkpoint['state_dict']))
        print(("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss)))
        logging.info("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model
