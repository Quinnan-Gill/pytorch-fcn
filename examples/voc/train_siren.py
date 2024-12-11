#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import torchfcn

from train_fcn32s import get_parameters
from train_fcn32s import git_hash


here = osp.dirname(osp.abspath(__file__))


def get_siren_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, torchfcn.models.Siren):
            if bias:
                yield m.bias
            else:
                yield m.weight

def freeze_non_siren_params(model):
    for m in model.modules():
        if not isinstance(m, torchfcn.models.Siren):
            if hasattr(m, 'weight'):
                m.weight.requires_grad = False
    return model

def main():
    fcn_models = {
        'fcn32s': torchfcn.models.FCN32s,
        'fcn16s': torchfcn.models.FCN16s,
        'fcn8s': torchfcn.models.FCN8s,
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument('--fcn', default='fcn32s', choices=['fcn32s', 'fcn16s', 'fcn8s'])
    parser.add_argument('--height', default=366, type=int)
    parser.add_argument('--width', default=500, type=int)
    parser.add_argument('--filters', type=str, default=None, help='comma seperated list of all the ReLUs to convert to SIRENs')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--epochs', type=int, default=5, help='# of epochs to train'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-12, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument('-o', '--out', type=str, default=here)
    args = parser.parse_args()

    args.model = 'Siren'
    args.git_hash = git_hash()
    args.filters = args.filters.split(',') if args.filters is not None

    now = datetime.datetime.now()
    args.out = osp.join(args.out, 'siren', args.fcn, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    resize = (args.height, args.width)
    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SBDClassSeg(root, split='train', transform=True, resize=resize),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True, resize=resize),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    start_epoch = 0
    start_iteration = 0

    model = None

    if args.resume:
        fcn_model = fcn_models[args.fcn]()
        model = torchfcn.models.SirenFCN(
            fcn_model, args.height, args.width, filters=args.filters)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn_model = fcn_models[args.fcn]()
        pretrained_model = fcn_models[args.fcn].download()
        state_dict = torch.load(pretrained_model)
        try:
            fcn_model.load_state_dict(state_dict)
        except RuntimeError:
            fcn_model.load_state_dict(state_dict['model_state_dict'])
        model = torchfcn.models.SirenFCN(
            fcn_model, args.height, args.width, filters=args.filters)

    if cuda:
        model = model.cuda()

    model = freeze_non_siren_params(model)
    # 3. optimizer

    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_epochs=args.epochs,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
