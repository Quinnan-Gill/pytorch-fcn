#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import torch
import yaml

import torchfcn


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('-o', '--out', type=str, default=here)
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    args = parser.parse_args()

    args.model = 'FCN32s'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(args.out, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = None
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN32s(n_class=21)
    args.pretrained_model = torchfcn.models.FCN32s.download()
    state_dict = torch.load(args.pretrained_model)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict['model_state_dict'])

    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = None

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=None,
        interval_validate=4000,
    )
    trainer.validate()


if __name__ == '__main__':
    main()
