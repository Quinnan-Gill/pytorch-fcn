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


here = osp.dirname(osp.abspath(__file__))


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
    parser.add_argument('-o', '--out', type=str, default=here)
    parser.add_argument('--height', default=366, type=int)
    parser.add_argument('--width', default=500, type=int)
    parser.add_argument('--fcn', default='fcn32s', choices=['fcn32s', 'fcn16s', 'fcn8s'])
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    args = parser.parse_args()


    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(args.out, args.fcn, now.strftime('%Y%m%d_%H%M%S.%f'))

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
    resize = (args.height, args.width)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True, resize=resize),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = fcn_models[args.fcn](n_class=21)
    args.pretrained_model = fcn_models[args.fcn].download()
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
        max_epochs=5,
        interval_validate=4000,
    )
    trainer.validate()


if __name__ == '__main__':
    main()
