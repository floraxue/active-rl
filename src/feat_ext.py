import torch
import torch.nn as nn
import numpy as np
import os
from os.path import exists, join
import argparse
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import ImageData
import time
import pickle
from pathlib import Path
from util import logger
from train_new import IMAGE_DIR_TRAIN, GT_PATH


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet101',
                        help='name of the pretrained model')
    parser.add_argument('--batch-size', '-b', type=int, default=400,
                        help='batch size')
    parser.add_argument('--num-workers', '-n', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='printing frequency')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='cropping size')
    parser.add_argument('--key-path','-k',
                        default='/data3/floraxue/cs294/active-rl-data/pool/cat_train_keys.p',
                        help='train key path')
    parser.add_argument('--gpu-id','-g', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def extract_feats(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.__dict__[args.model](pretrained=True)
    feat_model = nn.Sequential(*list(model.children())[:-1])
    feat_model = torch.nn.DataParallel(feat_model).to(device)

    normalize = transforms.Normalize(mean=[0.49911337, 0.46108112, 0.42117174],
                                     std=[0.29213443, 0.2912565, 0.2954716])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = ImageData(args.key_path, IMAGE_DIR_TRAIN, GT_PATH, transform_train)
    loader = data.DataLoader(
        trainset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True)

    batch_time = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets, keys) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = feat_model(inputs)
            for j in range(inputs.size(0)):
                k = keys[j]
                path = join(IMAGE_DIR_TRAIN, k + '.jpg')
                outfile = path.replace('images', 'feats').replace('jpg','npz')
                p = Path(outfile)
                p.parent.mkdir(parents=True, exist_ok = True)
                p.touch(exist_ok=True)
                out = np.array(outputs[j]).squeeze()
                np.savez(outfile, out)
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info("Finished batch {}/{}".format(i, len(loader)))


def main():
    args = parse_arguments()
    extract_feats(args)


if __name__ == '__main__':
    main()
