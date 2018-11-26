import torch
import torch.nn as nn
import numpy as np
import os
from os.path import join, exists
import argparse
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import ImageFolderWithPaths
import time
from pathlib import Path

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
    parser.add_argument('--datadir','-d', help='datadir')
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    traindir = os.path.join(args.datadir, 'train')
    trainset = ImageFolderWithPaths(traindir, transform_train)
    loader = data.DataLoader(
        trainset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True)

    batch_time = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets, paths) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = feat_model(inputs)
            for j in range(inputs.size(0)):
                path = paths[j]
                outfile = path.replace('train', 'feat/train').replace('JPEG','npz')
                p = Path(outfile)
                p.parent.mkdir(parents=True, exist_ok = True)
                p.touch(exist_ok=True)
                out = np.array(outputs[j]).squeeze()
                if exists(outfile):
                    continue
                np.savez(outfile, out)
            batch_time.update(time.time() - end)
            end = time.time()
            print(i)

def main():
    args = parse_arguments()
    extract_feats(args)

if __name__ == '__main__':
    main()
