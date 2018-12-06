import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import argparse
from dataset import ImageData
import network
from util import logger, softmax
from env_obj import Env
import os
from os.path import join, exists
import shutil
import pdb
import torch.nn.functional as F
import numpy as np
import json
import pickle


env = Env()
IMAGE_DIR_TRAIN = '/data/active-rl-data/data/images/train/cat'
IMAGE_DIR_FIXED = '/data/active-rl-data/data/images/fixed/cat'
IMAGE_DIR_HOLDOUT = '/data/active-rl-data/data/images/holdout/cat'
GT_PATH = '/data/active-rl-data/ground_truth/cat_gt_cached.p'


def args_parser():
    parser = argparse.ArgumentParser(description='Pytorch training')
    parser.add_argument('cmd', type=str, choices=['train', 'test', 'test_all'])
    parser.add_argument('--train-keys', '-t', metavar='DIR',
                        help='path to training dataset')
    parser.add_argument('--eval-keys', '-e', metavar='DIR',
                        help='path to eval datasets')
    parser.add_argument('--save-dir', '-s', metavar='DIR', help='save dir')
    parser.add_argument('--method', '-m', metavar='ARCH', default='resnet',
                        help='model architecture:')
    parser.add_argument('--category', '-ct', default='cat1', type=str,
                        help='category')
    parser.add_argument('--train-prefix', '-tp', default='', type=str,
                        help='train prefix')
    parser.add_argument('--trial', '-tt', default=0, type=int,
                        help='current trial id')
    parser.add_argument('--stage', '-st', default=0, type=int,
                        help='stage id')
    parser.add_argument('--test-prefix', default='train', type=str,
                        help='choose data to split')
    parser.add_argument('--model-file', default='', type=str,
                        help='trained model checkpoint')

    # flags for training
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--iters', default=15000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-c', '--crop-size', default=224, type=int,
                        metavar='N', help='image crop size (default: 224)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of workers')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=1000)
    args = parser.parse_args()

    return args


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = network.__dict__[args.method]()
    model = torch.nn.DataParallel(model).to(device)

    #NEW load state_dict
    model_path = join(args.save_dir, 'bal_model_best.pth.tar')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.train()
    logger.info('model {} has been initialized'.format(args.method))

    mean = np.array([0.49911337, 0.46108112, 0.42117174])
    std = np.array([0.29213443, 0.2912565, 0.2954716])

    normalize = transforms.Normalize(mean=mean, std=std)
    trans = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    logger.info('creating train/val data loader')
    # train_prefix = args.train_prefix + '_train'
    train_loader = data.DataLoader(
        ImageData(args.train_keys, IMAGE_DIR_TRAIN, GT_PATH,
                  transform=trans), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, pin_memory=True)
    # val_loader = data.DataLoader(
    #     ImageData(args.eval_keys, IMAGE_DIR, GT_PATH, transform=trans),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers, shuffle=False, pin_memory=True)

    val_bal_loader = data.DataLoader(
        ImageData(args.eval_dir, IMAGE_DIR_TRAIN, GT_PATH, transform=trans),
        batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    logger.info('finished creating data loaders')

    optim_params = model.module.optim_parameters()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(optim_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # best_prec1_raw = 0
    best_prec1_bal = 0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # actual_iters = args.iters // len(train_loader) * len(train_loader)
    # logger.info('actual training iters = {}'.format(actual_iters))
    # args.iters = actual_iters

    for start in range(0, args.iters, len(train_loader)):
        end = time.time()
        for j, (input, target, _, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            i = j + start
            adjust_learning_rate(args, optimizer, i)
            # switch to train mode
            model.train()
            input = input.to(device)
            target = target.to(device)

            # compute output
            # no need to call variable in PyTorch 0.4
            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (i+1) % args.print_freq == 0:
                logger.info('Iter: [{}/{}]\t'
                            'Time {batch_time.val:.3f} '
                            '({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                i+1, args.iters, batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1))

            if (i+1) % args.eval_every == 0 or (i+1) == args.iters:
                # evaluation
                # prec1_raw = validate(args.print_freq, val_loader, model, criterion, i)
                prec1_bal = validate(args.print_freq, val_bal_loader, model, criterion, i)

                # is_best_raw = prec1_raw > best_prec1_raw
                is_best_bal = prec1_bal > best_prec1_bal

                # best_prec1_raw = max(prec1_raw, best_prec1_raw)
                best_prec1_bal = max(prec1_bal, best_prec1_bal)

                checkpoint_path = join(args.save_dir,
                                       'checkpoint_latest.pth.tar')
                save_checkpoint({
                    'iters': i + 1,
                    'arch': args.method,
                    'state_dict': model.state_dict(),
                    # 'best_prec1': [best_prec1_raw, best_prec1_bal],
                    'best_prec1': [best_prec1_bal],
                    'optimizer': optimizer.state_dict(),
                # }, [is_best_raw, is_best_bal], filename=checkpoint_path)
                }, [is_best_bal], filename=checkpoint_path)
                history_path = join(
                    args.save_dir,
                    'checkpoint_{:05d}.pth.tar'.format(i + 1))
                shutil.copyfile(checkpoint_path, history_path)

            end = time.time()

            if (i+1) == args.iters:
                break


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    # if is_best[0]:  # raw data
    #     shutil.copyfile(filename, join(dirname, 'raw_model_best.pth.tar'))
    if is_best[0]:  # bal data
        shutil.copyfile(filename, join(dirname, 'bal_model_best.pth.tar'))


def adjust_learning_rate(args, optimizer, iter):
    """ Sets the learning rate to the initial LR decayed by
    10 every 30 epochs """
    # lr = args.lr * (0.1 ** (epoch // 30))
    if 10000 <= iter < 20000:
        lr = args.lr * 0.1
    elif iter > 20000:
        lr = args.lr * 0.01
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(print_freq, val_loader, model, criterion, _iter, for_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    scores, labels, keys = [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # switch to evaluate mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, key) in enumerate(val_loader):
            # target = target.cuda(non_blocking=True)
            input = input.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            if for_test:
                softmax_ = F.softmax(output, dim=-1)
                if len(softmax_.size()) == 1:
                    softmax_ = softmax_.reshape((1, softmax_.size(0)))
                score = softmax_[:, 1]
                scores.append(score)
                keys += key
                labels.append(target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Val: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} '
                            '({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                i, len(val_loader), batch_time=batch_time,
                                loss=losses, top1=top1))

        logger.info('[Iter {}] * Prec@1 {top1.avg:.3f}'.format(
            _iter+1, top1=top1))

    if for_test:
        scores = torch.cat(scores)
        labels = torch.cat(labels)
        return top1.avg, scores, keys, labels
    else:
        return top1.avg


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = network.__dict__[args.method]()
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    logger.info('model {} has been initialized'.format(args.method))

    mean = np.array([0.49911337, 0.46108112, 0.42117174])
    std = np.array([0.29213443, 0.2912565, 0.2954716])

    normalize = transforms.Normalize(mean=mean, std=std)
    trans = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    # data loader
    # the first stage will be using the full test data and for the later
    # stages, use the unknown part of the data for testing
    test_loader = data.DataLoader(
        ImageData(args.eval_keys, IMAGE_DIR_TRAIN, GT_PATH,
                  transform=trans), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    # find the split
    # test_names = ['raw', 'bal']
    test_names = ['bal']
    split_results = []
    accuracies = []
    work_dir = os.path.dirname(args.save_dir)
    for i, tn in enumerate(test_names):
        logger.info('Try on the best model evaluated on {} data'.format(tn))
        model_path = join(args.save_dir, '{}_model_best.pth.tar'.format(tn))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint['iters']

        prec1, scores, keys, labels = validate(
            args.print_freq, test_loader, model, criterion, iteration, for_test=True)

        accuracies.append(prec1.item())
        test_labels = np.array(labels)
        test_scores = np.array(scores)

        order = np.argsort(test_scores)[::-1]
        ordered_labels = test_labels[order]
        ordered_scores = test_scores[order]
        precisions = np.cumsum(ordered_labels) / np.arange(
            1, len(ordered_labels) + 1)
        recalls = np.cumsum(ordered_labels) / np.sum(test_labels)
        good_precisions = precisions > 0.95
        good_recall = recalls > 0.99
        try:
            pos_cut = int(np.nonzero(good_precisions)[0][-1])
            pos_thresh = float(ordered_scores[pos_cut])
        except IndexError:
            pos_cut = 0
            pos_thresh = 2
        logger.info('Pos cut: %d %f score: %f',
                    pos_cut, pos_cut / len(test_labels), pos_thresh)

        try:
            neg_cut = int(np.nonzero(good_recall)[0][0])
            neg_thresh = float(ordered_scores[neg_cut])
        except IndexError:
            neg_cut = len(test_labels)
            neg_thresh = -1
        recall_percentages = [i / 10. for i in range(1, 10)]
        recall_cuts = [int(np.nonzero(recalls > p)[0][0]) for p in
                       recall_percentages]
        recall_thresholds = [float(ordered_scores[c]) for c in recall_cuts]
        logger.info('Neg cut: %d %f score: %f',
                    neg_cut, neg_cut / len(test_labels), neg_thresh)

        result = {'posCut': pos_cut, 'posRatio': pos_cut / len(test_labels),
                  'posThresh': pos_thresh,
                  'negCut': neg_cut, 'negRatio': neg_cut / len(test_labels),
                  'negThresh': neg_thresh,
                  'recallCuts': recall_cuts,
                  'recallThresholds': recall_thresholds,
                  'recallPercentages': recall_percentages,
                  'total': len(test_labels),
                  'iteration': iteration,
                  'method': args.method,
                  'category': args.category,
                  'crops': 1,
                  'trials': args.trial_prefix}

        out_path = join(work_dir, 'split_{}_{}.json'.format(tn, iteration))
        with open(out_path, 'w') as fp:
            json.dump(result, fp)

        out_path = join(work_dir, 'test_predictions_{}_{}.txt'.format(
            tn, iteration))
        with open(out_path, 'w') as fp:
            print('key,label,prediction,score', file=fp)
            for i in range(len(keys)):
                index = order[i]
                p = int(test_scores[index] > pos_thresh) + \
                    int(test_scores[index] > neg_thresh)
                tar = int(labels[index])
                if tar == 0:
                    tar = -1
                print(keys[index], tar, p, test_scores[index],
                      sep=',', file=fp)

        split_results.append(result)

    # select best split
    best_split = -1
    best_ratio = 1
    for i, s in enumerate(split_results):
        ratio = s['negRatio'] - s['posRatio']
        s['name'] = test_names[i]
        s['accuracy'] = accuracies[i]
        if ratio < best_ratio:
            best_split = s
            best_ratio = ratio

    best_split = best_split.copy()
    best_split['splitResults'] = split_results
    out_path = join(work_dir, 'split.json')

    with open(out_path, 'w') as fp:
        json.dump(best_split, fp)

    model_path = join(args.save_dir, '{}_model_best.pth.tar'.format(
        best_split['name']))
    out_model_link = join(work_dir, 'split_model.pth.tar')
    if exists(out_model_link):
        os.remove(out_model_link)
    os.symlink(model_path, out_model_link)

    # if env.mode() != 'LSUN':
    #     pred_path = join(work_dir, 'test_predictions_{}_{}.txt'.format(
    #         best_split['name'], best_split['iteration']))
    #     test_keys = [[] for _ in range(3)]
    #     test_gts = [[] for _ in range(3)]
    #     test_dbs = [[] for _ in range(3)]
    #     with open(pred_path, 'r') as fp:
    #         for line in fp:
    #             fields = line.split(',')
    #             try:
    #                 score = float(fields[3].strip())
    #             except Exception:
    #                 continue
    #             label = int(score > best_split['posThresh']) \
    #                 + int(score > best_split['negThresh'])
    #             test_keys[label].append(fields[0])
    #             test_gts[label].append(fields[1])
    #             test_dbs[label].append(fields[-1])
    #
    #     trials = [int(t) for t in args.trial_prefix.split('_')[1:]]
    #
    #     for i in range(3):
    #         key_path = env.label_result_key_path(
    #             args.category, args.stage, trials, i, prefix='test')
    #         gt_path = env.label_result_gt_path(
    #             args.category, args.stage, trials, i, prefix='test')
    #         db_path = env.label_result_db_path(
    #             args.category, args.stage, trials, i, prefix='test')
    #         logger.info('Writing %s', key_path)
    #         with open(key_path, 'w') as fp:
    #             for k in test_keys[i]:
    #                 print(k, file=fp)
    #         logger.info('Writing %s', gt_path)
    #         with open(gt_path, 'w') as fp:
    #             for k, g in zip(test_keys[i], test_gts[i]):
    #                 print(k.strip() + ' ' + g, file=fp)
    #         logger.info('Writing %s', db_path)
    #         with open(db_path, 'w') as fp:
    #             for k in test_dbs[i]:
    #                 print(k.strip(), file=fp)
    #     logger.info('# images: %d %d %d', len(test_keys[0]), len(test_keys[1]),
    #                 len(test_keys[2]))


def test_fixed_set(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = network.__dict__[args.method]()
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    logger.info('model {} has been initialized'.format(args.method))

    mean = np.array([0.49911337, 0.46108112, 0.42117174])
    std = np.array([0.29213443, 0.2912565, 0.2954716])

    normalize = transforms.Normalize(mean=mean, std=std)
    trans = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    # data loader
    test_loader = data.DataLoader(
        ImageData(args.eval_keys, IMAGE_DIR_FIXED, GT_PATH,
                  transform=trans), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    # run on the fixed set
    work_dir = os.path.dirname(args.save_dir)
    model_file = join(work_dir, 'split_model.pth.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    _iter = checkpoint['iters']
    logger.info('loaded model {} at iteration {}'.format(args.method, _iter))

    split_file = join(work_dir, 'split.json')
    with open(split_file, 'r') as fp:
        split_info = json.load(fp)

    pos_thresh = split_info['posThresh']
    neg_thresh = split_info['negThresh']

    prec1, scores, keys, labels = validate(
        args.print_freq, test_loader, model, criterion, _iter, for_test=True)

    test_labels = np.array(labels)
    test_scores = np.array(scores)

    overall_correct = 0

    for index in range(len(keys)):
        # p = 0 negative, p = 2 positive
        p = int(test_scores[index] > pos_thresh) + \
            int(test_scores[index] > neg_thresh)
        tar = int(test_labels[index])
        if tar == -1:
            tar = 0
        if tar == 1:
            tar = 2
        if p == tar:
            overall_correct += 1

    overall_acc = overall_correct / len(keys)
    mode = env.mode()
    logger.info("env mode is {}".format(mode))
    out_file = join(work_dir, 'fixed_set_acc_{}.p'.format(mode))
    pickle.dump(overall_acc, open(out_file, 'wb'))
    return overall_acc


def test_all(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = network.__dict__[args.method]()
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    work_dir = os.path.dirname(args.save_dir)
    model_file = join(work_dir, 'split_model.pth.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    _iter = checkpoint['iters']
    logger.info('loaded model {} at iteration {}'.format(args.method, _iter))

    mean = np.array([0.49911337, 0.46108112, 0.42117174])
    std = np.array([0.29213443, 0.2912565, 0.2954716])

    normalize = transforms.Normalize(mean=mean, std=std)
    trans = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    args.batch_size = args.batch_size

    test_loader = data.DataLoader(
        ImageData(args.eval_dir, IMAGE_DIR_TRAIN, GT_PATH,
                  transform=trans), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # result_dir = join(work_dir, 'split_results')
    # os.makedirs(result_dir, exist_ok=True)
    split_file = join(work_dir, 'split.json')
    with open(split_file, 'r') as fp:
        split_info = json.load(fp)

    pos_thresh = split_info['posThresh']
    neg_thresh = split_info['negThresh']
    # label_dir = env.label_dir(args.category)
    # TODO need to setup args properly with episode
    category = args.category
    trial = args.trial

    # out_file = join(result_dir, args.test_prefix+'_score.txt')
    # if exists(out_file):
    #     os.remove(out_file)
    #     logger.info('delete existing out_file {}'.format(out_file))
    #
    # for i in range(3):
    #     key_path = env.label_result_key_path(category, stage, trials, i)
    #     gt_path = env.label_result_gt_path(category, stage, trials, i)
    #     if exists(key_path):
    #         os.remove(key_path)
    #         logger.info('remove existing key_path {}'.format(key_path))
    #     if exists(gt_path):
    #         os.remove(gt_path)
    #         logger.info('remove existing gt_path {}'.format(gt_path))

    batch_time = AverageMeter()
    scores, keys, labels = [], [], []
    with torch.no_grad():
        end = time.time()
        for i,  (input, target, key) in enumerate(test_loader):
            input = input.to(device)
            output, _ = model(input)
            softmax_ = F.softmax(output, dim=-1)

            if len(softmax_.size()) == 1:
                softmax_ = softmax_.reshape((1, softmax_.size(0)))
            score = softmax_[:, 1]
            scores.append(score)
            keys += key
            labels.append(target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('{}/{} batches have been evaluated, '
                            'batch time = {bt.val:.3f}s '
                            '({bt.avg:.3f})s'.format(
                                i, len(test_loader), bt=batch_time))
            # if (i+1) % 100 == 0 or (i+1) == len(test_loader):
            #     logger.info('[{}] Writing {}'.format((i+1)//100, out_file))
            #     scores = torch.cat(scores)
            #     labels = torch.cat(labels)
            #     write2file(keys, scores, labels)
            #     logger.info('write to file and release memory')
            #     scores, keys, labels = [], [], []

    logger.info('finish evaluation all data')

    # Write to file
    split_keys = [[] for _ in range(3)]
    split_gts = [[] for _ in range(3)]
    out_pos_path = join(work_dir, '{}_trial_{}_pos.txt'.format(category, trial))
    out_unsure_path = join(work_dir, '{}_trial_{}_unsure.txt'.format(category, trial))
    out_unsure_pickle = join(work_dir, '{}_trial_{}_unsure.p'.format(category, trial))
    out_neg_path = join(work_dir, '{}_trial_{}_neg.txt'.format(category, trial))
    for i in range(len(keys)):
        k = keys[i]
        score = float(scores[i].item())
        lab = int(score > pos_thresh) + int(score > neg_thresh)
        split_keys[lab].append(k)
        tar = int(labels[i].item())
        if tar == 0:
            tar = -1
        split_gts[lab].append(str(tar))

    with open(out_pos_path, 'w') as fp:
        for k in split_keys[2]:
            fp.write(k + '\n')
    with open(out_unsure_path, 'w') as fp:
        for k in split_keys[1]:
            fp.write(k + '\n')
    with open(out_neg_path, 'w') as fp:
        for k in split_keys[0]:
            fp.write(k + '\n')
    with open(out_unsure_pickle, 'wb') as fp:
        pickle.dump(split_keys[1], fp)

    logger.info("finished writing results for current test all")

    # for jj in range(3):
    #     key_path = env.label_result_key_path(category, stage, trials, jj)
    #     gt_path = env.label_result_gt_path(category, stage, trials, jj)
    #     logger.info('Writing %s', key_path)
    #     with open(key_path, 'a') as fp:
    #         for k in split_keys[jj]:
    #             print(k, file=fp)
    #
    #     logger.info('Writing %s', gt_path)
    #     with open(gt_path, 'a') as fp:
    #         for k, g in zip(split_keys[jj], split_gts[jj]):
    #             print(k + ' ' + g, file=fp)
    #     img_cnt[jj] += len(split_keys[jj])

    # out_split_path = env.label_result_info_path(category, stage, trials)
    # logger.info('Moving split info %s', out_split_path)
    # shutil.copyfile(split_file, out_split_path)
    # logger.info('# images: %d %d %d',
    #             img_cnt[0], img_cnt[1], img_cnt[2])


if __name__ == '__main__':
    args = args_parser()
    if args.cmd == 'train':
        logger.info('running training')
        os.makedirs(args.save_dir, exist_ok=True)
        train(args)
    elif args.cmd == 'test':
        logger.info('running testing')
        test(args)
    elif args.cmd == 'test_fixed':
        logger.info('running test fixed set')
        test_fixed_set(args)
    elif args.cmd == 'test_all':
        logger.info('running testing all')
        test_all(args)
    else:
        raise NotImplementedError
