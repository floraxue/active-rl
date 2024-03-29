from os.path import join
import pickle
import numpy as np


MACHINE_ROOT = '/data3/floraxue/cs294/exp/lsun_pretrained/machine_labels'

accs = []
corrects = []
pos_reads = []
pos_corrects = []
neg_reads = []
neg_corrects = []
print('trial', 'pos_reads', 'neg_reads', 'unsure', 'pos_corrects', 'neg_corrects', 'corrects', 'acc')

for trial in range(1,6):

    pos_path = join(MACHINE_ROOT, 'cat_trial_{}_pos.txt'.format(trial))
    neg_path = join(MACHINE_ROOT, 'cat_trial_{}_neg.txt'.format(trial))
    unsure_path = join(MACHINE_ROOT, 'cat_trial_{}_unsure.txt'.format(trial))
    GT_PATH = '/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached.p'

    dic = pickle.load(open(GT_PATH, 'rb'))

    correct = 0
    pos_correct = 0
    neg_correct = 0

    with open(pos_path, 'r') as fp:
        lines = fp.readlines()
        pos_read = len(lines)
        for line in lines:
            key = line.strip()
            target = dic[key]
            if target == 1:
                correct += 1
                pos_correct += 1

    with open(neg_path, 'r') as fp:
        lines = fp.readlines()
        neg_read = len(lines)
        for line in lines:
            key = line.strip()
            target = dic[key]
            if target == -1:
                correct += 1
                neg_correct += 1

    with open(unsure_path, 'r') as fp:
        lines = fp.readlines()
        total = len(lines)

    acc = correct / (pos_read + neg_read)

    accs.append(acc)
    corrects.append(correct)
    pos_reads.append(pos_read)
    pos_corrects.append(pos_correct)
    neg_reads.append(neg_read)
    neg_corrects.append(neg_correct)
    print(trial, pos_read,  neg_read, total, pos_correct,neg_correct, correct, acc)


all_correct = np.sum(corrects)
all_reads = np.sum(pos_reads) + np.sum(neg_reads)

print("Final result")
print("acc", all_correct / all_reads)
print("pos all", np.sum(pos_reads))
print("pos correct", np.sum(pos_corrects))
print("neg all", np.sum(neg_reads))
print("neg correct", np.sum(neg_corrects))
