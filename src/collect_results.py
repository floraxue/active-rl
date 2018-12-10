from os.path import join
import pickle


MACHINE_ROOT = '/data/active-rl-data/machine_labels'

trial = 1

pos_path = join(MACHINE_ROOT, 'cat_trial_{}_pos.txt'.format(trial))
neg_path = join(MACHINE_ROOT, 'cat_trial_{}_neg.txt'.format(trial))

GT_PATH = '/data/active-rl-data/ground_truth/cat_gt_cached.p'

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

acc = correct / (pos_read + neg_read)


print("acc", acc)
print("pos all", pos_read)
print("pos correct", pos_correct)
print("neg all", neg_read)
print("neg correct", neg_correct)
