import pickle
from os.path import join
import os

data_root = '/data/active-rl-data/data'


def main():
    fpath = '/data/active-rl-data/ground_truth/cat_gt_holdout.txt'
    dic = {}
    with open(fpath, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            key, label = line.split(' ')
            dic[key] = int(label)

    pickle.dump(dic,
                open('/data/active-rl-data/ground_truth/cat_gt_cached_holdout.p', 'wb'))

    train_dir = join(data_root, 'images', 'holdout', 'cat')
    # test_dir = join(data_root, 'images', 'test', 'cat')
    train_names = os.listdir(train_dir)
    # test_names = os.listdir(test_dir)
    removed_keys = []
    for fname in train_names:
        key = fname.split('.')[0]
        if key not in dic:
            removed_keys.append(str(key))
            fullpath = join(train_dir, fname)
            os.remove(fullpath)
    # for fname in test_names:
    #     key = fname.split('.')[0]
    #     if key not in dic:
    #         removed_keys.append(str(key))
    #         fullpath = join(test_dir, fname)
    #         os.remove(fullpath)

    out_remove_keys = '/data/active-rl-data/ground_truth/no_gt_removed_keys_holdout.txt'
    with open(out_remove_keys, 'w') as fp:
        fp.write('\n'.join(removed_keys))


if __name__ == '__main__':
    main()
