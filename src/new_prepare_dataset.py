from os.path import join
import pickle
from PIL import Image, ImageFile
import lmdb
import os
import six


DATA_ROOT = '/data3/floraxue/cs294/active-rl-data'
GT_DIR = join(DATA_ROOT, 'ground_truth')
POOL_DIR = join(DATA_ROOT, 'pool')
IMAGES_DIR = join(DATA_ROOT, 'data', 'images')
LMDB_DIR = '/data3/lsun/data/raw/cat'


def read_keys():

    key_to_lmdb = {}
    pos_keys_all = []
    neg_keys_all = []

    for i in range(17):
        gt_file = join(GT_DIR, 'cat_lmdb_{:04d}_gt.txt'.format(i))
        with open(gt_file, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            tokens = line.split(' ')
            key = tokens[0]
            key_to_lmdb[key] = i
            if int(tokens[1]) == 1:
                pos_keys_all.append(key)
            elif int(tokens[1]) == -1:
                neg_keys_all.append(key)

    pos_keys_train = pos_keys_all[:13000]
    pos_keys_fixed = pos_keys_all[13000:16000]
    pos_keys_holdout = pos_keys_all[16000:27000]

    neg_keys_train = neg_keys_all[:13000]
    neg_keys_fixed = neg_keys_all[13000:16000]
    neg_keys_holdout = neg_keys_all[16000:27000]

    keys_train = pos_keys_train + neg_keys_train
    keys_fixed = pos_keys_fixed + neg_keys_fixed
    keys_holdout = pos_keys_holdout + neg_keys_holdout

    pickle.dump(keys_train, open(join(POOL_DIR, 'cat_train_keys.p'), 'wb'))
    pickle.dump(keys_fixed, open(join(POOL_DIR, 'cat_fixed_keys.p'), 'wb'))
    pickle.dump(keys_holdout, open(join(POOL_DIR, 'cat_holdout_keys.p'), 'wb'))

    pos_gts_train = {k: 1 for k in pos_keys_train}
    pos_gts_fixed = {k: 1 for k in pos_keys_fixed}
    pos_gts_holdout = {k: 1 for k in pos_keys_holdout}

    neg_gts_train = {k: -1 for k in neg_keys_train}
    neg_gts_fixed = {k: -1 for k in neg_keys_fixed}
    neg_gts_holdout = {k: -1 for k in neg_keys_holdout}

    gts = pos_gts_train
    gts.update(pos_gts_fixed)
    gts.update(neg_gts_train)
    gts.update(neg_gts_fixed)

    gts_holdout = pos_gts_holdout
    gts_holdout.update(neg_gts_holdout)

    pickle.dump(gts, open(join(GT_DIR, 'cat_gt_cached.p'), 'wb'))
    pickle.dump(gts_holdout, open(join(GT_DIR, 'cat_gt_cached_holdout.p'), 'wb'))

    lmdb_to_key_train = [[] for _ in range(17)]
    lmdb_to_key_fixed = [[] for _ in range(17)]
    lmdb_to_key_holdout = [[] for _ in range(17)]

    for key in keys_train:
        db = key_to_lmdb[key]
        lmdb_to_key_train[db].append(key)

    for key in keys_fixed:
        db = key_to_lmdb[key]
        lmdb_to_key_fixed[db].append(key)

    for key in keys_holdout:
        db = key_to_lmdb[key]
        lmdb_to_key_holdout[db].append(key)

    for i in range(17):
        create_images('cat', i, 'train', lmdb_to_key_train[i])
        create_images('cat', i, 'fixed', lmdb_to_key_fixed[i])
        create_images('cat', i, 'holdout', lmdb_to_key_holdout[i])


def create_images(category, db_index, subfolder, keys):
    # open the db
    in_db_path = join(LMDB_DIR, 'cat_{:04d}_lmdb'.format(db_index))
    img_out_dir = join(IMAGES_DIR, subfolder, category)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    max_dim = 400

    try:
        lmdb_env = lmdb.open(in_db_path,
                             # map_size=1073741824,
                             max_readers=100,
                             readonly=True)
        lmdb_txn = lmdb_env.begin(write=False)
    except Exception:
        print("error in opening lmdb {}".format(db_index))
        raise

    # read and save the images
    print("reading and saving for {}, {}".format(db_index, subfolder))
    i = 0
    for key in keys:
        out_path = join(img_out_dir, key + '.jpg')
        val = lmdb_txn.get(key.encode())
        # assert val is not None
        if val is None:
            continue
        buf = six.BytesIO()
        buf.write(val)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        if image is None:
            print('{} is bad'.format(key))
            continue
        if image.size[1] > image.size[0]:
            size = (max_dim, int(max_dim * image.size[0] / image.size[1]))
        else:
            size = (int(max_dim * image.size[1] / image.size[0]), max_dim)

        image.thumbnail(size)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        image.save(out_path, 'JPEG', quality=75)

        i += 1
        if (i + 1) % 1000 == 0:
            print('Creating {} thumbnails'.format(i + 1))


if __name__ == '__main__':
    read_keys()

