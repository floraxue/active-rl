import time
from PIL import Image, ImageFile
import os
from os.path import exists, join
import lmdb
import six
import shutil

import numpy as np
from env_obj import Env
from util import logger, checkdir

env = Env()
workspace = '/data/active-rl-data'
data_root = '/data/active-rl-data/data'


def create_images(category, db_index):
    # open the db
    in_db_path = env.category_raw_data_path(category, db_index)
    checkdir(join(data_root, 'local_paths', category))
    out_local_path = join(data_root, 'local_paths', category, str(db_index) + '.txt')
    img_out_dir = join(data_root, 'images', category)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    max_dim = 400

    try:
        lmdb_env = lmdb.open(in_db_path,
                             # map_size=1073741824,
                             max_readers=100,
                             readonly=True)
        lmdb_txn = lmdb_env.begin(write=False)
    except Exception:
        logger.warning(
            'Failed to open {0} for DB {1}'.format(in_db_path, db_index))
        raise

    # read and save the images
    lmdb_cursor = lmdb_txn.cursor()
    out_paths = []
    error_cnt = 0
    i = 0
    start_time = time.time()
    for key, value in lmdb_cursor:
        key = key.decode('ascii')
        out_path = join(img_out_dir, key + '.jpg')
        if not exists(out_path):
            val = lmdb_txn.get(key.encode())
            # assert val is not None
            if val is None:
                continue
            buf = six.BytesIO()
            buf.write(val)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            if image is None:
                logger.info('{} is bad'.format(key))
                error_cnt += 1
                continue
            if image.size[1] > image.size[0]:
                size = (max_dim, int(max_dim * image.size[0] / image.size[1]))
            else:
                size = (int(max_dim * image.size[1] / image.size[0]), max_dim)

            image.thumbnail(size)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            image.save(out_path, 'JPEG', quality=75)
        out_paths.append(out_path)

        i += 1
        if (i + 1) % 1000 == 0:
            logger.info('Creating %d thumbnails for %f seconds', i + 1,
                        time.time() - start_time)

    # save the out_paths
    logger.info('Writing %s', out_local_path)
    with open(out_local_path, 'w') as fp:
        for p in out_paths:
            print(p, file=fp)


def move_images(category):
    img_out_dir = join(data_root, 'images', category)
    train_dir = join(img_out_dir, 'train')
    test_dir = join(img_out_dir, 'test')
    all_names = [str(os.path.basename(fpath)) for fpath in os.listdir(img_out_dir)
                 if '.jpg' in fpath]
    train_names = all_names[:-10000]
    test_names = all_names[-10000:]
    import pdb
    pdb.set_trace()
    for name in train_names:
        src = join(img_out_dir, name)
        dst = join(train_dir, name)
        shutil.move(src, dst)
    for name in test_names:
        src = join(img_out_dir, name)
        dst = join(test_dir, name)
        shutil.move(src, dst)

    pool_dir = join(workspace, 'pool')
    train_keys = [key.split('.')[0] for key in train_names]
    test_keys = [key.split('.')[0] for key in test_names]
    train_pool = join(pool_dir, '{}_{:02d}_keys.txt'.format(category, 0))
    test_pool = join(pool_dir, '{}_holdout_keys.txt'.format(category))
    with open(train_pool, 'w') as fp:
        fp.write('\n'.join(train_keys))
    with open(test_pool, 'w') as fp:
        fp.write('\n'.join(test_keys))


def move_train_to_test(category):
    train_dir = join(data_root, 'images', 'train', category)
    test_dir = join(data_root, 'images', 'test', category)
    train_names = [str(os.path.basename(fpath)) for fpath in os.listdir(train_dir)
                   if '.jpg' in fpath]
    for name in train_names[:460]:
        src = join(train_dir, name)
        dst = join(test_dir, name)
        shutil.move(src, dst)


def main():
    category = "cat"
    # create_images(category, 0)
    # create_images(category, 1)
    # move_images(category)
    move_train_to_test(category)


if __name__ == "__main__":
    main()
