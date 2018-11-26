import time
from PIL import Image, ImageFile
import os
from os.path import exists, join
import lmdb
import six

import numpy as np
from env_obj import Env
from util import logger

env = Env()
workspace = '/data3/floraxue/active-rl/data/'


def create_images(category, db_index):
    # open the db
    in_db_path = env.category_raw_data_path(category, db_index)
    out_local_path = join(workspace, 'local_paths', category, str(db_index) + '.txt')
    img_out_dir = join(workspace, 'images', category, str(db_index))
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
        out_path = join(img_out_dir, key + '.jpg')
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


def main():
    category = "cat"
    create_images(category, 0)
    create_images(category, 1)


if __name__ == "__main__":
    main()
