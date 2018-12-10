import time
from PIL import Image, ImageFile
import os
from os.path import exists, join
import lmdb
import six
import shutil

from util import logger
import pickle

# workspace = '/data/active-rl-data'
data_root = '/data3/floraxue/cs294/active-rl-data/data'


def create_images(category, db_index):

    cat_gt = pickle.load(open('/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached_more.p', 'rb'))

    # open the db
    # in_db_path = env.category_raw_data_path(category, db_index)
    in_db_path = '/data3/lsun/data/raw/cat/cat_{:04d}_lmdb'.format(db_index)
    # checkdir(join(data_root, 'local_paths', category))
    # out_local_path = join(data_root, 'local_paths', category, str(db_index) + '.txt')
    img_out_dir = join(data_root, 'images', 'train', category)
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
    logger.info("reading and saving")
    lmdb_cursor = lmdb_txn.cursor()
    out_paths = []
    error_cnt = 0
    i = 0
    start_time = time.time()
    keys = []
    more_pos_count = 0
    for key, value in lmdb_cursor:
        key = key.decode('ascii')
        if key not in cat_gt:
            continue
        target = cat_gt[key]
        if target == -1:
            continue
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
        keys.append(key)
        more_pos_count += 1

        i += 1
        if (i + 1) % 1000 == 0:
            logger.info('Creating %d thumbnails for %f seconds', i + 1,
                        time.time() - start_time)

    # save the out_paths
    # logger.info('Writing %s', out_local_path)
    # with open(out_local_path, 'w') as fp:
    #     for p in out_paths:
    #         print(p, file=fp)
    print("db", db_index)
    print("got", more_pos_count)
    return keys


def dump_gts():
    fpath = '/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_more.txt'
    dic = {}
    with open(fpath, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            key, label = line.split(' ')
            dic[key] = int(label)

    pickle.dump(dic,
                open('/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached_more.p', 'wb'))


if __name__ == '__main__':
    dump_gts()
    keys = create_images('cat', 16)
    pickle.dump(keys, open('/data3/floraxue/cs294/active-rl-data/pool/cat_{:04d}_lmdb_poskeys.p'.format(16), 'wb'))

