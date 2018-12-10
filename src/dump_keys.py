import os
from os.path import join
import pickle


pool_dir = '/data/active-rl-data/pool'
fixed_keys_path = join(pool_dir, 'cat_fixed_keys.p')
train_keys_path = join(pool_dir, 'cat_train_keys.p')

old_fixed_keys = pickle.load(open(fixed_keys_path, 'rb'))
# old_train_keys = pickle.load(open(train_keys_path, 'rb'))

new_fixed_keys = []
new_train_keys = []

# for i in range(4, 10):
#     new_train_keys_path = join(pool_dir, 'cat_{:04d}_lmdb_poskeys.p'.format(i))
#     new_train_keys = pickle.load(open(new_train_keys_path, 'rb'))

for i in range(12, 15):
    new_fixed_keys_path = join(pool_dir, 'cat_{:04d}_lmdb_poskeys.p'.format(i))
    new_fixed_keys = pickle.load(open(new_fixed_keys_path, 'rb'))

# image_dir = '/data/active-rl-data/data/images/'
# fixed_image_dir = join(image_dir, 'fixed/cat/')
# train_image_dir = join(image_dir, 'train/cat/')

# fixed_image_names = [str(os.path.basename(fpath)) for fpath in os.listdir(fixed_image_dir)
#                      if '.jpg' in fpath]
fixed_keys = old_fixed_keys + new_fixed_keys
pickle.dump(fixed_keys, open(fixed_keys_path, 'wb'))

# train_image_names = [str(os.path.basename(fpath)) for fpath in os.listdir(train_image_dir)
#                      if '.jpg' in fpath]
# train_keys = old_train_keys + new_train_keys
# pickle.dump(train_keys, open(train_keys_path, 'wb'))

