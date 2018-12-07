import os
from os.path import join
import pickle


pool_dir = '/data/active-rl-data/pool'
fixed_keys_path = join(pool_dir, 'cat_fixed_keys.p')
train_keys_path = join(pool_dir, 'cat_train_keys.p')

image_dir = '/data/active-rl-data/data/images/'
fixed_image_dir = join(image_dir, 'fixed/cat/')
train_image_dir = join(image_dir, 'train/cat/')

fixed_image_names = [str(os.path.basename(fpath)) for fpath in os.listdir(fixed_image_dir)
                     if '.jpg' in fpath]
fixed_keys = [name.split('.')[0] for name in fixed_image_names]
pickle.dump(fixed_keys, open(fixed_keys_path, 'wb'))

train_image_names = [str(os.path.basename(fpath)) for fpath in os.listdir(train_image_dir)
                     if '.jpg' in fpath]
train_keys = [name.split('.')[0] for name in train_image_names]
pickle.dump(train_keys, open(train_keys_path, 'wb'))

