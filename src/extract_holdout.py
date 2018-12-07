import os
from shutil import move

def main():
   for root, dirs, files in os.walk("/data/active-rl-data/data/images/train/cat"):
       for f in files[:2000]:
           path = os.path.join(root, f)
           move(path, path.replace('train', 'holdout'))

if __name__ == '__main__':
    main()