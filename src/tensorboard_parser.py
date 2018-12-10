import tensorflow as tf

path =

for e in tf.train.summary_iterator(path):
    for v in e.summary.value:
        if v.tag == 'loss' or v.tag == 'accuracy':
            print(v.simple_value)