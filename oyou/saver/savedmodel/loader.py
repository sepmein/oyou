# coding=utf-8
import tensorflow as tf

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './model')
    print(sess.run('a:0'))
