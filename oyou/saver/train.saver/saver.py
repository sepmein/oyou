# coding=utf-8

import tensorflow as tf

a = tf.get_variable(name='a', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
add_one = tf.assign_add(a, [1])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    sess.run(add_one)

    saver.save(sess=sess, save_path='./model')
