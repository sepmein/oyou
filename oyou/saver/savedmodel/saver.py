# coding=utf-8

import tensorflow as tf

a = tf.get_variable(name='a', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
add_one = tf.assign_add(a, [1])
init = tf.global_variables_initializer()
saver = tf.saved_model.builder.SavedModelBuilder(export_dir='./model')
with tf.Session() as sess:
    sess.run(init)

    saver.add_meta_graph_and_variables(sess=sess,
                                       tags=[tf.saved_model.tag_constants.SERVING])
    sess.run(add_one)
    saver.save()
    sess.run(add_one)
    saver.add_meta_graph(tags=[tf.saved_model.tag_constants.SERVING])
    # saver.save()
