import os
import shutil
import unittest
from unittest import TestCase

import numpy as np
import tensorflow as tf

import oyou
from oyou.model import Model, _default_compare_fn_for_saving_strategy


class TestOyou(TestCase):
    def test_is_string(self):
        s = oyou
        self.assertTrue(1, 1)

    def test_load_build(self):
        # define a super simple model
        # dummy input and output
        x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
        y = np.array([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=np.float32)
        w = tf.get_variable(name='weight',
                            shape=[1, 1],
                            dtype=tf.float32)
        b = tf.get_variable(name='bias',
                            shape=[1],
                            initializer=tf.initializers.zeros)
        input_x = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 1],
                                 name='input_x')
        input_y = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 1],
                                 name='input_y')
        prediction = tf.matmul(input_x, w)

        loss = tf.reduce_sum(tf.abs(prediction - input_y))
        shutil.rmtree('./model')
        model = Model(name='simple_model')

        self.assertIsInstance(model, Model, msg='Create model instance, should be the instance of the Model class')

        self.assertIs(model.graph, tf.get_default_graph())
        self.assertIs(model.name, 'simple_model')
        self.assertEqual(model.folder, os.getcwd())
        model.features = input_x
        model.targets = input_y
        model.prediction = prediction
        self.assertEqual(model.prediction, prediction)
        self.assertEqual(model._prediction, prediction)

        model.losses = loss
        self.assertEqual(model.losses, loss)

        # test get tensor by name
        self.assertEqual(model.get_tensor_by_name(prediction.name), prediction)

        # test get tensor
        self.assertEqual(model.get_tensor(prediction), prediction)

        for tensor in model:
            self.assertIsInstance(tensor, tf.Tensor)

        model.create_log_group(name='training',
                               feed_tensors=[input_x, input_y],
                               record_interval=10
                               )

        model.create_log_group(name='cross_validation',
                               feed_tensors=[input_x, input_y],
                               record_interval=20
                               )
        self.assertIsInstance(model.file_writers, list)
        self.assertEqual(len(model.file_writers), 2)
        for writer in model.file_writers:
            if writer['name'] is 'training':
                self.assertIs(writer['record_interval'], 10)
                self.assertEqual(writer['feed_dict'], [input_x, input_y])
            if writer['name'] is 'cross_validation':
                self.assertEqual(writer['feed_dict'], [input_x, input_y])

        model.log_scalar(name='loss', tensor=loss, group='training')
        model.log_scalar(name='loss', tensor=loss, group='cross_validation')
        self.assertEqual(model.file_writers[0]['summaries'], [model.get_tensor_by_name('loss:0')])

        model.finalized_log()
        for writer in model.file_writers:
            print(writer['summary_op'])
            self.assertEqual(writer['summary_op'], model.get_tensor_by_name(
                writer['name'] + '_summaries' + '/' + writer['name'] + '_summaries' + ':0'))

        session = tf.Session()
        model.hook_session(session)
        self.assertEqual(model.session, session)

        # init variables
        # global_variables = model.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # init = tf.variables_initializer(global_variables)
        # session.run(init)
        #
        # model.add_meta_graph_and_variables(tags=['test_model'])
        model.define_saving_strategy(
            indicator_tensor=loss,
            interval=10,
            max_to_keep=5
        )
        self.assertDictEqual({
            'interval': 10,
            'max_to_keep': 5,
            'indicator_tensor': loss,
            'top_model_list': [dict(performance=None)],
            'compare_fn': _default_compare_fn_for_saving_strategy
        }, model.saving_strategy)

        model.train(
            features=x,
            targets=y,
            training_steps=10000,
            learning_rate=0.001,
            training_input_x=x,
            training_input_y=y,
            cross_validation_input_x=x,
            cross_validation_input_y=y,
            saving_indicator_feed={
                input_x: x,
                input_y: y
            }
        )

        session.close()


if __name__ == '__main__':
    unittest.main()
