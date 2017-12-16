import unittest
from unittest import TestCase

import tensorflow as tf
import numpy as np
import oyou
from oyou.model import Model
from oyou.train import Trainer
from oyou.logger import Logger
from oyou.save import Saver


class TestOyou(TestCase):
    def test_is_string(self):
        s = oyou
        self.assertTrue(1, 1)

    def test_load_build(self):
        # define a super simple model
        # dummy input and output
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float32)
        w = tf.get_variable(name='weight',
                            shape=[1, 1],
                            dtype=tf.float32)
        input_x = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 1],
                                 name='input_x')
        input_y = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 1],
                                 name='input_y')
        prediction = tf.matmul(input_x, w, name='prediction')

        loss = tf.reduce_sum(tf.abs(prediction - input_y))

        graph = tf.get_default_graph()
        model = Model(graph=graph)
        model.loss = loss
        model.prediction = prediction
        # define logger
        logger = Logger(model=model)
        logger.add_writer('training')
        logger.bind('training', w)
        logger.merge_all()
        # define saver
        saver = Saver(
            export_dir='c:/',
            tags='test')
        # train model use trainer
        trainer = Trainer(model=model,
                          logger=logger,
                          saver=saver
                          )
        trainer.train(input_x=x.reshape(-1, 1),
                      input_y=y.reshape(-1, 1))


if __name__ == '__main__':
    unittest.main()
