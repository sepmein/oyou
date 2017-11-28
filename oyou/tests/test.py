from unittest import TestCase

import tensorflow as tf
import oyou
from oyou.build import Model
from oyou.train import Trainer
from oyou.logger import Logger
from oyou.save import Saver


class TestOyou(TestCase):
    def test_is_string(self):
        s = oyou
        self.assertTrue(1, 1)

    def test_load_build(self):
        # define a super simple model
        constant = tf.constant(1)
        tf.summary.scalar(constant)
        graph = tf.get_default_graph()
        model = Model(graph=graph)
        logger = Logger(model=model)
        logger.add_writer('training')
        logger.merge_all()
        saver = Saver()
        self.assertIsInstance(model, Model)
        trainer = Trainer(model=model,
                          logger=logger,
                          saver=saver
                          )
