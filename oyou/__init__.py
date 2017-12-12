# define imports in this init file
import tensorflow as tf

logdir = None

class Trainer:
    def __init__(self,
                 trainer,
                 saver):
        self.trainer = trainer
        self.saver = saver
        self.session = tf.Session()

        # inject session into saver
        self.saver.session = self.session
