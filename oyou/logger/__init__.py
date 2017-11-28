import tensorflow as tf


class Logger:
    """
        logger class
    """

    def __init__(self, model,
                 record_interval=10):
        self.model = model
        self.file_writers = {}
        self.record_interval = record_interval

    def add_writer(self, name):
        """
        add file writer for later usage
        :param name:
        :return:
        """
        self.file_writers[name] = tf.summary.FileWriter(logdir='./', graph=self.model.graph)

    