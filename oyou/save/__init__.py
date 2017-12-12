import tensorflow as tf


class Saver:
    def __init__(self,
                 export_dir,
                 tags,
                 ):
        self.tags = tags
        self.export_dir = export_dir
        self.saver = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
        self.saver.add_meta_graph_and_variables(session, tags)
        self.model = None
        self._session = None


@property
def session(self):
    return self._session


@session.setter
def session(self, sess):
    self._session = sess


def save(self):
    self.saver.save()


def load(self, sess):
    self.model = tf.saved_model.loader.load(sess, self.tags, self.export_dir)
