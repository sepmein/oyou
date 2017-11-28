import tensorflow as tf


class Trainer:
    """
        Model trainer
    """

    def __init__(self,
                 model,
                 logger,
                 saver,
                 log_dir='./',
                 model_dir='./model',
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001,
                 training_steps=100000,
                 record_interval=10,
                 save_interval=10
                 ):
        self.model = model
        self.saver = saver
        self.logger = logger
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.record_interval = record_interval
        self.save_interval = save_interval

    def train(self,
              input_x,
              input_y):
        """
        train a model
        :return:
        """
        with tf.Session() as session:
            # just a fancier version of tf.global_variables_initializer()
            # get variable first
            global_variables = self.model.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # create init op of global variables
            init = tf.variables_initializer(global_variables)
            session.run(init)
            # add training step
            with tf.name_scope('training'):
                train = self.optimizer(learning_rate=self.learning_rate) \
                    .minimize(self.model.loss)

            for i in range(self.training_steps):
                session.run(train,
                            feed_dict={
                                input_x: input_x,
                                input_y: input_y
                            })

                if i % self.record_interval == 0:
                    self.logger.log()

                if i % self.save_interval == 0:
                    self.saver.save()

    def write_summaries(self):
        pass

    def save(self):
        pass
