import os

import tensorflow as tf


def _default_compare_fn_for_saving_strategy(a, b):
    if b is None:
        return True
    if a is None:
        return False
    if a < b:
        return True
    else:
        return False


class Model:
    """
    Tensorflow model builder
    """

    def __init__(self,
                 graph=tf.get_default_graph(),
                 name=None,
                 folder=os.getcwd()
                 ):
        """
        Init a model object
        :param graph: tf.Graph object, default use tf.get_default_graph() function to get the default graph
        :param name: the name of the model
        :param folder: define the folder for model saver(tf.saved_model) and logger(tf.summary),
        default as current folder
        """
        if isinstance(graph, tf.Graph):
            self.graph = graph
        else:
            raise Exception('Graph should be instance of tf.Graph')
        self._name = name
        self._input_x = None
        self._input_y = None
        self._loss = None
        self._prediction = None
        self._saver_indicator = None
        self.tags = self._name
        self.session = None
        self.file_writers = []
        self.folder = folder
        self.log_folder = folder + '/log'
        self.model_folder = folder + '/model'
        # FIXME: if trained before, model saver dir is existed, reinitiating the same instance will cause error
        self.saver = tf.saved_model.builder.SavedModelBuilder(export_dir=self.model_folder)
        # TODO: define default saving strategy
        self.saving_strategy = {
            'interval': 10,
            'max_to_keep': 5,
            'indicator_tensor': self._loss,
            'top_model_list': [
                dict(performance=None)
            ],
            'compare_fn': _default_compare_fn_for_saving_strategy
        }

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def input_x(self):
        return self._input_x

    @input_x.setter
    def input_x(self, input_x):
        self._input_x = self.get_tensor(input_x)

    @property
    def input_y(self):
        return self._input_y

    @input_y.setter
    def input_y(self, input_y):
        self._input_y = self.get_tensor(input_y)

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = self.get_tensor(prediction)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = self.get_tensor(loss)

    # @property
    # def saver_indicator(self):
    #     return self._saver_indicator
    #
    # @saver_indicator.setter
    # def saver_indicator(self, saver_indicator):
    #     self._saver_indicator = self.get_tensor(saver_indicator)

    def get_tensor_by_name(self, name):
        return self.graph.get_tensor_by_name(name)

    def get_tensor(self, tensor):
        # type = self.check_type(tensor)
        # if type is 'Str':
        #     return self.get_tensor_by_name(tensor)
        # else:
        #     return tensor
        return self.graph.as_graph_element(tensor)

    def __iter__(self):
        # if targets is 'ops':
        return iter(self.tensors_generator())
        # elif targets is 'tensor':
        #     for op in get_operations()
        #       return op.values()
        #     return iter(self.tensors_generator())

    # def __next__(self):
    #     return 0

    def tensors_generator(self):
        for ops in self.graph.get_operations():
            for input_tensor in ops.inputs:
                yield input_tensor

            for output_tensor in ops.outputs:
                yield output_tensor

    def create_log_group(self,
                         name,
                         feed_tensors,
                         record_interval=10,
                         ):
        """
        create log group
        :param name:
        :param record_interval:
        :param feed_tensors list obj to hold the name of the feed_dict
        self.train will check the **kwargs in order to match those names
        :return:
        """
        # TODO: add feed_dict placeholder to the log group
        _existed = False
        for writer in self.file_writers:
            if writer['name'] is name:
                _existed = True

        directory = self.log_folder + '/' + name
        # sanitary check the feed tensor
        for tensor in feed_tensors:
            if not isinstance(tensor, tf.Tensor):
                raise Exception('feed tensor of the file writer should be tf.tensor')

        if not _existed:
            self.file_writers.append({
                'name': name,
                'writer': tf.summary.FileWriter(logdir=directory, graph=self.graph),
                'record_interval': record_interval,
                'summaries': [],
                'feed_dict': feed_tensors
            })
        else:
            raise Exception('Calling create log group, log group: ' + name + ' already existed.')

    def _add_to_summary_writer(self, log_group, summary):
        _existed = False
        index = None
        for idx, writer in enumerate(self.file_writers):
            if writer['name'] is log_group:
                _existed = True
                index = idx

        if _existed:
            writer = self.file_writers[index]
        else:
            raise Exception('Called _added_to_summary_writer, summary group should not defined')
        writer['summaries'].append(summary)

    def log_scalar(self, name, tensor, group):
        """
        log a scalar summary,
        to a log group
        :param name:
        :param tensor:
        :param group:
        :return:
        """
        summary = tf.summary.scalar(name=name, tensor=tensor)
        self._add_to_summary_writer(log_group=group, summary=summary)

    def log_histogram(self, name, tensor, group):
        summary = tf.summary.histogram(name, tensor)
        self._add_to_summary_writer(log_group=group, summary=summary)

    def finalized_log(self):
        for writer in self.file_writers:
            summary_op = tf.summary.merge(writer['summaries'],
                                          name=writer['name'] + '_summaries')
            writer['summary_op'] = summary_op

    def log(self, session, step, log_group, feed_dict):
        """Log the predefined summaries on the run

        Call the log function when training.

        The log function takes 4 parameters:

        :param session: tf.Session object for running the summaries
        :param step: int representing the global training step
        :param log_group: string representing the tf.FileWriter that was predefined in the model
        :param feed_dict: the feed_dict object for running the summaries
        :return:
        """
        for index, writer in enumerate(self.file_writers):
            if writer['record_interval'] % step == 0 and writer['name'] is log_group:
                summaries = session.run(writer['summary_op'], feed_dict=feed_dict)
                writer['writer'].add_summary(summary=summaries, global_step=step)

    def hook_session(self, session):
        self.session = session

    def add_meta_graph_and_variables(self, tags):
        """
        add meta graph and variables to saver
        :param tags:
        :return:
        """
        if not self.session:
            raise Exception('call hook session first')
        self.tags = tags
        self.saver.add_meta_graph_and_variables(self.session, tags)

    def define_saving_strategy(self,
                               indicator_tensor,
                               interval,
                               max_to_keep=5,
                               compare_fn=_default_compare_fn_for_saving_strategy
                               ):
        """
        Before defining saving strategy, model.saving_indicator should be set.
        self defined saving strategy for the model
        :return:
        """
        # check saver indicator first, it should be a type of tensor
        # if self.saver_indicator is None and not isinstance(self.saver_indicator, tf.Tensor):
        #     raise Exception('Please set saver_indicator first to use define_saving_strategy function,'
        #                     ' because the model should know what to decide which one to save.')
        self.saving_strategy['interval'] = interval
        self.saving_strategy['max_to_keep'] = max_to_keep
        self.saving_strategy['indicator_tensor'] = indicator_tensor
        self.saving_strategy['compare_fn'] = compare_fn

    def save(self, step, feed):
        # detect current_best_model
        current_best_model = dict(
            performance=None
        )
        if len(self.saving_strategy['top_model_list']) is not 0:
            current_best_model = self.saving_strategy['top_model_list'][-1]

        if self.saving_strategy is None:
            raise Exception('Should define saving strategy before saving.')
        if step % self.saving_strategy['interval'] == 0:
            # check performance
            # TODO: how to handle the situation when multiple input of saver indicator
            performance = self.session.run(self.saving_strategy['indicator_tensor'],
                                           feed_dict=feed)
            print(performance)
            # compare it to the current best model
            # if performance is better, add it to the current best model list
            # and save it on the disk
            # TODO: how to decide performance is better? should use greater? or miner? how to predefine this
            # particular information in the saving strategy.
            if self.saving_strategy['compare_fn'](performance, current_best_model['performance']):
                self.saving_strategy['top_model_list'].append({
                    'performance': performance,
                    'step': step
                })
                self.saving_strategy['top_model_list'].pop()
                self.saver.save()
                # TODO: delete previous saved model, check python os fs delete api

    def load(self, session):
        tf.saved_model.loader.load(session, self.tags, self.model_folder)

    def train(self,
              input_x,
              input_y,
              learning_rate=0.001,
              training_steps=100000,
              optimizer=tf.train.AdamOptimizer,
              **kwargs):

        """

        :param input_x:
        :param input_y:
        :param learning_rate:
        :param training_steps:
        :param optimizer:
        :param kwargs:
        :return:
        """
        with tf.Session(graph=self.graph) as sess:
            # define training ops
            train = optimizer(learning_rate=learning_rate).minimize(self.loss)

            # just a fancier version of tf.global_variables_initializer()
            # get variable first
            global_variables = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # create init op of global variables
            init_global = tf.variables_initializer(global_variables)
            local_variables = self.graph.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
            init_local = tf.variables_initializer(local_variables)
            sess.run([init_global, init_local])

            # create log op
            self.finalized_log()

            # hook session for saver
            self.hook_session(sess)

            # add meta graph and variables
            self.add_meta_graph_and_variables(tags=self.tags)

            # training steps
            for i in range(training_steps):
                sess.run(train,
                         feed_dict={
                             self.input_x.name: input_x,
                             self.input_y.name: input_y
                         })
                # TODO: if the log group is undecided or multiple,
                # how could we define the parameters of the training function
                # TODO: add some explanations for better understanding
                # for every file writer, check all the input kwargs
                # if the args name is the following api : writer name + _ + input tensor name
                # then add the arg to the collection
                # then run the file writer with the collection
                # TODO: Does all the inputs of the log group has been defined? It should be checked
                for index, writer in enumerate(self.file_writers):
                    collection = {}
                    for key, value in kwargs.items():
                        for tensor in writer['feed_dict']:
                            if writer['name'] + '_' + tensor.name == key + ':0':
                                collection[tensor.name] = value
                    self.log(session=sess,
                             step=i + 1,
                             log_group=writer['name'],
                             feed_dict=collection)

                # TODO: saving strategy
                for key, value in kwargs.items():
                    if key is 'saving_indicator_feed':
                        self.save(step=i,
                                  feed=value)


# TODO add a DNN model for convenience
class DNN(Model):
    def __init__(self):
        super().__init__()
        pass
