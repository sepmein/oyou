# coding=utf-8
"""
Define Model for building tensorflow object
"""
import os
from types import FunctionType, GeneratorType
from numpy import ndarray
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
        self._features = None
        self._targets = None
        self._losses = None
        self._prediction = None
        self._saver_indicator = None
        self.tags = self._name
        self.session = None
        self.file_writers = []
        self.folder = folder
        self.log_folder = folder + '/log'
        self.model_folder = folder + '/model'
        # FIXME: if trained before, model saver dir is existed, reinitiating the same instance will cause error
        self.savers = []
        # TODO: define default saving strategy
        self.saving_strategy = None

    @property
    def name(self):
        """
        set the name of the model
        :return: a name string.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def features(self):
        """
        Input property of the model.
        :return: input tensor
        """
        return self._features

    @features.setter
    def features(self, features):
        self._features = self.get_tensor(features)

    @property
    def targets(self):
        """
        Targets property of the model
        :return:
        """
        return self._targets

    @targets.setter
    def targets(self, targets):
        self._targets = self.get_tensor(targets)

    @property
    def prediction(self):
        """
        Prediction property of the model
        :return:
        """
        return self._prediction

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = self.get_tensor(prediction)

    @property
    def losses(self):
        """
        Losses property of the model
        :return:
        """
        return self._losses

    @losses.setter
    def losses(self, losses):
        self._losses = self.get_tensor(losses)

    # @property
    # def saver_indicator(self):
    #     return self._saver_indicator
    #
    # @saver_indicator.setter
    # def saver_indicator(self, saver_indicator):
    #     self._saver_indicator = self.get_tensor(saver_indicator)

    def get_tensor_by_name(self, name):
        """
        Given a `name` string, Find it in the model.graph
        :param name: string
        :return: tensor
        """
        return self.graph.get_tensor_by_name(name)

    def get_tensor(self, tensor):
        """
        Get a tensor from a `tensor` name or a tensor object.
        It's just a wrap of tf.graph.as_graph_element fn.
        Check the tensorflow document for more information.
        Current Version: 1.4.1
        :param tensor: Tensor object or string
        :return:
        """
        return self.graph.as_graph_element(tensor)

    def __iter__(self):
        return iter(self.tensors_generator())

    def tensors_generator(self):
        """
        Get all tensor of the graph.
        :return: iterator object
        """
        for ops in self.graph.get_operations():
            for input_tensor in ops.inputs:
                yield input_tensor

            for output_tensor in ops.outputs:
                yield output_tensor

    def create_log_group(self,
                         name,
                         record_interval=10,
                         feed_tensors=None):
        """
        Create log groups for tf.summary.File_Writer. Example usage:
        We may want to record different indicators for training and cross validation process. So we could create two log
        groups as follows:
        `
        self.create_log_group(name='training', feed_tensors=some_training_tensors)
        self.create_log_group(name='cross_validation', feed_tensors=some_cross_validation_tensors)
        `
        The name of the log group will be fed to `tf.summary.file_writer` and will be shown in the Tensorboard. The
        `feed_tensors` represents the tensors that will be the placeholder for the indices in the log group.
        :param name: string. name of the log group
        :param record_interval: int, record the interval of the log group
        :param feed_tensors a list of Tensor obj to hold the name of the feed_dict
        self.train will check the **kwargs in order to match those names
        :return:
        """
        # TODO: add feed_dict placeholder to the log group
        _existed = False
        for writer in self.file_writers:
            if writer['name'] is name:
                _existed = True

        directory = self.log_folder + '/' + name
        if feed_tensors is None:
            feed_tensors = [self.features, self.targets]
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
                'ops': [],
                'feed_dict': feed_tensors
            })
        else:
            raise Exception('Calling create log group, log group: ' + name + ' already existed.')

    def _add_to_summary_writer(self, log_group, summary, op=None):
        _existed = False
        index = None
        # check for duplication
        for idx, writer in enumerate(self.file_writers):
            if writer['name'] is log_group:
                _existed = True
                index = idx
        if _existed:
            writer = self.file_writers[index]
        else:
            raise Exception('Called _added_to_summary_writer, summary group should not defined')
        writer['summaries'].append(summary)
        if op is not None:
            writer['ops'].append(op)

    def log_scalar(self, name, tensor, group, op=None):
        """
        log a scalar summary,
        to a log group
        :param name:
        :param tensor:
        :param group:
        :param op:
        :return:
        """
        summary = tf.summary.scalar(name=name, tensor=tensor)
        self._add_to_summary_writer(log_group=group, summary=summary, op=op)

    def log_histogram(self, name, tensor, group, op=None):
        """
        Add a tf.summary.histogram log to the log_group
        :param name: string, name of the histogram
        :param tensor: tf.Tensor, the log tensor
        :param group: string, log group name
        :param op:
        :return:
        """
        summary = tf.summary.histogram(name, tensor)
        self._add_to_summary_writer(log_group=group, summary=summary, op=op)

    def finalized_log(self):
        """
        Finalized log process, merge all the summary ops.
        :return:
        """
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
            # print('log step:', step, '. writer name:',
            #       writer['name'], ' , log_group is :', log_group,
            #       '. record interval: ', writer['record_interval'])
            if step % writer['record_interval'] == 0 and writer['name'] is log_group:
                summaries, _ops = session.run([writer['summary_op'], writer['ops']], feed_dict=feed_dict)
                writer['writer'].add_summary(summary=summaries, global_step=step)

    def hook_session(self, session):
        """
        Saving process should know session, so should hook session here.
        :param session:
        :return:
        """
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
        for saver in self.savers:
            saver.add_meta_graph_and_variables(self.session, tags)

    def define_saving_strategy(self,
                               indicator_tensor,
                               interval,
                               feed_dict,
                               max_to_keep,
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
        self.saving_strategy = SavingStrategy(
            indicator_tensor=indicator_tensor,
            interval=interval,
            feed_dict=feed_dict,
            max_to_keep=max_to_keep,
            compare_fn=compare_fn
        )
        self.savers = [
            tf.saved_model.builder.SavedModelBuilder(export_dir=self.model_folder + '/' + str(_))
            for _ in range(max_to_keep)
        ]

    def save(self, step: int, feed_dict):
        """
        Save the model by step and feed_dict
        :param step:
        :param feed_dict:
        :return:
        """
        if self.saving_strategy is None:
            raise Exception('Should define saving strategy before saving.')
        if step % self.saving_strategy.interval == 0:
            # check performance
            # TODO: how to handle the situation when multiple input of saver indicator

            performance = self.session.run(self.saving_strategy.indicator_tensor,
                                           feed_dict=feed_dict)
            # compare it to the current best model
            # if performance is better, add it to the current best model list
            # and save it on the disk
            # TODO: how to decide performance is better? should use greater? or miner? how to predefine this
            # particular information in the saving strategy.
            for index, model in enumerate(self.saving_strategy.top_model_list):
                if self.saving_strategy.compare_fn(performance, model['performance']):
                    self.saving_strategy.top_model_list.insert(
                        index + 1,
                        {
                            'performance': performance,
                            'step': step
                        })
                    self.savers[index].save()
                    break

            # remove the first item of the top list
            self.saving_strategy.pop_top()

            # TODO: delete previous saved model, check python os fs delete api

    def load(self, session):
        """
        Load the saved model unfinished version.
        :param session:
        :return:
        """
        tf.saved_model.loader.load(session, self.tags, self.model_folder)

    def train(self,
              features,
              targets,
              learning_rate=0.001,
              training_steps=100000,
              optimizer=tf.train.AdamOptimizer,
              **kwargs):

        """
        TODO: 1. rename input_x and targets
        TODO: 2. input_x could accept not only numpy array, but also iterator of numpy array
        :param features:
        :param targets:
        :param learning_rate:
        :param training_steps:
        :param optimizer:
        :param kwargs:
        :return:
        """
        with tf.Session(graph=self.graph) as sess:
            # define training ops
            train = optimizer(learning_rate=learning_rate).minimize(self.losses)

            # just a fancier version of tf.global_variables_initializer()
            # get variable first
            global_variables = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # create init op of global variables
            init_global = tf.variables_initializer(global_variables)
            local_variables = self.graph.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
            init_local = tf.variables_initializer(local_variables)
            sess.run([init_global, init_local])

            # create log op by calling finalized log
            self.finalized_log()

            # hook session for saver
            self.hook_session(sess)

            # add meta graph and variables
            self.add_meta_graph_and_variables(tags=self.tags)

            # training steps
            for i in range(training_steps):
                # if isinstance(feed_dict, types.GeneratorType):
                #     features, targets = next(feed_dict)
                # elif feed_dict is list or feed_dict is tuple:
                #     # TODO: add sanity checks
                #     features, targets = feed_dict
                # else:
                #     raise Exception('Training feed dict should be a generator, list or tuple.')
                sess.run(train,
                         feed_dict={
                             self.features.name: self.get_data(features),
                             self.targets.name: self.get_data(targets)
                         })
                # TODO: if the log group is undecided or multiple,
                # how could we define the parameters of the training function
                # TODO: add some explanations for better understanding
                # for every file writer, check all the input kwargs
                # if the args name is the following api : writer name + _ + input tensor name
                # then add the arg to the collection
                # then run the file writer with the collection
                # TODO: Does all the inputs of the log group has been defined? It should be checked
                # loop through kwargs
                # for all file writers, check it's name
                for index, writer in enumerate(self.file_writers):
                    collection = {}
                    for key, value in kwargs.items():
                        for tensor in writer['feed_dict']:
                            if writer['name'] + '_' + tensor.name == key + ':0':
                                collection[tensor.name] = self.get_data(value)
                    self.log(session=sess,
                             step=i + 1,
                             log_group=writer['name'],
                             feed_dict=collection)

                # for feed in saving strategy, if name in kwargs matches its name
                saving_feeds = {}
                for key, value in kwargs.items():
                    for feed in self.saving_strategy.feed_dict:
                        if key + ':0' == 'saving' + '_' + feed.name:
                            saving_feeds[feed.name] = self.get_data(value)
                self.save(step=i,
                          feed_dict=saving_feeds)

    @staticmethod
    def get_data(inputs):
        if callable(inputs):
            return inputs()
        elif isinstance(inputs, GeneratorType):
            return next(inputs)
        elif isinstance(inputs, ndarray) or isinstance(inputs, list):
            return inputs


class SavingStrategy:
    """

    """

    def __init__(self,
                 indicator_tensor,
                 feed_dict,
                 compare_fn=_default_compare_fn_for_saving_strategy,
                 interval=50,
                 max_to_keep=10):
        """

        """
        self.interval = interval
        self.max_to_keep = max_to_keep
        self.indicator_tensor = indicator_tensor
        self.compare_fn = compare_fn
        self.feed_dict = feed_dict
        self.top_model_list = [
            {'performance': None}
        ]

    def pop_top(self):
        if len(self.top_model_list) >= self.max_to_keep:
            self.top_model_list.pop(0)
