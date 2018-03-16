# coding=utf-8
"""
Define Model for building tensorflow object
"""
from os import getcwd
from os.path import isdir
from random import randint
from shutil import rmtree
from types import GeneratorType

import tensorflow as tf
from numpy import ndarray


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
                 folder=getcwd()
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
        self.signature_definition_map = None
        self.tags = None

    @property
    def name(self):
        """
        set the name of the model
        :return: a name string.
        """
        return self._name

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
        _existed = False
        for writer in self.file_writers:
            if writer['name'] is name:
                _existed = True

        directory = self.log_folder + '/' + name
        if feed_tensors is None:
            feed_tensors = [self.features, self.targets]
        # sanitary check the feed tensor
        # for tensor in feed_tensors:
        #     if not isinstance(tensor, tf.Tensor):
        #         raise Exception('feed tensor of the file writer should be tf.tensor')

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

    def log_with_result(self, step, log_group, result):
        """
        log tensor
        {
            name,
            writer: tf.summary.FileWriter,
            record_interval,
            summaries: ?,
            ops: ?,
            feed_dict
        }
        :param step:
        :param log_group:
        :param result:
        :return:
        """
        for index, writer in enumerate(self.file_writers):
            if step % writer['record_interval'] == 0 and writer['name'] is log_group:
                writer['writer'].add_summary(summary=result, global_step=step)

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

    def build_meta_graph_and_variables(self):
        """
        Build meta graph and variables, output a signature definition map
        :return:
        """
        # dependencies: start
        input_key = 'input'
        output_key = 'output'
        input_item = self.features
        output_item = self.prediction
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        # end
        signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={input_key: input_item},
            outputs={output_key: output_item}
        )
        self.signature_definition_map = {
            signature_key: signature_definition
        }
        self.tags = [tf.saved_model.tag_constants.SERVING]

    def add_meta_graph_and_variables(self, saver_index):
        """
        add meta graph and variables to saver
        :arg saver
        :return:
        """
        if not self.session:
            raise Exception('call hook session first')

        if self.signature_definition_map is None:
            self.build_meta_graph_and_variables()

        # check if the model has been saved before
        variable_path = self.model_folder + '/' + str(saver_index) + '/variables'
        saver = self.savers[saver_index]
        if isdir(variable_path):
            saver.add_meta_graph(tags=self.tags,
                                 signature_def_map=self.signature_definition_map)
        else:
            saver.add_meta_graph_and_variables(sess=self.session,
                                               tags=self.tags,
                                               signature_def_map=self.signature_definition_map)

    @classmethod
    def load(cls, path):
        """
        Load the saved model
        :return:
        """
        tf.reset_default_graph()
        session = tf.Session()
        # dependencies re-injection
        # TODO, concat this part with self.add_meta_graph_variables part
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        input_key = 'input'
        output_key = 'output'
        tags = [tf.saved_model.tag_constants.SERVING]

        # load meta graph
        meta_graph_definition = tf.saved_model.loader.load(
            sess=session,
            tags=tags,
            export_dir=path
        )

        # get signature_definition_map
        signature_definition_map = meta_graph_definition.signature_def
        target_signature = signature_definition_map[signature_key]

        # load features_signature, get feature name and targets name
        features_name = target_signature.inputs[input_key].name
        targets_name = target_signature.outputs[output_key].name

        # get features in the graph by name
        features = session.graph.get_tensor_by_name(features_name)
        targets = session.graph.get_tensor_by_name(targets_name)

        # TODO: How to get name?
        model = cls(graph=session.graph, folder=path)
        model.features = features
        model.prediction = targets
        model.session = session
        return model

    def predict(self, inputs):
        """
        :param inputs:
        :return:
        """
        predictions = self.session.run(self.prediction, {
            self.features.name: inputs
        })
        return predictions

    def define_saving_strategy(self,
                               indicator_tensor,
                               interval,
                               max_to_keep,
                               compare_fn=_default_compare_fn_for_saving_strategy,
                               feed_dict=None
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
            # particular information in the saving strategy.
            for index, model in enumerate(self.saving_strategy.top_model_list):
                if self.saving_strategy.compare_fn(performance, model['performance']):
                    self.saving_strategy.top_model_list.insert(
                        index,
                        {
                            'performance': performance,
                            'step': step
                        })
                    self.savers[index].save()
                    break

            # remove the first item of the top list
            self.saving_strategy.top_model_list.pop()

            # TODO: delete previous saved model, check python os fs delete api

    def run_step(self,
                 name,
                 tensor_to_run,
                 feed_dict,
                 frequency,
                 output_travel_forward_in_time,
                 output_to_other_run_step
                 ):
        """
        TODO: define a single running step, that:
            pick a tensor to run,
            given a feed dict,
            at some frequency,
            output to other run step or log step

        :param name:
        :param tensor_to_run:
        :param feed_dict:
        :param frequency:
        :param output_travel_forward_in_time:
        :param output_to_other_run_step:
        :return:
        """
        pass

    def train(self,
              features,
              targets,
              learning_rate=0.001,
              training_steps=100000,
              optimizer=tf.train.AdamOptimizer,
              close_session=True,
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
        :param close_session: Bool
        :return:
        """
        sess = tf.Session(graph=self.graph)
        # define training ops
        global_step = tf.train.create_global_step(graph=sess.graph)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                           global_step=global_step,
                                                           decay_steps=1000,
                                                           decay_rate=0.5)

        optimizer_fn = optimizer(learning_rate=decayed_learning_rate)
        gradient_and_vars = optimizer_fn.compute_gradients(self.losses)
        i = 0
        for grad, var in gradient_and_vars:
            self.log_histogram(str(i), grad, 'training')
            i += 1
        capped_gvs = [
            (tf.clip_by_norm(grad, clip_norm=1.0), var) for grad, var in gradient_and_vars]
        for grad, var in capped_gvs:
            self.log_histogram(grad.name, grad, 'training')
        train = optimizer_fn.apply_gradients(grads_and_vars=capped_gvs)
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
        self.add_meta_graph_and_variables()

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

        # if close session(default):
        if close_session:
            sess.close()

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
        self._initial_state = None

    def pop(self):
        if len(self.top_model_list) >= self.max_to_keep:
            self.top_model_list.pop()


class RnnModel(Model):
    def __init__(self,
                 graph=None,
                 name=None,
                 folder=None,
                 log_interval=50
                 ):
        if graph is None:
            graph = tf.get_default_graph()
        if folder is None:
            folder = getcwd()
        Model.__init__(self, graph=graph, name=name, folder=folder)
        self._states = None
        self._final_states = None
        self._log_save_interval = 50
        self.create_log_group(name='training', record_interval=self._log_save_interval)
        self.create_log_group(name='cv', record_interval=self._log_save_interval)

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        self._states = self.get_tensor(states)

    @property
    def final_states(self):
        return self._final_states

    @final_states.setter
    def final_states(self, final_states):
        self._final_states = final_states

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        get initial state for input to training function
        :return:
        """
        self._initial_state = initial_state

    @property
    def log_save_interval(self):
        return self._log_save_interval

    @log_save_interval.setter
    def log_save_interval(self, interval):
        self._log_save_interval = interval

    def log_scalar_to_training_group(self, name, tensor, op=None):
        self.log_scalar(name=name,
                        tensor=tensor,
                        group='training',
                        op=None)

    def log_histogram_to_training_group(self, name, tensor, op=None):
        self.log_histogram(name=name,
                           tensor=tensor,
                           group='training',
                           op=None)

    def log_scalar_to_cv_group(self, name, tensor, op=None):
        self.log_scalar(name=name,
                        tensor=tensor,
                        group='cv',
                        op=None)

    def log_histogram_to_cv_group(self, name, tensor, op=None):
        self.log_histogram(name=name,
                           tensor=tensor,
                           group='cv',
                           op=None)

    def _get_training_tensor_to_run(self, train, step):
        """
        build running tensor for session.run
        summary structure
        {
            interval:
            tensor:
            group
        }
        :return:
        """
        to_run = [train]
        if self._summaries is not None:
            for summary in self._summaries:
                if step % summary['interval']:
                    to_run.append(summary['tensor'])
        if step % self.saving_strategy.interval:
            to_run.append(self.saving_strategy.indicator_tensor)
        return to_run

    def _get_cv_tensor_to_run(self, step):
        pass

    # def log(self, tensor, interval, output_fn):
    #     pass

    def train(self,
              features,
              targets,
              cv_features,
              cv_targets,
              training_epochs,
              cv_epochs,
              learning_rate=0.001,
              training_steps=100000,
              optimizer=tf.train.AdamOptimizer,
              close_session=True,
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
        :param close_session: Bool
        :return:
        """
        sess = tf.Session(graph=self.graph)
        # define training ops
        global_step = tf.train.create_global_step(graph=sess.graph)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                           global_step=global_step,
                                                           decay_steps=1000,
                                                           decay_rate=0.5)

        optimizer_fn = optimizer(learning_rate=decayed_learning_rate)
        train = optimizer_fn.minimize(self.losses)
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

        # get summary writer
        training_writer = None
        cv_writer = None
        for writer in self.file_writers:
            if writer['name'] is 'training':
                training_writer = writer
            elif writer['name'] is 'cv':
                cv_writer = writer

        should_log_save = False
        one_randomly_picked_int_from_j_training_epochs = 0
        one_randomly_picked_int_from_k_cv_epochs = 0

        # build running tensor
        training_tensor = [train, self.final_states]
        training_tensor_with_log = [train, self.final_states, training_writer['summary_op']]
        cv_tensor = self.final_states
        cv_tensor_with_log_save = [self.final_states, cv_writer['summary_op'], self.saving_strategy.indicator_tensor]

        # training steps
        for i in range(training_steps):
            # get initial state
            states = sess.run(self.initial_state)

            print(i)
            # randomly pick one of j to log and save the model
            if ((i + 1) % self._log_save_interval) == 0:
                should_log_save = True
                one_randomly_picked_int_from_j_training_epochs = randint(0, training_epochs)
                one_randomly_picked_int_from_k_cv_epochs = randint(0, cv_epochs)
            else:
                should_log_save = False

            if not should_log_save:
                for l in range(training_epochs):
                    _, states = sess.run(training_tensor,
                                         feed_dict={
                                             self.features.name: self.get_data(features),
                                             self.targets.name: self.get_data(targets),
                                             self.states.name: states
                                         })
            if should_log_save:
                for j in range(training_epochs):
                    if j is not one_randomly_picked_int_from_j_training_epochs:
                        _, states = sess.run(training_tensor,
                                             feed_dict={
                                                 self.features.name: self.get_data(features),
                                                 self.targets.name: self.get_data(targets),
                                                 self.states.name: states
                                             })
                    else:
                        _, states, training_summary = sess.run(training_tensor_with_log,
                                                               feed_dict={
                                                                   self.features.name: self.get_data(features),
                                                                   self.targets.name: self.get_data(targets),
                                                                   self.states.name: states
                                                               })
                        # TODO add summary to training summary
                        training_writer['writer'].add_summary(summary=training_summary, global_step=i)
                for k in range(cv_epochs):
                    if k is not one_randomly_picked_int_from_k_cv_epochs:
                        states = sess.run(cv_tensor,
                                          feed_dict={
                                              self.features.name: self.get_data(cv_features),
                                              self.targets.name: self.get_data(cv_targets),
                                              self.states.name: states
                                          })
                    else:
                        states, cv_summary, saving_indicator_result = sess.run(cv_tensor_with_log_save,
                                                                               feed_dict={
                                                                                   self.features.name: self.get_data(
                                                                                       cv_features),
                                                                                   self.targets.name: self.get_data(
                                                                                       cv_targets),
                                                                                   self.states.name: states
                                                                               })
                        # TODO: add summary to cv summary
                        cv_writer['writer'].add_summary(summary=cv_summary, global_step=i)

                        # TODO: running saving strategy
                        self.save(step=i, performance=saving_indicator_result)

        # if close session(default):
        if close_session:
            sess.close()

    def save(self, step, performance):
        if self.saving_strategy is None:
            raise Exception('Should define saving strategy before saving.')
        # compare it to the current best model
        # if performance is better, add it to the current best model list
        # and save it on the disk
        # particular information in the saving strategy.
        for index, model in enumerate(self.saving_strategy.top_model_list):
            # remove the last item of the top list
            if len(self.saving_strategy.top_model_list) >= 10:
                self.saving_strategy.top_model_list.pop()
            better_performance = self.saving_strategy.compare_fn(performance, model['performance'])

            if better_performance:
                # update top model list
                self.saving_strategy.top_model_list.insert(
                    index,
                    {
                        'performance': performance,
                        'step': step
                    })

                # remove saved path
                # 1. check path is existed
                path = self.model_folder + '/' + str(index)
                if isdir(path):
                    rmtree(path)

                # save model
                # 2. build a new saved_model saver
                saver = tf.saved_model.builder.SavedModelBuilder(export_dir=path)
                # TODO: remove self.savers, useless

                # 3. get meta graph and variables
                if self.signature_definition_map is None:
                    self.build_meta_graph_and_variables()

                # 4. save model
                saver.add_meta_graph_and_variables(
                    sess=self.session,
                    tags=self.tags,
                    signature_def_map=self.signature_definition_map
                )
                saver.save()

                ## for debugging
                print(self.saving_strategy.top_model_list)

                break

    def define_saving_strategy(self,
                               indicator_tensor,
                               max_to_keep,
                               compare_fn=_default_compare_fn_for_saving_strategy,
                               ):
        self.saving_strategy = SavingStrategy(
            indicator_tensor=indicator_tensor,
            interval=self._log_save_interval,
            feed_dict=None,
            max_to_keep=max_to_keep,
            compare_fn=compare_fn
        )

    def predict(self,
                features,
                predict_features,
                epochs,
                predict_epochs
                ):
        # fixme: get initial state
        # todo: saving initial state in the signature map
        # todo: run the state using the procedures that runs the training
        with self.session as sess:
            initial_state = sess.run(self.initial_state)
            states = None
            for l in range(epochs):
                if l is 0:
                    states = sess.run(self.final_states,
                                      feed_dict={
                                          self.features.name: self.get_data(features),
                                          self.states.name: initial_state
                                      })
                else:
                    states = sess.run(self.final_states,
                                      feed_dict={
                                          self.features.name: self.get_data(features),
                                          self.states.name: states
                                      })
            for k in range(predict_epochs):
                prediction, states = sess.run([self.prediction, self.final_states],
                                              feed_dict={
                                                  self.features.name: self.get_data(predict_features),
                                                  self.states.name: states
                                              })
                print(prediction)

    @classmethod
    def load(cls, path):
        """
        Load the saved model
        :return:
        """
        tf.reset_default_graph()
        session = tf.Session()
        # dependencies re-injection
        # TODO, concat this part with self.add_meta_graph_variables part
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        input_key = 'input'
        initial_state_key = 'initial_state'
        final_states_key = 'final_states'
        states_key = 'state'
        output_key = 'output'
        tags = [tf.saved_model.tag_constants.SERVING]

        # load meta graph
        meta_graph_definition = tf.saved_model.loader.load(
            sess=session,
            tags=tags,
            export_dir=path
        )

        # get signature_definition_map
        signature_definition_map = meta_graph_definition.signature_def
        target_signature = signature_definition_map[signature_key]

        # load features_signature, get feature name and targets name
        features_name = target_signature.inputs[input_key].name
        initial_state_name = target_signature.inputs[initial_state_key].name
        final_states_name = target_signature.inputs[final_states_key].name
        states_name = target_signature.inputs[states_key].name
        targets_name = target_signature.outputs[output_key].name

        # get features in the graph by name
        features = session.graph.get_tensor_by_name(features_name)
        initial_state = session.graph.get_tensor_by_name(initial_state_name)
        final_states = session.graph.get_tensor_by_name(final_states_name)
        states = session.graph.get_tensor_by_name(states_name)
        targets = session.graph.get_tensor_by_name(targets_name)

        # TODO: How to get name?
        model = cls(graph=session.graph, folder=path)
        model.features = features
        model.initial_state = tuple(tf.unstack(initial_state))
        model.final_states = tuple(tf.unstack(final_states))
        model.states = states
        model.prediction = targets
        model.session = session
        return model

    def build_meta_graph_and_variables(self):
        """
                Build meta graph and variables, output a signature definition map
                :return:
                """
        # dependencies: start
        input_key = 'input'
        initial_state_key = 'initial_state'
        final_states_key = 'final_states'
        state_key = 'state'
        output_key = 'output'
        input_item = self.features
        initial_state_item = tf.stack(self.initial_state)
        final_states_item = tf.stack(self.final_states)
        state_item = self.states
        output_item = self.prediction
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        # end
        signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                input_key: input_item,
                initial_state_key: initial_state_item,
                state_key: state_item,
                final_states_key: final_states_item
            },
            outputs={output_key: output_item}
        )
        self.signature_definition_map = {
            signature_key: signature_definition
        }
        self.tags = [tf.saved_model.tag_constants.SERVING]
