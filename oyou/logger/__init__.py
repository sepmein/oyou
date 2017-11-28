import tensorflow as tf


class Logger:
    """
        logger class
    """

    def __init__(self, model,
                 record_interval=10):
        self.model = model
        self.file_writers = {}
        self.summaries = {}
        self.record_interval = record_interval
        self.ops = {}

    def add_writer(self, name):
        """
        add file writer for later usage
        :param name:
        :return:
        """
        self.file_writers[name] = tf.summary.FileWriter(logdir='./' + name, graph=self.model.graph)

    def bind(self, writer, summary):
        writer = self.file_writers[writer]
        if writer is None:
            raise Exception('Could not file file writer by name')
        t = summary
        self.summaries[self.file_writers[writer]] = t

    def merge_all(self):
        """
        Merge all summaries
        :return:
        """
        for writer in self.file_writers:
            self.ops[writer] = tf.summary.merge(summary for summary in self.summaries[writer])

    def log(self, session, step):
        if step % self.record_interval == 0:
            for writer, op in self.ops:
                summaries = session.run(op)
                writer.add_summary(
                    summary=summaries,
                    global_step=step
                )
