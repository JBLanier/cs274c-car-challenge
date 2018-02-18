"""
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.

    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir))

    Author: Marc Stogaitis
    Source: https://github.com/marcsto/rl/blob/master/src/fast_predict.py
    Modified to support base tf.estimator with input functions through dataset.from_generator() by J.B. Lanier
 """

import tensorflow as tf


class FastPredict:

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def _get_generator_input_fn(self):

        def generator_input_fn():

            ds = tf.data.Dataset.from_generator(self.generator, tf.float32, tf.TensorShape([None, 66, 200, 3]))
            return ds.make_one_shot_iterator().get_next()

        return generator_input_fn

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.batch_size = None
        self.predictions = None
        self.next_features = None
        self.generator = self._create_generator

    def predict(self, features):
        self.next_features = features
        if self.first_run:
            self.batch_size = len(features)
            self.predictions = self.estimator.predict(input_fn=self._get_generator_input_fn())
            self.first_run = False
        elif self.batch_size != len(features):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(features)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed=True
        next(self.predictions)