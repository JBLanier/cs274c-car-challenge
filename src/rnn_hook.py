import tensorflow as tf
import numpy as np


class RNNStateHook(tf.train.SessionRunHook):

 def __init__(self, state_size):
    print("RNN STATE HOOK INIT")
    # self.counter = 0
    self.init_states = None
    self._batch_size = 1  # Batch size better be 1, or else the concept of this hook does not apply.
    self._state_size = state_size

 def before_run(self, run_context):
    # print("HOOK BEFORE RUN")
    run_args = tf.train.SessionRunArgs(tf.get_default_graph().get_tensor_by_name('RNN/output_states:0'),
                                       {self.init_states: self.current_state})
    return run_args

 def after_run(self, run_context, run_values):
    # self.counter += 1
    # print("AFTER RUN VALUES SHAPE: \n{}\n".format(run_values[0].shape))
    # print("Counter: {}".format(self.counter))
    self.current_state = run_values[0]

 def begin(self):
    print("HOOK BEGIN (Current state is now zeroed out)")
    self.current_state = np.zeros((self._batch_size, self._state_size))  # this will change when we move to LSTM
    self.init_states = tf.get_default_graph().get_tensor_by_name('RNN/init_states:0')
