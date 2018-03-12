import tensorflow as tf
import numpy as np

class RNNStateHook(tf.train.SessionRunHook):

 def __init__(self, params):
    self.init_states = None
    self.current_state = np.zeros((params["batch_size"], params["state_size"])) #this will change when we move to LSTM

 def before_run(self, run_context):
    run_args = tf.train.SessionRunArgs(tf.get_default_graph().get_tensor_by_name('RNN/output_states:0'),
                                       {self.init_states: self.current_state, },)
    return run_args

 def after_run(self, run_context, run_values):
    self.current_state = run_values[0][0] #depends on your session run arguments!!!!!!!

 def begin(self):
    self.init_states = tf.get_default_graph().get_tensor_by_name('RNN/init_states:0')
