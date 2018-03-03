import tensorflow as tf
from tensorflow.python.training import training_util


class LowPerformanceStopHook(tf.train.SessionRunHook):
  """Hook that stops the process if global_step/sec is continuously too low."""

  def __init__(self,
               every_n_steps=100,
               min_steps_per_sec=10,
               output_dir=None,
               summary_writer=None):

    if every_n_steps is None:
      raise ValueError(
          "every_n_steps should be provided.")
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps, every_secs=None)

    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._last_global_step = None
    self._global_step_check_count = 0
    self._min_steps_per_sec = min_steps_per_sec
    self._every_n_steps = every_n_steps
    self._next_infraction_will_stop = False

  def begin(self):
    tf.logging.info("Create LowPerformanceStopHook.")
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.contrib.SummaryWriterCache.get(self._output_dir)
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use LowPerformanceStopHook.")
    self._summary_tag = "Monitored global_step/sec for stopping"

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def _log_record_and_maybe_stop(self, elapsed_steps, elapsed_time, global_step):
    steps_per_sec = elapsed_steps / elapsed_time
    if self._summary_writer is not None:
      summary = tf.Summary(value=[tf.Summary.Value(
          tag=self._summary_tag, simple_value=steps_per_sec)])
      self._summary_writer.add_summary(summary, global_step)
    # tf.logging.info("%s: %g", self._summary_tag, steps_per_sec)

    if steps_per_sec < self._min_steps_per_sec:
      if self._next_infraction_will_stop:
        tf.logging.error("Global steps per sec (%g) was below minimum of %g continuously over %g steps.",
                        steps_per_sec, self._min_steps_per_sec, self._every_n_steps * 2)
        tf.logging.error("Exiting.")
        exit(1)
      else:
        tf.logging.info("Global steps per sec (%g) is below minimum of %g, if continued, process will exit.",
                        steps_per_sec, self._min_steps_per_sec)
        self._next_infraction_will_stop = True
    else:
      self._next_infraction_will_stop = False

  def after_run(self, run_context, run_values):
    _ = run_context

    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step+1):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
            global_step)
        if elapsed_time is not None:
          self._log_record_and_maybe_stop(elapsed_steps, elapsed_time, global_step)

    # Check whether the global step has been increased. Here, we do not use the
    # timer.last_triggered_step as the timer might record a different global
    # step value such that the comparison could be unreliable. For simplicity,
    # we just compare the stale_global_step with previously recorded version.
    if stale_global_step == self._last_global_step:
      # Here, we use a counter to count how many times we have observed that the
      # global step has not been increased. For some Optimizers, the global step
      # is not increased each time by design. For example, SyncReplicaOptimizer
      # doesn't increase the global step in worker's main train step.
      self._global_step_check_count += 1
      if self._global_step_check_count % 20 == 0:
        self._global_step_check_count = 0
        tf.logging.warning(
            "It seems that global step (tf.train.get_global_step) has not "
            "been increased. Current value (could be stable): %s vs previous "
            "value: %s. You could increase the global step by passing "
            "tf.train.get_global_step() to Optimizer.apply_gradients or "
            "Optimizer.minimize.", stale_global_step, self._last_global_step)
    else:
      # Whenever we observe the increment, reset the counter.
      self._global_step_check_count = 0

    self._last_global_step = stale_global_step