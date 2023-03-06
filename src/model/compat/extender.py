# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extenders of tf.estimator.Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import clip_ops
from tensorflow.python.training import optimizer as optimizer_lib


def clip_gradients_by_norm(optimizer, clip_norm):
  """Returns an optimizer which clips gradients before appliying them.

  Example:

  ```python
  optimizer = tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001)
  optimizer = tf.contrib.estimator.clip_gradients_by_norm(
      optimizer, clip_norm)
  estimator = tf.estimator.DNNClassifier(
      feature_columns=[...],
      hidden_units=[1024, 512, 256],
      optimizer=optimizer)
  ```

  Args:
    optimizer: An `tf.Optimizer` object to apply gradients.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.

  Returns:
    A `tf.Optimizer`.
  """

  def clip_grads(grads_and_vars):
    gradients, variables = zip(*grads_and_vars)
    gradients = clip_ops.clip_by_global_norm(gradients, clip_norm)[0]
    grads_and_vars = list(zip(gradients, variables))
    return grads_and_vars

  return _TransformGradients(
    optimizer=optimizer,
    transform_grads_fn=clip_grads,
    name='ClipByNorm' + optimizer.get_name())


class _TransformGradients(optimizer_lib.Optimizer):
  """Add given gradient transformation to the optimizer."""

  def __init__(self, optimizer, transform_grads_fn, name=None):
    """Construct an `tf.Optimizer` wrapper to apply given transformations.

    Example:

    ```python
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001)
    def clip_grads(grads_and_vars):
      gradients, variables = zip(*grads_and_vars)
      gradients = tf.clip_by_global_norm(grads, my_norm)[0]
      grads_and_vars = list(zip(gradients, variables))
      return grads_and_vars
    optimizer = _TransformGradients(
        opt=optimizer, transform_grads_fn=clip_grads)
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[...],
        hidden_units=[1024, 512, 256],
        optimizer=optimizer)
    ```

    Args:
      optimizer: An `tf.Optimizer` object to apply gradients.
      transform_grads_fn: A function which takes a single argument, a list of
        gradient to variable pairs (tuples), performs any requested gradient
        updates, such as gradient clipping or multipliers, and returns the
        updated list.
      name: A string which will be used for debugging purposes.
    """
    super(_TransformGradients, self).__init__(
      use_locking=False, name=name or optimizer.get_name())
    self._optimizer = optimizer
    self._transform_grads_fn = transform_grads_fn

  def compute_gradients(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Calls `transform_grads_fn`, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    grads_and_vars = self._transform_grads_fn(grads_and_vars)
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

  def get_slot(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.get_slot_names(*args, **kwargs)
