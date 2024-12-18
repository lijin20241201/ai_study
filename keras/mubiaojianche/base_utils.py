# 如果大于clip_value_max，则设置为clip_value_max。这样可以防止数值溢出或其他因数值范围过大或过小引起的问题。
# 这段代码展示了一个较为复杂的实现细节，包括对IndexedSlices类型的支持，这在处理稀疏数据时是非常有用的。然而，对于
# 大多数使用场景而言，我们通常不会直接看到这么底层的实现细节，而是直接调用tf.clip_by_value函数。
@tf_export("clip_by_value") # 用于将此函数导出到TensorFlow的公共API中。
@dispatch.register_unary_elementwise_api # 注册此函数为单参数逐元素操作API。
@dispatch.add_dispatch_support # 增加对操作的调度支持。
# 接受四个参数，其中t是要被裁剪的张量，clip_value_min和clip_value_max分别是裁剪的最小值和最大值，
# name是可选的操作名。
def clip_by_value(t, clip_value_min, clip_value_max,
                  name=None):
  # 创建一个命名空间，以便于跟踪和调试。
  with ops.name_scope(name, "clip_by_value",
                      [t, clip_value_min, clip_value_max]) as name:
    # 将输入t转换为张量。如果t本身是一个IndexedSlices对象（用于表示稀疏数据），
    # 那么就转换IndexedSlices的values属性，否则直接转换t
    values = ops.convert_to_tensor(
        t.values if isinstance(t, indexed_slices.IndexedSlices) else t,
        name="t")
    # 这里使用math_ops.minimum函数来确保values中的每个元素都不超过clip_value_max。
    # 也就是说，如果某个元素大于clip_value_max，则将该元素的值设置为clip_value_max。
    # values是一个张量，而clip_value_max可能是一个标量（如果是单一数值）或者是与values形状相同的张
    # 量。如果clip_value_max是一个标量，那么它会被广播至values的形状，这意味着clip_value_max的值将
    # 会与values中的每个元素进行比较。如果clip_value_max是一个张量，并且与values具有相同的形状，那
    # 么这两个张量将按元素进行比较。
    t_min = math_ops.minimum(values, clip_value_max)
    # Assert that the shape is compatible with the initial shape,
    # to prevent unintentional broadcasting.
    # values.shape.assert_is_compatible_with(t_min.shape) 是TensorFlow中的一种静态形状检查机制，
    # 用于确保两个张量的形状是兼容的。这通常用于防止无意的广播（broadcasting）操作，确保在执行某些操作
    # 之前，张量的形状是一致的。
    values.shape.assert_is_compatible_with(t_min.shape)
    t_max = math_ops.maximum(t_min, clip_value_min, name=name)
    # 在裁剪之后，使用shape.assert_is_compatible_with()来确保裁剪后的张量形状
    # 与原始张量形状兼容
    values.shape.assert_is_compatible_with(t_max.shape)
    # 如果输入t是一个IndexedSlices对象，则将裁剪后的张量t_max转换回IndexedSlices对象并返回。
    if isinstance(t, indexed_slices.IndexedSlices):
      t_max = indexed_slices.IndexedSlices(t_max, t.indices, t.dense_shape)
  return t_max
@tf_export(v1=["nn.sigmoid_cross_entropy_with_logits"])
@dispatch.add_dispatch_support
def sigmoid_cross_entropy_with_logits(
    labels=None,
    logits=None,
    name=None):
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", labels, logits)
  # pylint: enable=protected-access
  # y_true 是每个锚框的真实类别标签，采用 one-hot 编码形式。(b,total_anchors,num_classes)
  # y_pred 的形状：(batch_size, total_anchors, num_classes)。
  # 这意味着每个锚框都有一个长度为 num_classes 的向量，表示该锚框的真实类别标签（对于 y_true）
  # 或预测得分（对于 y_pred）。
  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    # 检查 labels 和 logits 的形状是否兼容
    try:
      labels.get_shape().assert_is_compatible_with(logits.get_shape())
    except ValueError:
      raise ValueError("`logits` and `labels` must have the same shape, "
                       f"received ({logits.get_shape()} vs "
                       f"{labels.get_shape()}).")
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)  # pylint: disable=invalid-unary-operand-type
    return math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)
@dispatch.add_dispatch_support
def sigmoid_cross_entropy_with_logits_v2(  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  r"""Computes sigmoid cross entropy given `logits`.
  Measures the probability error in tasks with two outcomes in which each
  outcome is independent and need not have a fully certain label. For instance,
  one could perform a regression where the probability of an event happening is
  known and used as a label. This loss may also be used for binary
  classification, where labels are either zero or one.

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `labels` must have the same type and shape.

  >>> logits = tf.constant([1., -1., 0., 1., -1., 0., 0.])
  >>> labels = tf.constant([0., 0., 0., 1., 1., 1., 0.5])
  >>> tf.nn.sigmoid_cross_entropy_with_logits(
  ...     labels=labels, logits=logits).numpy()
  array([1.3132617, 0.3132617, 0.6931472, 0.3132617, 1.3132617, 0.6931472,
         0.6931472], dtype=float32)

  Compared to the losses which handle multiple outcomes,
  `tf.nn.softmax_cross_entropy_with_logits` for general multi-class
  classification and `tf.nn.sparse_softmax_cross_entropy_with_logits` for more
  efficient multi-class classification with hard labels,
  `sigmoid_cross_entropy_with_logits` is a slight simplification for binary
  classification:

        sigmoid(x) = softmax([x, 0])[0]

  $$\frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + e^0}$$

  While `sigmoid_cross_entropy_with_logits` works for soft binary labels
  (probabilities between 0 and 1), it can also be used for binary classification
  where the labels are hard. There is an equivalence between all three symbols
  in this case, with a probability 0 indicating the second class or 1 indicating
  the first class:

  >>> sigmoid_logits = tf.constant([1., -1., 0.])
  >>> softmax_logits = tf.stack([sigmoid_logits, tf.zeros_like(sigmoid_logits)],
  ...                           axis=-1)
  >>> soft_binary_labels = tf.constant([1., 1., 0.])
  >>> soft_multiclass_labels = tf.stack(
  ...     [soft_binary_labels, 1. - soft_binary_labels], axis=-1)
  >>> hard_labels = tf.constant([0, 0, 1])
  >>> tf.nn.sparse_softmax_cross_entropy_with_logits(
  ...     labels=hard_labels, logits=softmax_logits).numpy()
  array([0.31326166, 1.3132616 , 0.6931472 ], dtype=float32)
  >>> tf.nn.softmax_cross_entropy_with_logits(
  ...     labels=soft_multiclass_labels, logits=softmax_logits).numpy()
  array([0.31326166, 1.3132616, 0.6931472], dtype=float32)
  >>> tf.nn.sigmoid_cross_entropy_with_logits(
  ...     labels=soft_binary_labels, logits=sigmoid_logits).numpy()
  array([0.31326166, 1.3132616, 0.6931472], dtype=float32)

  Args:
    labels: A `Tensor` of the same type and shape as `logits`. Between 0 and 1,
      inclusive.
    logits: A `Tensor` of type `float32` or `float64`. Any real number.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  return sigmoid_cross_entropy_with_logits(
      logits=logits, labels=labels, name=name)
