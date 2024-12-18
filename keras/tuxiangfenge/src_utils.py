@tf_export('image.convert_image_dtype') # 导出此函数供外部使用
@dispatch.register_unary_elementwise_api # 注册为一个单参数的逐元素操作API
@dispatch.add_dispatch_support # 添加支持动态调度的能力
def convert_image_dtype(image, dtype, saturate=False, name=None): # 定义函数及其参数
  image = ops.convert_to_tensor(image, name='image') # 将输入转换为TensorFlow中的Tensor类型
  dtype = dtypes.as_dtype(dtype)  # 将数据类型转换为TensorFlow内部使用的DType对象
  # 检查目标数据类型是否为浮点数或整数类型
  if not dtype.is_floating and not dtype.is_integer:
    raise AttributeError('dtype must be either floating point or integer') # 如果不是，则抛出异常
  # 如果 image 和 dtype 都是整数类型，那么函数将根据转换方向（放大或缩小）采用不同的策略来避
  if not image.dtype.is_floating and not image.dtype.is_integer:
    raise AttributeError('image dtype must be either floating point or integer')
  if dtype == image.dtype: # 如果输入和目标数据类型相同，则直接返回输入
    return array_ops.identity(image, name=name)
  # 使用命名空间管理操作名称  
  with ops.name_scope(name, 'convert_image', [image]) as name:
    # Both integer: use integer multiplication in the larger range
    if image.dtype.is_integer and dtype.is_integer: # 如果都是整数类型，则使用整数乘法来转换
      scale_in = image.dtype.max # 获取输入数据类型的最大值
      scale_out = dtype.max # 获取目标数据类型的最大值
      if scale_in > scale_out:# 如果输入范围大于目标范围，则先缩小范围再转换
        # Scaling down, scale first, then cast. The scaling factor will
        # cause in.max to be mapped to above out.max but below out.max+1,
        # so that the output is safely in the supported range.
        
        scale = (scale_in + 1) // (scale_out + 1) # 计算缩放因子
        scaled = math_ops.floordiv(image, scale)  # 应用整数除法
        if saturate: # 如果启用饱和转换，则使用饱和转换
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)  # 否则直接转换
      else: # 如果输入范围小于等于目标范围，则先转换再放大
        # Scaling up, cast first, then scale. The scale will not map in.max to
        # out.max, but converting back and forth should result in no change.
        if saturate: # 如果启用饱和转换，则使用饱和转换
          cast = math_ops.saturate_cast(image, dtype)
        else: 
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1) # 计算缩放因子
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoring float overflows. If your image dynamic range
      # exceeds float range, you're on your own.
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        # Converting to float: first cast, then scale. No saturation possible.
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.multiply(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.multiply(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
def shape_list2(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    # 如果 tensor 是 np.ndarray 类型，直接返回其形状列表
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)
    # 使用 tf.shape(tensor) 获取张量的动态形状。动态形状是一个 Tensor，可以在运行时确定。
    dynamic = tf.shape(tensor)
    # 如果静态形状未知（即 tensor.shape == tf.TensorShape(None)），则直接返回动态形状。
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    # 如果静态形状已知，但某些维度的大小未知（即 None），则使用动态形状来替代未知的尺寸。
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
# stable_softmax 主要是在 CPU 上使用 XLA 时作为解决数值稳定性问题的手段，但它同样可以作为通用的稳健实现
# 用于其他场景。如果你的应用程序遇到了与 softmax 相关的数值稳定性问题，无论是在 CPU 还是 GPU 上，都可以
# 尝试使用这个函数作为解决方案。
def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    # 增加小常数：通过在 logits 上加上一个极小的常数（如 1e-9），可以避免在某些极端情况下的数值不稳定问题。
    # 保持接口一致：该函数的输入和输出与 tf.nn.softmax 保持一致，使得它可以作为替代方案直接使用。
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)
class Conv(Layer):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.rank = rank

        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. "
                "Expected a strictly positive value. "
                f"Received filters={filters}."
            )
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, "kernel_size"
        )
        self.strides = conv_utils.normalize_tuple(
            strides, rank, "strides", allow_zero=True
        )
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == "causal"
        self._channels_first = self.data_format == "channels_first"
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2
        )

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                "number of groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). Received: %s"
                % (self.kernel_size,)
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0(s). Received: %s"
                % (self.strides,)
            )

        if self.padding == "causal":

            from keras.src.layers.convolutional.conv1d import Conv1D
            from keras.src.layers.convolutional.separable_conv1d import (
                SeparableConv1D,
            )

            if not isinstance(self, (Conv1D, SeparableConv1D)):
                raise ValueError(
                    "Causal padding is only supported for `Conv1D`"
                    "and `SeparableConv1D`."
                )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                "the number of groups. Received groups={}, but the input "
                "has {} channels (full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        self.built = True

    def convolution_op(self, inputs, kernel):
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
        )

    # TODO(b/213173659): remove this when grouped convolutions are fully
    # supported on the CPU for compiled functions. For now, we need this as a
    # workaround for CPU support.
    @tf.function(jit_compile=True)
    def _jit_compiled_convolution_op(self, inputs, kernel):
        return self.convolution_op(inputs, kernel)

    def call(self, inputs):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        if self.groups > 1:
            outputs = self._jit_compiled_convolution_op(
                inputs, tf.convert_to_tensor(self.kernel)
            )
        else:
            outputs = self.convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(
                            o, self.bias, data_format=self._tf_data_format
                        )

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1
                    )
                else:
                    outputs = tf.nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format
                    )

        if not tf.executing_eagerly() and input_shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format == "channels_last":
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters]
                )
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + [self.filters]
                    + self._spatial_output_shape(input_shape[batch_rank + 1 :])
                )

        except ValueError:
            raise ValueError(
                "One of the dimensions in the output is <= 0 "
                f"due to downsampling in {self.name}. Consider "
                "increasing the input size. "
                f"Received input shape {input_shape} which would produce "
                "output shape with a zero or negative value in a "
                "dimension."
            )

    def _recreate_conv_op(self, inputs):
        return False

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, "ndims", None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == "channels_last":
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == "causal":
            op_padding = "valid"
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding
# unpack_inputs 装饰器实现非常详细，它不仅处理了输入参数的解包，还考虑了函数签名的保持，这对
# 于确保装饰器不会影响原有函数的行为非常重要。
# func 在这里指的是被装饰的函数本身。当使用装饰器语法（如 @unpack_inputs）时，装饰器函数会
# 接收被装饰的函数作为参数。在这个上下文中，func 就是你想要应用解包逻辑的函数，通常是类的方法，
# 比如 TFSegformerMainLayer 类中的 call 方法。
# unpack_inputs 装饰器的主要作用是对被装饰的函数（如 call 方法）的输入进行解包处理。它确保输入
# 参数以适当的形式传递给被装饰的函数，即使输入是以字典或元组等形式提供的
def unpack_inputs(func):
    # 获取原始函数签名,这里使用 inspect.signature 获取原始函数的签名信息，包括参数名称和类型等。
    original_signature = inspect.signature(func)
    # 定义装饰器内部函数,使用 functools.wraps 装饰器来确保装饰后的函数保留原始函数的元数据，
    # 如文档字符串、名称等。
    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        # 分离实际的 **kwargs,这里分离出了那些不在原始函数签名中的键值对，这些键值对可能是一些
        # 额外的参数，需要特殊处理。
        # isolates the actual `**kwargs` for the decorated function
        kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
        # 收集函数的参数和关键字参数,这部分代码收集了原始函数所需的参数和关键字参数，并将分离出来的 kwargs_call 
        # 作为一个键值对添加到 fn_args_and_kwargs 中。
        fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
        # 将位置参数转换为关键字参数,将位置参数转换为关键字参数，并更新到 fn_args_and_kwargs 中。注意，
        # 这里假设第一个参数是 self，因此从第二个参数开始处理。
        fn_args_and_kwargs.update({"kwargs_call": kwargs_call})
        # move any arg into kwargs, if they exist
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))
        # Encoder Decoder models delegate the application of the configuration options to their inner models.
        # 这部分代码处理了特殊情况，即当类名为 EncoderDecoder 的时候，不使用配置；否则使用 self.config。
        if "EncoderDecoder" in self.__class__.__name__:
            config = None
        else:
            config = self.config
        # 这里调用 input_processing 函数来处理输入参数，input_processing 函数应该负责将输入
        # 参数解包并准备成适合 func 调用的形式。
        unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
        # 最后，使用解包后的参数调用原始函数，并返回结果。
        return func(self, **unpacked_inputs)
    # Keras enforces the first layer argument to be passed, and checks it through `inspect.getfullargspec()`. This
    # function does not follow wrapper chains (i.e. ignores `functools.wraps()`), meaning that without the line below
    # Keras would attempt to check the first argument against the literal signature of the wrapper.
    # 为了确保 Keras 正确识别函数的第一个参数，这里显式设置了装饰后函数的签名为原始签名。
    run_call_with_unpacked_inputs.__signature__ = original_signature
    return run_call_with_unpacked_inputs
class TFModelUtilsMixin:
    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        else:
            return self.count_params()
def keras_serializable(cls):
    initializer = cls.__init__
    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")
    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)
        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        self._config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on keras.layers.Layer subclasses")
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg["config"] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(keras.utils, "register_keras_serializable"):
        cls = keras.utils.register_keras_serializable()(cls)
    return cls
class TFCausalLanguageModelingLoss:
    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100 affect the loss
            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)
        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 affect the loss
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))
class TFQuestionAnsweringLoss:
    # 适合问答的损失函数
    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        start_loss = loss_fn(labels["start_position"], logits[0])
        end_loss = loss_fn(labels["end_position"], logits[1])
        return (start_loss + end_loss) / 2.0

class TFTokenClassificationLoss:
    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if tf.executing_eagerly():  # Data-dependent conditionals are forbidden in XLA
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")

        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100
            # are taken into account as loss
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
                active_loss = tf.reshape(labels, (-1,)) != -1
            else:
                active_loss = tf.reshape(labels, (-1,)) != -100
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

            return loss_fn(labels, reduced_logits)
        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 or -1
        # are taken into account as loss
        loss_mask = tf.cast(labels >= 0, dtype=unmasked_loss.dtype)
        # Avoid possible division by zero later
        # Masked positions will have a loss of NaN because -100 and -1 are not valid labels
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))
class TFSequenceClassificationLoss:
    def hf_compute_loss(self, labels, logits):
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                # MeanSquaredError returns a scalar loss if the labels are 1D, so avoid that
                labels = tf.expand_dims(labels, axis=-1)
        else:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)

class TFMultipleChoiceLoss:
   
    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        return loss_fn(labels, logits)


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):

class TFNextSentencePredictionLoss:

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100
            # are taken into account as loss
            next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
            next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)

            return loss_fn(next_sentence_label, next_sentence_reduced_logits)

        # make sure only labels that are not equal to -100
        # are taken into account as loss

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        ns_loss_mask = tf.cast(labels != -100, dtype=unmasked_ns_loss.dtype)
        # Just zero out samples where label is -100, no reduction
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        return masked_ns_loss


def booleans_processing(config, **kwargs):
    final_booleans = {}

    # Pure conv models (such as ConvNext) do not have `output_attentions`. If the signature has
    # `output_attentions`, it will be present here in `kwargs`, even if unset (in that case, as `None`)
    if "output_attentions" in kwargs:
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
    final_booleans["output_hidden_states"] = (
        kwargs["output_hidden_states"] if kwargs["output_hidden_states"] is not None else config.output_hidden_states
    )
    final_booleans["return_dict"] = kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict

    if "use_cache" in kwargs:
        final_booleans["use_cache"] = (
            kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(config, "use_cache", None)
        )
    return final_booleans


def unpack_inputs(func):

    original_signature = inspect.signature(func)

    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        # isolates the actual `**kwargs` for the decorated function
        kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
        fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
        fn_args_and_kwargs.update({"kwargs_call": kwargs_call})

        # move any arg into kwargs, if they exist
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))

        # Encoder Decoder models delegate the application of the configuration options to their inner models.
        if "EncoderDecoder" in self.__class__.__name__:
            config = None
        else:
            config = self.config

        unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
        return func(self, **unpacked_inputs)

    # Keras enforces the first layer argument to be passed, and checks it through `inspect.getfullargspec()`. This
    # function does not follow wrapper chains (i.e. ignores `functools.wraps()`), meaning that without the line below
    # Keras would attempt to check the first argument against the literal signature of the wrapper.
    run_call_with_unpacked_inputs.__signature__ = original_signature

    return run_call_with_unpacked_inputs


def input_processing(func, config, **kwargs):
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop("kwargs", None))
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    main_input_name = parameter_names[0]
    main_input = kwargs.pop(main_input_name, None)
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
            " `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values`"
            " instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")

    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                "The following keyword arguments are not supported by this model:"
                f" {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(main_input, (tuple, list)):
        for i, input in enumerate(main_input):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if is_tf_symbolic_tensor(input):
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    elif isinstance(main_input, Mapping):
        if "inputs" in main_input:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids`"
                " instead.",
                FutureWarning,
            )

            output["input_ids"] = main_input.pop("inputs")

        if "decoder_cached_states" in main_input:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = main_input.pop("decoder_cached_states")

        for k, v in dict(main_input).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if tf.is_tensor(main_input) or main_input is None:
            output[main_input_name] = main_input
        else:
            raise ValueError(
                f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
                f" {main_input_name}."
            )

    # Populates any unspecified argument with their default value, according to the signature.
    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and is_tf_symbolic_tensor(output["args"]):
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    cast_output = {}
    for key, val in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            cast_output[key] = val.astype(np.int32)
        else:
            cast_output[key] = val

    output = cast_output
    del cast_output

    if config is not None:
        boolean_dict = {
            k: v
            for k, v in output.items()
            if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
        }

        output.update(
            booleans_processing(
                config=config,
                **boolean_dict,
            )
        )

    return output


def dtype_byte_size(dtype):
   
    if dtype == tf.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def strip_model_name_and_prefix(name, _prefix=None):
    if _prefix is not None and name.startswith(_prefix):
        name = name[len(_prefix) :]
        if name.startswith("/"):
            name = name[1:]
    if "model." not in name and len(name.split("/")) > 1:
        name = "/".join(name.split("/")[1:])
    return name


def tf_shard_checkpoint(weights, max_shard_size="10GB", weights_name: str = TF2_WEIGHTS_NAME):
   
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = []
    current_block_size = 0
    total_size = 0

    for item in weights:
        weight_size = item.numpy().size * dtype_byte_size(item.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = []
            current_block_size = 0

        current_block.append(item)
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".h5", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.h5")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for weight in shard:
            weight_name = weight.name
            weight_map[weight_name] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_tf_sharded_weights(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    # Load the index
    unexpected_keys = set()
    saved_keys = set()
    mismatched_keys = set()

    # Since TF adds the name of the class to its weights, and uses the index and not the name of the layer to load
    # the weight, we have to get rid of the first prefix of the name of the layer.
    model_keys = set()
    model_layer_map = {}
    for i, k in enumerate(model.weights):
        layer_name = k.name
        if _prefix is not None and layer_name.startswith(_prefix):
            layer_name = layer_name[len(_prefix) :]
            layer_name = layer_name.lstrip("/")
        if not ("model." in layer_name or len(layer_name.split("/")) == 1):
            layer_name = "/".join(layer_name.split("/")[1:])
        model_keys.add(layer_name)
        model_layer_map[layer_name] = i

    for shard_file in shard_files:
        saved_weight_names_set, unexpected_keys_set, mismatched_keys_set = load_tf_shard(
            model,
            model_layer_map,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        saved_keys.update(saved_weight_names_set)
        unexpected_keys.update(unexpected_keys_set)
        mismatched_keys.update(mismatched_keys_set)
        gc.collect()

    missing_keys = model_keys - saved_keys
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    return missing_keys, unexpected_keys, mismatched_keys


def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
   
    saved_weight_names_set = set()
    saved_weights = {}
    mismatched_keys = set()
    unexpected_keys = set()
    # Read the H5 file
    try:
        with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
            # Retrieve the name of each layer from the H5 file
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))
            weight_value_tuples = []

            # Compute missing and unexpected sub layers
            # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
            for layer_name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer_name]
                saved_weights[layer_name] = np.asarray(h5_layer_object)

                saved_weight_names_set.add(layer_name)

                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    symbolic_weight = model.weights[model_layer_map[layer_name]]

                    saved_weight_value = saved_weights[layer_name]
                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add(
                                        (layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                    # We create the tuple that will be loaded and add it to the final list
                    weight_value_tuples.append((symbolic_weight, array))

        K.batch_set_value(weight_value_tuples)

        return saved_weight_names_set, unexpected_keys, mismatched_keys

    except Exception as e:
        try:
            with open(resolved_archive_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained"
                        " model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' "
                f"at '{resolved_archive_file}'. "
                "If you tried to load a TF model from a sharded checkpoint, you should try converting the model "
                "by loading it in pytorch and saving it localy. A convertion script should be realeased soon."
            )


def load_tf_sharded_weights_from_safetensors(
    model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None
):
    # Load the index
    unexpected_keys = set()
    all_missing_keys = []
    mismatched_keys = set()

    for shard_file in shard_files:
        missing_layers, unexpected_layers, mismatched_layers = load_tf_weights_from_safetensors(
            model,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        all_missing_keys.append(set(missing_layers))
        unexpected_keys.update(unexpected_layers)
        mismatched_keys.update(mismatched_layers)
        gc.collect()
    missing_keys = set.intersection(*all_missing_keys)

    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    return missing_keys, unexpected_keys, mismatched_keys


def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    
    if resolved_archive_file.endswith(".safetensors"):
        load_function = load_tf_weights_from_safetensors
    else:
        load_function = load_tf_weights_from_h5

    return load_function(
        model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix
    )


def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    mismatched_layers = []

    # Read the H5 file
    with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
        # Retrieve the name of each layer from the H5 file
        saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))

        # Find the missing layers from the high level list of layers
        missing_layers = list({layer.name for layer in model.layers} - saved_h5_model_layers_name)

        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(saved_h5_model_layers_name - {layer.name for layer in model.layers})
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []

        # Compute missing and unexpected sub layers
        # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
        for layer in model.layers:
            # if layer_name from the H5 file belongs to the layers from the instantiated model
            if layer.name in saved_h5_model_layers_name:
                # Get the H5 layer object from its name
                h5_layer_object = sharded_checkpoint_file[layer.name]
                # Get all the weights as a list from the layer object
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}

                # Create a dict from the H5 saved model that looks like {"weight_name": weight_value}
                # And a set with only the names
                for weight_name in load_attributes_from_hdf5_group(h5_layer_object, "weight_names"):
                    # TF names always start with the model name so we ignore it
                    name = "/".join(weight_name.split("/")[1:])

                    if _prefix is not None:
                        name = _prefix + "/" + name

                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])

                    # Add the updated name to the final list for computing missing/unexpected values
                    saved_weight_names_set.add(name)

                # Loop over each weights from the instantiated model and compare with the weights from the H5 file
                for symbolic_weight in symbolic_weights:
                    # TF names always start with the model name so we ignore it
                    if _prefix is not None:
                        delimeter = len(_prefix.split("/"))
                        symbolic_weight_name = "/".join(
                            symbolic_weight.name.split("/")[:delimeter]
                            + symbolic_weight.name.split("/")[delimeter + 1 :]
                        )
                    else:
                        symbolic_weight_name = "/".join(symbolic_weight.name.split("/")[1:])

                    # here we check if the current weight is among the weights from the H5 file
                    # If yes, get the weight_value of the corresponding weight from the H5 file
                    # If not, make the value to None
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Retrocompatibility patch: some embeddings are stored with the weights name (e.g. Bart's
                    # `model.shared/embeddings:0` are stored as `model.shared/weights:0`)
                    if saved_weight_value is None and symbolic_weight_name.endswith("embeddings:0"):
                        symbolic_weight_name = symbolic_weight_name[:-12] + "weight:0"
                        saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Add the updated name to the final list for computing missing/unexpected values
                    symbolic_weights_names.add(symbolic_weight_name)

                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_layers.append(
                                        (symbolic_weight_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                        # We create the tuple that will be loaded and add it to the final list
                        weight_value_tuples.append((symbolic_weight, array))

    # Load all the weights
    K.batch_set_value(weight_value_tuples)

    # Compute the missing and unexpected layers
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    return missing_layers, unexpected_layers, mismatched_layers


def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # Read the safetensors file
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        mismatched_layers = []
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        # Find the missing layers from the high level list of layers
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))

        for weight in model.weights:
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                # Check if the shape of the current weight and the one from the H5 file are different
                if K.int_shape(weight) != weight_value.shape:
                    # If yes we reshape the weight from the H5 file accordingly to the current weight
                    # If the two shapes are not compatible we raise an issue
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        if ignore_mismatched_sizes:
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e

                K.set_value(weight, weight_value)  # weight.assign() might break if weight is a DTensor
    return missing_layers, unexpected_layers, mismatched_layers


def init_copy_embeddings(old_embeddings, new_num_tokens):
    
    old_num_tokens, old_embedding_dim = shape_list(old_embeddings)
    size_diff = new_num_tokens - old_num_tokens

    # initialize new embeddings
    # Copy token embeddings from the previous ones
    if tf.math.greater(size_diff, 0):
        # if the new size is greater than the old one, we extend the current embeddings with a padding until getting new size
        # and we create a mask to properly identify the padded values and be replaced by the values of the newly created
        # embeddings
        current_weights = tf.pad(
            old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
        )
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        # if the new size if lower than the old one, we take the current embeddings until the new size
        current_weights = tf.slice(
            old_embeddings.value(),
            tf.convert_to_tensor([0, 0]),
            tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
        )
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

    return mask, current_weights
class TFSharedEmbeddings(keras.layers.Layer):

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size**-0.5 if initializer_range is None else initializer_range
        warnings.warn(
            "`TFSharedEmbeddings` is scheduled for deletion in v4.32, use `keras.layers.Embedding` instead.",
            DeprecationWarning,
        )

    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class TFSequenceSummary(keras.layers.Layer):
    def __init__(self, config: PretrainedConfig, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)

        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        self.has_activation = False
        activation_string = getattr(config, "summary_activation", None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)

        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = keras.layers.Dropout(config.summary_first_dropout)

        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = keras.layers.Dropout(config.summary_last_dropout)
        self.hidden_size = config.hidden_size

    def call(self, inputs, cls_index=None, training=False):
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        else:
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == "cls_index":
            hidden_shape = shape_list(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)
            # else:
            # cls_index = cls_index[..., tf.newaxis]
            # cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        if self.has_summary:
            output = self.summary(output)

        if self.has_activation:
            output = self.activation(output)

        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        if getattr(self, "summary", None) is not None:
            with tf.name_scope("summary"):
                self.summary.build(self.hidden_size)


def get_initializer(initializer_range: float = 0.02) -> keras.initializers.TruncatedNormal:
    """
    Creates a `keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return keras.initializers.TruncatedNormal(stddev=initializer_range)
class TFConv1D(keras.layers.Layer):
    
    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x