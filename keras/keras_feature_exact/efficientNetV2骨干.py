def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )
def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))
BN_AXIS = 3 # 特征轴
CONV_KERNEL_INITIALIZER = { # 核初始化
    "class_name": "VarianceScaling", # 方差比例
    "config": { 
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}
# 用标准卷积替换了MBConv中的扩张点卷积+深度卷积
@keras_cv_export("keras_cv.layers.FusedMBConvBlock")
class FusedMBConvBlock(keras.layers.Layer):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="swish",
        survival_probability: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.survival_probability = survival_probability
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))
        # 具有指定内核大小和步长的标准卷积，而不是1x1 扩张卷积(这里用扩张通道的标准卷积替换了
        # MBconv中的扩张点卷积和深度卷积)
        self.conv1 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "expand_conv",
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.bn_momentum,
            name=self.name + "expand_bn",
        )
        self.act = keras.layers.Activation(
            self.activation, name=self.name + "expand_activation"
        )

        self.bn2 = keras.layers.BatchNormalization(
            axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "bn"
        )
        # 挤压点卷积,切换到挤压通道数
        self.se_conv1 = keras.layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_reduce",
        )
        # 激励点卷积,切换到扩张通道数
        self.se_conv2 = keras.layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_expand",
        )
        # 看扩张比例,如果是1,应该是普通卷积,否则是点卷积,压缩通道数到output_filters
        self.output_conv = keras.layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "project_conv",
        )
        self.bn3 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.bn_momentum,
            name=self.name + "project_bn",
        )
        # V2自有的dropout机制,随网络深度增加,逐渐加大
        if self.survival_probability:
            self.dropout = keras.layers.Dropout(
                self.survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=self.name + "drop",
            )
    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        # 扩张阶段,如果expand_ratio== 1,不改变,否则用扩张标准卷积,而不是MBConv中的先进行 1x1 
        # 扩张卷积，之后深度卷积,这时用的是标准卷积
        if self.expand_ratio != 1:
            x = self.conv1(inputs) # 扩张标准卷积
            x = self.bn1(x)
            x = self.act(x)
        else:
            x = inputs
        # se块(使用的条件是0<se_ratio <= 1)
        if 0 < self.se_ratio <= 1:
            # 全局平均池化
            se = keras.layers.GlobalAveragePooling2D(
                name=self.name + "se_squeeze"
            )(x)
            if BN_AXIS == 1:
                se_shape = (self.filters, 1, 1)
            else:
                se_shape = (1, 1, self.filters)
            # 变形
            se = keras.layers.Reshape(se_shape, name=self.name + "se_reshape")(
                se
            )
            # 压缩激励,用sigmoid打分
            se = self.se_conv1(se)
            se = self.se_conv2(se)
            # 通道加权
            x = keras.layers.multiply([x, se], name=self.name + "se_excite")
        # 输出阶段,如果expand_ratio 等于 1,使用普通卷积,否则使用点卷积降低通道数，并进行批量归一化。如果 
        # expand_ratio 等于 1时，则在输出卷积后应用激活函数。
        x = self.output_conv(x)
        x = self.bn3(x)
        if self.expand_ratio == 1:
            x = self.act(x)
        # 残差,条件是strides == 1 and input_filters == output_filters
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.survival_probability: # dropout(基于深度可变)
                x = self.dropout(x)
            # 这里把经过FusedMBConv处理的数据和传入的数据残差
            x = keras.layers.Add(name=self.name + "add")([x, inputs])
        return x
    # 获取配置
    def get_config(self):
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "survival_probability": self.survival_probability,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
# MBConv块是在面向移动设备和高效的架构中常用的模块，出现在诸如MobileNet、EfficientNet、MaxViT等架构中。
# MBConv块遵循窄-宽-窄的结构——通过扩展点卷积，应用深度卷积，然后缩小回压缩点卷积，这比传统的宽-窄-宽结构更
# 有效率。由于这些模块经常用于部署到边缘设备的模型中，因此为了便于使用和复用，我们将其实现为一个层。
# @keras_cv_export("keras_cv.layers.MBConvBlock")
class MBConvBlock(keras.layers.Layer):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="swish",
        survival_probability: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.survival_probability = survival_probability
        self.filters = self.input_filters * self.expand_ratio # 扩张通道数
        self.filters_se = max(1, int(input_filters * se_ratio)) # 挤压通道数
        # 扩张点卷积
        self.conv1 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "expand_conv",
        )
        # 批次标准化
        self.bn1 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.bn_momentum,
            name=self.name + "expand_bn",
        )
        self.act = keras.layers.Activation( # 激活函数
            self.activation, name=self.name + "activation"
        )
        self.depthwise = keras.layers.DepthwiseConv2D( # 深度卷积
            kernel_size=self.kernel_size,
            strides=self.strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "dwconv2",
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "bn"
        )
        # 挤压点卷积
        self.se_conv1 = keras.layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_reduce",
        )
        # 激励点卷积,sigmoid会给每个通道单独打分
        self.se_conv2 = keras.layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_expand",
        )
        
        self.output_conv = keras.layers.Conv2D(
            filters=self.output_filters,
            # 这意味着扩张比例是1时,核大小可以不是1,那就是普通卷积,否则是压缩点卷积
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "project_conv",
        )

        self.bn3 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.bn_momentum,
            name=self.name + "project_bn",
        )
        # dropout,noise_shape=(None, 1, 1, 1)
        # 在实际应用中，选择哪种 noise_shape 取决于具体的应用场景和网络架构的特点：
        # 对于特征图的空间一致性要求较高的任务：比如物体检测、分割等任务，通常需要特征图在空间上具有一致性，
        # 这时使用 noise_shape=(None, 1, 1, 1) 更合适。
        # 对于需要更高随机性的任务：如果任务要求每个像素位置上的特征都要尽可能独立地随机处理，那么可以使用
        # 默认的 noise_shape=None。
        if self.survival_probability:
            self.dropout = keras.layers.Dropout(
                self.survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=self.name + "drop",
            )
    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")
    
    def call(self, inputs): # inputs:(b,h,w,c)
        # 扩张阶段,如果扩张比例不等于1,用扩张点卷积,否则不改变
        if self.expand_ratio != 1:
            # 扩张卷积块
            x = self.conv1(inputs) 
            x = self.bn1(x)
            x = self.act(x)
        else:
            x = inputs
        # 深度卷积块
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        # se块,全局平均池化,获取通道描述符,之后压缩激励,打分加权
        if 0 < self.se_ratio <= 1:
            se = keras.layers.GlobalAveragePooling2D(
                name=self.name + "se_squeeze"
            )(x)
            if BN_AXIS == 1:
                se_shape = (self.filters, 1, 1)
            else:
                se_shape = (1, 1, self.filters)
            se = keras.layers.Reshape(se_shape, name=self.name + "se_reshape")(
                se
            )
            se = self.se_conv1(se)
            se = self.se_conv2(se)
            x = keras.layers.multiply([x, se], name=self.name + "se_excite")
        # 输出阶段,一般是压缩点卷积
        x = self.output_conv(x)
        x = self.bn3(x)
        # 残差连接,条件是步长是1,并且输入输出通道数相同
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.survival_probability:
                x = self.dropout(x)
            x = keras.layers.Add(name=self.name + "add")([x, inputs])
        # 步长为2,下采样时,不残差
        return x
    def get_config(self):
        # 子类特有的配置(字典)
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "survival_probability": self.survival_probability,
        }
        # 获取父类的配置
        base_config = super().get_config() 
        # 返回合并的配置
        return dict(list(base_config.items()) + list(config.items()))
# 获取conv构造函数,如果传入"unfused",就使用MBConvBlock,否则使用FusedMBConvBlock
def get_conv_constructor(conv_type):
    if conv_type == "unfused": # 扩张点卷积和深度卷积组合,这个用于网络的后部分
        return MBConvBlock
    elif conv_type == "fused": # 融合,就是扩张标准卷积,这个用于网络的前部分
        return FusedMBConvBlock
    else:
        raise ValueError(
            "Expected `conv_type` to be "
            "one of 'unfused', 'fused', but got "
            f"`conv_type={conv_type}`"
        )

# 这个注解用于提供外部访问这个类的导入路径
@keras_cv_export("keras_cv.models.EfficientNetV2Backbone")
class EfficientNetV2Backbone(Backbone):
    def __init__(
        self,
        *,
        include_rescaling,
        width_coefficient,
        depth_coefficient,
        stackwise_kernel_sizes,
        stackwise_num_repeats,
        stackwise_input_filters,
        stackwise_output_filters,
        stackwise_expansion_ratios,
        stackwise_squeeze_and_excite_ratios,
        stackwise_strides,
        stackwise_conv_types,
        skip_connection_dropout=0.2,
        depth_divisor=8,
        min_depth=8,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # 确定合适的输入
        img_input = utils.parse_model_inputs(input_shape, input_tensor)
        x = img_input
        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255.0)(x) # 归一化
        # 规范化卷积通道数(一般是8的倍数)
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        # 第一个下采样块(112,112,3)
        x = keras.layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=conv_kernel_initializer(),
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9,
            name="stem_bn",
        )(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)
        
        # 提取块的块索引
        block_id = 0
        blocks = float( # 总的提取块数
            sum(num_repeats for num_repeats in stackwise_num_repeats)
        )
        # 金字塔层级的特征图层名列表
        pyramid_level_inputs = []
        # 遍历每个层级
        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i] # 指定层级的重复提取次数
            input_filters = stackwise_input_filters[i] # 卷积的输入通道数
            output_filters = stackwise_output_filters[i] # 卷积的输出通道数
            # 规范化输入输出通道数
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )
            # 规范化重复次数
            repeats = round_repeats(
                repeats=num_repeats,
                depth_coefficient=depth_coefficient,
            )
            strides = stackwise_strides[i] # 步长
            squeeze_and_excite_ratio = stackwise_squeeze_and_excite_ratios[i] # 挤激比
            # 遍历一个层级下的每个提取块
            for j in range(repeats):
                # 如果不是第一个提取块,这时步长是1,并且输入和输出通道数相同
                if j > 0:
                    strides = 1
                    input_filters = output_filters
                # 步长等于2,这时把之前的特征图加入FPN
                if strides != 1:
                    pyramid_level_inputs.append(utils.get_tensor_input_name(x))
                # 用来设置提取块前缀,a...
                letter_identifier = chr(j + 97)
                block = get_conv_constructor(stackwise_conv_types[i])(
                    input_filters=input_filters,
                    output_filters=output_filters,
                    expand_ratio=stackwise_expansion_ratios[i], # 扩张比例
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    se_ratio=squeeze_and_excite_ratio, # 挤激比
                    activation=activation,
                    # dropout比率,随着层级的加深,比率变大
                    survival_probability=skip_connection_dropout
                    * block_id
                    / blocks,
                    bn_momentum=0.9,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                # 通过提取块处理
                x = block(x)
                block_id += 1 # 块计数器+1
        # 规范化顶部输出通道数
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient, # 宽度系数
            min_depth=min_depth, 
            depth_divisor=depth_divisor, # 深度因子
        )
        # 点卷积切换通道
        x = keras.layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer(),
            padding="same", # 填充
            data_format="channels_last",
            use_bias=False,
            name="top_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9,
            name="top_bn",
        )(x)
        x = keras.layers.Activation(
            activation=activation, name="top_activation"
        )(x)
        # FPN特征提取层列表
        pyramid_level_inputs.append(utils.get_tensor_input_name(x))

        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)
        # 设置实例属性
        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.skip_connection_dropout = skip_connection_dropout
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.activation = activation
        self.input_tensor = input_tensor
        # FPN特征提取字典,idx-->name
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
        # 各个层级的配置信息。
        self.stackwise_kernel_sizes = stackwise_kernel_sizes
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_input_filters = stackwise_input_filters
        self.stackwise_output_filters = stackwise_output_filters
        self.stackwise_expansion_ratios = stackwise_expansion_ratios
        self.stackwise_squeeze_and_excite_ratios = (
            stackwise_squeeze_and_excite_ratios
        )
        self.stackwise_strides = stackwise_strides
        self.stackwise_conv_types = stackwise_conv_types
    # 配置,用于序列化和反序列化
    def get_config(self):
        # 获取父类的配置对象,字典形式
        config = super().get_config()
        config.update( # 更新子类独有的设置
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "skip_connection_dropout": self.skip_connection_dropout,
                "depth_divisor": self.depth_divisor,
                "min_depth": self.min_depth,
                "activation": self.activation,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_squeeze_and_excite_ratios": self.stackwise_squeeze_and_excite_ratios,  # noqa: E501
                "stackwise_strides": self.stackwise_strides,
                "stackwise_conv_types": self.stackwise_conv_types,
            }
        )
        return config
    # presets 是一个类属性，用于存储预设的配置信息
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    # 类属性,预设的权重
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)

backbone_presets_no_weights = {
    "efficientnetv2_s": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional blocks."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_s/2",  # noqa: E501
    },
    "efficientnetv2_m": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional blocks."
            ),
            "params": 53150388,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_m/2",  # noqa: E501
    },
    "efficientnetv2_l": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional "
                "blocks, but more filters the in `efficientnetv2_m`."
            ),
            "params": 117746848,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_l/2",  # noqa: E501
    },
    "efficientnetv2_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b0/2",  # noqa: E501
    },
    "efficientnetv2_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b1/2",  # noqa: E501
    },
    "efficientnetv2_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b2/2",  # noqa: E501
    },
    "efficientnetv2_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 12930622,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b3/2",  # noqa: E501
    },
}
backbone_presets_with_weights = {
    "efficientnetv2_s_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional "
                "blocks. Weights are initialized to pretrained imagenet "
                "classification weights.Published weights are capable of "
                "scoring 83.9%top 1 accuracy "
                "and 96.7% top 5 accuracy on imagenet."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_s_imagenet/2",  # noqa: E501
    },
    "efficientnetv2_b0_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights. "
                "Published weights are capable of scoring 77.1%	top 1 accuracy "
                "and 93.3% top 5 accuracy on imagenet."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b0_imagenet/2",  # noqa: E501
    },
    "efficientnetv2_b1_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 79.1%	top 1 accuracy "
                "and 94.4% top 5 accuracy on imagenet."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b1_imagenet/2",  # noqa: E501
    },
    "efficientnetv2_b2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`. "
                "Weights are initialized to pretrained "
                "imagenet classification weights."
                "Published weights are capable of scoring 80.1%	top 1 accuracy "
                "and 94.9% top 5 accuracy on imagenet."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b2_imagenet/2",  # noqa: E501
    },
}
# 通过合并 backbone_presets_no_weights 和 backbone_presets_with_weights，形成一个完整的字典，包含所有预设配置。
backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
# 这是一个装饰器，用于指定该类将在外部如何被导入。这意味着用户可以按照路径 
# keras_cv.models.EfficientNetV2B0Backbone 来导入这个类。
@keras_cv_export("keras_cv.models.EfficientNetV2B0Backbone")
class EfficientNetV2B0Backbone(EfficientNetV2Backbone):
    # 这是一个构造函数的替代方法，在对象创建之前被调用。在这里，__new__ 方法接收一些参数，并更新传递给父类构造函
    # 数的关键字参数字典 kwargs。
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        # EfficientNetV2Backbone被调用以创建一个新的实例，这里使用预设 "efficientnetv2_b0" 
        # 并传入更新后的 kwargs。
        return EfficientNetV2Backbone.from_preset("efficientnetv2_b0", **kwargs)
    # presets 和 presets_with_weights 是类属性，它们定义了可用的预训练模型配置。这里的 presets 是一个字典，
    # 包含了预训练模型的名字及其配置信息。presets_with_weights 则简单地返回 presets 的内容。
    @classproperty
    def presets(cls):
        return {
            "efficientnetv2_b0_imagenet": copy.deepcopy(
                backbone_presets["efficientnetv2_b0_imagenet"]
            ),
        }
    @classproperty
    def presets_with_weights(cls):
        return cls.presets
# 提取必要的参数
input_shape = (224, 224, 3)
stackwise_kernel_sizes = [3, 3, 3, 3, 3, 3]
stackwise_num_repeats = [2, 4, 4, 6, 9, 15]
stackwise_input_filters = [24, 24, 48, 64, 128, 160]
stackwise_output_filters = [24, 48, 64, 128, 160, 256]
stackwise_expansion_ratios = [1, 4, 4, 4, 6, 6]
stackwise_squeeze_and_excite_ratios = [0.0, 0.0, 0, 0.25, 0.25, 0.25]
stackwise_strides = [1, 2, 2, 2, 1, 2]
stackwise_conv_types = ["fused", "fused", "fused", "unfused", "unfused", "unfused"]
width_coefficient = 1.0
depth_coefficient = 1.0
include_rescaling = False
# 构建 EfficientNetV2-B0 模型
# 提取块浅层用的普通卷积,深层用的扩张点卷积+深度卷积组合
model = EfficientNetV2Backbone(
    input_shape=input_shape,
    stackwise_kernel_sizes=stackwise_kernel_sizes,
    stackwise_num_repeats=stackwise_num_repeats,
    stackwise_input_filters=stackwise_input_filters,
    stackwise_output_filters=stackwise_output_filters,
    stackwise_expansion_ratios=stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios=stackwise_squeeze_and_excite_ratios,
    stackwise_strides=stackwise_strides,
    stackwise_conv_types=stackwise_conv_types,
    width_coefficient=width_coefficient,
    depth_coefficient=depth_coefficient,
    include_rescaling=include_rescaling,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_backbone import EfficientNetV2Backbone
model=EfficientNetV2Backbone.from_preset('efficientnetv2_b2')
model.get_config()
model=EfficientNetV2Backbone.from_preset('efficientnetv2_b2',input_shape=(224,224,3))
model.summary()
from keras.applications.efficientnet_v2 import EfficientNetV2S
aa=EfficientNetV2S(include_top=False,input_shape=(224,224,3))
aa.get_config()
# 渐进式学习（Progressive Learning）是一种训练策略，旨在通过逐步增加输入图像尺寸来优化模型的训练过程。这种方法有几个优点：

#     加速初始训练：
#         使用较小的图像尺寸可以更快地进行训练，因为处理较小的图像所需的计算资源较少。

#     逐步提升性能：
#         随着图像尺寸的增加，模型可以逐步学习更高分辨率的特征，从而提高最终的准确性。

#     防止过拟合：
#         较小的图像尺寸提供了较少的信息，这有助于防止模型在训练初期过拟合。
#         当图像尺寸增加时，通过调整正则化技术（如 dropout）来防止过拟合。

# 为什么使用渐进式学习？

# 渐进式学习的目标是让模型在不同尺寸的图像上表现良好。这种方法通过以下方式实现这一目标：

#     从小尺寸开始：
#         使用较小的图像尺寸可以快速启动训练过程，因为处理小尺寸图像所需的计算资源较少。

#     逐步增加图像尺寸：
#         在训练的每个阶段，逐步增加图像尺寸，以逐步提高模型的学习能力。这有助于模型在不同尺度的特征上都有良好的表现。

#     调整正则化技术：
#         当图像尺寸增加时，调整正则化技术（如 dropout）以防止过拟合。例如，随着图像尺寸的增加，逐步增加 dropout 的比例。

# 具体实现步骤

# 以下是使用渐进式学习训练模型的具体实现步骤：

#     定义初始图像尺寸：
#         选择一个较小的初始图像尺寸，如 128x128。

#     定义最终图像尺寸：
#         选择一个较大的最终图像尺寸，如 224x224。

#     定义每个阶段的 epoch 数量：
#         决定每个阶段训练多少 epoch，例如每 5 个 epoch 增加一次图像尺寸。

#     逐步增加图像尺寸：
#         在每个阶段结束后，逐步增加图像尺寸，直到达到最终图像尺寸。

#     调整正则化技术：
#         当图像尺寸增加时，逐步增加 dropout 的比例或其他正则化技术的强度。
# import tensorflow as tf
# from tensorflow.keras import layers, models, datasets, callbacks
# # 定义模型
# def build_efficientnet_v2(input_shape=(224, 224, 3), num_classes=1000):
#     # 构建模型...
#     model = models.Sequential([
#         layers.Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=input_shape),
#         layers.BatchNormalization(),
#         layers.Activation('swish'),
#         # 添加更多的层...
#     ])
#     return model

# # 加载数据集
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# # 数据预处理
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)

# # 定义训练参数
# initial_image_size = 128
# final_image_size = 224
# epochs_per_stage = 5
# current_image_size = initial_image_size

# # 构建模型
# model = build_efficientnet_v2()

# while current_image_size <= final_image_size:
#     # 使用当前图像尺寸训练
#     train_dataset = train_dataset.map(lambda img, label: (tf.image.resize(img, (current_image_size, current_image_size)), label))
    
#     # 编译模型
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     # 训练
#     model.fit(train_dataset, epochs=epochs_per_stage)
    
#     # 更新图像尺寸
#     current_image_size += 32
    
#     # 调整正则化强度
#     if current_image_size > initial_image_size:
#         model.layers[-1].rate = min(model.layers[-1].rate + 0.1, 0.5)
# import tensorflow as tf
# from tensorflow.keras import layers, models, datasets, callbacks
# from tensorflow.keras.applications import EfficientNetB0

# # 定义训练参数
# initial_image_size = 128
# final_image_size = 224
# epochs_per_stage = 5
# current_image_size = initial_image_size

# # 加载预训练模型
# base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# # 添加自定义分类层
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(1024, activation='relu')(x)
# predictions = layers.Dense(num_classes, activation='softmax')(x)

# # 创建完整模型
# model = models.Model(inputs=base_model.input, outputs=predictions)

# # 设置某些层不可训练（可选）
# for layer in base_model.layers:
#     layer.trainable = False

# # 加载数据集
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# # 数据预处理
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# # 加载训练数据
# train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
# validation_generator = validation_datagen.flow(test_images, test_labels, batch_size=32)

# while current_image_size <= final_image_size:
#     # 使用当前图像尺寸训练
#     train_generator = train_datagen.flow(train_images, train_labels, batch_size=32, target_size=(current_image_size, current_image_size))
#     validation_generator = validation_datagen.flow(test_images, test_labels, batch_size=32, target_size=(current_image_size, current_image_size))
    
#     # 编译模型
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     # 训练
#     model.fit(
#         train_generator,
#         steps_per_epoch=len(train_images) // 32,
#         epochs=epochs_per_stage,
#         validation_data=validation_generator,
#         validation_steps=len(test_images) // 32)
    
#     # 更新图像尺寸
#     current_image_size += 32
    
#     # 动态调整数据增强强度
#     if current_image_size > initial_image_size:
#         train_datagen.rotation_range = min(train_datagen.rotation_range + 10, 40)
#         train_datagen.width_shift_range = min(train_datagen.width_shift_range + 0.1, 0.5)
#         train_datagen.height_shift_range = min(train_datagen.height_shift_range + 0.1, 0.5)

# # 最终评估
# model.evaluate(validation_generator)
# # 最终评估
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).map(lambda img, label: (tf.image.resize(img, (final_image_size, final_image_size)), label))
# model.evaluate(test_dataset)
# {'name': 'efficient_net_v2_backbone',
#  'trainable': True,
#  'include_rescaling': True,
#  'width_coefficient': 1.1,
#  'depth_coefficient': 1.2,
#  'skip_connection_dropout': 0.2,
#  'depth_divisor': 8,
#  'min_depth': 8,
#  'activation': 'swish',
#  'input_shape': (None, None, 3),
#  'input_tensor': None,
#  'stackwise_kernel_sizes': [3, 3, 3, 3, 3, 3],
#  'stackwise_num_repeats': [1, 2, 2, 3, 5, 8],
#  'stackwise_input_filters': [32, 16, 32, 48, 96, 112],
#  'stackwise_output_filters': [16, 32, 48, 96, 112, 192],
#  'stackwise_expansion_ratios': [1, 4, 4, 4, 6, 6],
#  'stackwise_squeeze_and_excite_ratios': [0, 0, 0, 0.25, 0.25, 0.25],
#  'stackwise_strides': [1, 2, 2, 2, 1, 2],
#  'stackwise_conv_types': ['fused',
#   'fused',
#   'fused',
#   'unfused',
#   'unfused',
#   'unfused']}