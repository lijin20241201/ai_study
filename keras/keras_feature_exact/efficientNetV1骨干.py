def round_filters(filters, width_coefficient, divisor):
    filters *= width_coefficient # 滤镜数
    # 能整除divisor的滤镜数
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # 确保四舍五入的降幅不超过10%。
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)
def correct_pad_downsample(inputs, kernel_size):
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)] # (h,w)
    # 如果核大小是整数类型,转换成元组
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    # 如果输入的尺寸中的任何一个维度为 None（这意味着它是一个动态尺寸）,那么调整值 adjust 被设置为 (1, 1)
    if input_size[0] is None:
        adjust = (1, 1)
    # 否则，adjust 计算为每个维度的奇偶校验，用来决定是否需要额外的填充来确保尺寸正确。
    # 如果h,w是偶数,adjust=(1,1),如果h,w是奇数,adjust=(0,0)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    # 然后计算正确的填充量 correct，这通常是卷积核尺寸的一半。
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    # 最后，返回一个元组，表示在每个维度的两侧应分别添加多少填充。这里减去 adjust 值是为了确保即使输入尺寸
    # 不是奇数，也能得到正确的输出尺寸。
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
# scale 参数用于控制权重的方差缩放因子，这里使用了传入的 scale 参数值
# mode 参数指定缩放的方式，这里使用 "fan_out" 表示输出边数（即权重矩阵的输出通道数）。
# distribution 参数指定了随机分布的方式，这里使用 "truncated_normal" 表示截断的正态分布。
def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )
# 向上取整
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))
def get_tensor_input_name(tensor):
    if keras_3():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name
# 单独的一个efficientnet特征提取块
def apply_efficientnet_block(
    inputs,
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    activation="swish",
    expand_ratio=1,
    se_ratio=0.0,
    dropout_rate=0.0,
    name="",
):
   
    filters = filters_in * expand_ratio # 扩张通道数
    # 如果扩张系数不是1
    if expand_ratio != 1:
        # 用扩张点卷积切换通道
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name=name + "expand_conv",
        )(inputs)
        # 批次标准化块
        x = keras.layers.BatchNormalization(
            axis=3,
            name=name + "expand_bn",
        )(x)
        x = keras.layers.Activation(
            activation, name=name + "expand_activation"
        )(x)
    # 如果扩张系数==1,不改变
    else:
        x = inputs
    # 深度卷积,如果步长是2,要用自定义的填充,否则用conv_pad = "same"
    if strides == 2:
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, kernel_size),
            name=name + "dwconv_pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    # 深度卷积
    x = keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=conv_kernel_initializer(),
        name=name + "dwconv",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=3,
        name=name + "dwconv_bn",
    )(x)
    x = keras.layers.Activation(activation, name=name + "dwconv_activation")(x)
    # 挤压激励阶段,条件是0 < se_ratio <= 1
    if 0 < se_ratio <= 1:
        # 挤压通道数
        filters_se = max(1, int(filters_in * se_ratio))
        # 先全局平均池化
        se = keras.layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se_shape = (1, 1, filters) 
        # 变形
        se = keras.layers.Reshape(se_shape, name=name + "se_reshape")(se)
        # 挤压点卷积切换通道,挤压步骤实际上起到了特征浓缩的作用。通过减少通道数，模型可以更
        # 高效地学习到通道间的相互依赖关系。
        # 用的swish,swish 激活函数的设计目的是为了克服 ReLU 及其变种的一些缺点，如“死区”问题（当输入为负数时，ReLU 的输出为0），
        # 同时它保持了非线性且平滑的特性，这有助于优化过程中的梯度传播。swish(x)=x⋅σ(x),是自我门控
        se = keras.layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=conv_kernel_initializer(),
            name=name + "se_reduce",
        )(se)
        # 还原到扩张通道数的点卷积,激活函数用sigmoid,给通道打分
        se = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=conv_kernel_initializer(),
            name=name + "se_expand",
        )(se)
        # 对se模块之前的输入特征图进行通道加权
        x = keras.layers.multiply([x, se], name=name + "se_excite")
    
    # 输出阶段,压缩点卷积切换通道到filters_out
    x = keras.layers.Conv2D(
        filters=filters_out,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=conv_kernel_initializer(),
        name=name + "project",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=3,
        name=name + "project_bn",
    )(x)
    x = keras.layers.Activation(activation, name=name + "project_activation")(x)
    # 如果步长等于1,并且filters_in == filters_out,残差连接
    if strides == 1 and filters_in == filters_out:
        if dropout_rate > 0: # dropout
            x = keras.layers.Dropout(
                dropout_rate,
                noise_shape=(None, 1, 1, 1),
                name=name + "drop",
            )(x)
        x = keras.layers.Add(name=name + "add")([x, inputs])
    # 这意味着步长是2时,不会进行残差连接
    return x
# 用于注册自定义类（如模型或层），使其能够在 Keras 中被正确地序列化和反序列化。这个装饰器接收一个 package 参数，
# 用于指定注册的类别所属的包。
@keras_cv_export("keras_cv.models.EfficientNetV1Backbone")
@keras.saving.register_keras_serializable(package="keras_cv.models")
class EfficientNetV1Backbone(Backbone):
    def __init__(
        self,
        *,
        include_rescaling,# 是否对输入进行缩放
        width_coefficient, # 网络宽度的缩放系数。
        depth_coefficient, # 网络深度的缩放系数。
        stackwise_kernel_sizes,  # 列表形式的整数，每个卷积块使用的内核大小。
        stackwise_num_repeats, # 列表形式的整数，每个卷积块重复的次数。
        stackwise_input_filters, # 列表形式的整数，每个卷积块的输入滤波器数量
        stackwise_output_filters,# 列表形式的整数，每个卷积堆栈中每个卷积块的输出滤波器数量。
        stackwise_expansion_ratios,# 列表形式的浮点数，传递给挤压和激励块的扩展比率。
        stackwise_strides, # 列表形式的整数，每个卷积块的步长。
        stackwise_squeeze_and_excite_ratios, # 列表形式的浮点数，传递给挤压和激励块的挤压和激励比率。
        dropout_rate=0.2, # 在最终分类层之前的丢弃率。
        drop_connect_rate=0.2, # 在跳过连接处的丢弃率。默认值设置为 0.2。
        depth_divisor=8, # 网络宽度的一个单位。默认值设置为 8。
        input_shape=(None, None, 3), # 可选的形状元组，应该正好有 3 个输入通道。
        input_tensor=None,# 可选的 Keras 张量的输出作为模型的图像输入。
        activation="swish", # 在每两个卷积层之间使用的激活函数。
        **kwargs,
    ):  
        img_input = utils.parse_model_inputs(input_shape, input_tensor) # (b,h,w,c)
        x = img_input
        # 如果include_rescaling为True,就归一化
        if include_rescaling:
            x = keras.layers.Rescaling(1.0 / 255.0)(x) 
        # 正确的填充,确保输入尺寸是偶数时,也能正确填充
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, 3), name="stem_conv_pad"
        )(x)
        # 确保输入的通道数是depth_divisor的整数倍
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
            width_coefficient=width_coefficient,
            divisor=depth_divisor,
        )
        # 卷积下采样(112,112,3)
        x = keras.layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="stem_conv",
        )(x)
        # 批次标准化和激活函数处理块
        x = keras.layers.BatchNormalization(
            axis=3,
            name="stem_bn",
        )(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)

        # 统计所有层级中特征提取块的数目
        block_id = 0
        blocks = float(sum(stackwise_num_repeats)) # 提取块的总数
        # 金字塔层级的特征提取块
        pyramid_level_inputs = []
        # 遍历每个层级
        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i] # 块深度
            input_filters = stackwise_input_filters[i] # 块输入通道数
            output_filters = stackwise_output_filters[i] # 块输出通道数

            # 规范化输入输出通道数
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                divisor=depth_divisor,
            )
            # 重复次数
            repeats = round_repeats(
                repeats=num_repeats,
                depth_coefficient=depth_coefficient,
            )
            strides = stackwise_strides[i] # 步长
            squeeze_and_excite_ratio = stackwise_squeeze_and_excite_ratios[i]
            # 特征图被提取特征的重复次数,一个层级内的多个提取块
            for j in range(repeats):
                # j大于0,是指第二个开始
                if j > 0:
                    strides = 1
                    input_filters = output_filters
                # 只有第一次时,步长不等于1,那时是下采样,金字塔层级输入里添加之前的特征图名称
                if strides != 1:
                    pyramid_level_inputs.append(utils.get_tensor_input_name(x))

                # 97是小写字母的开头,字母标志符
                letter_identifier = chr(j + 97)
                x = apply_efficientnet_block(
                    inputs=x,
                    filters_in=input_filters,
                    filters_out=output_filters,
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    expand_ratio=stackwise_expansion_ratios[i],
                    se_ratio=squeeze_and_excite_ratio,
                    activation=activation,
                    dropout_rate=drop_connect_rate * block_id / blocks,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                block_id += 1

        # 规范化特征提取顶部输出通道数
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient, # 宽度系数
            divisor=depth_divisor,
        )
        # 点卷积混合切换通道数
        x = keras.layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            padding="same",
            strides=1,
            kernel_initializer=conv_kernel_initializer(),
            use_bias=False,
            name="top_conv",
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=3,
            name="top_bn",
        )(x)
        x = keras.layers.Activation(
            activation=activation, name="top_activation"
        )(x)
        # 加入FPN
        pyramid_level_inputs.append(utils.get_tensor_input_name(x))
        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)
        # 设置实例属性
        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.input_tensor = input_tensor
        # 构建金字塔层级的特征提取字典,idx-->name
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
        self.stackwise_kernel_sizes = stackwise_kernel_sizes
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_input_filters = stackwise_input_filters
        self.stackwise_output_filters = stackwise_output_filters
        self.stackwise_expansion_ratios = stackwise_expansion_ratios
        self.stackwise_strides = stackwise_strides
        self.stackwise_squeeze_and_excite_ratios = (
            stackwise_squeeze_and_excite_ratios
        )
    # 这个方法的主要目的是为了让模型可以被序列化，也就是说，可以将模型的配置信息保存下来，以便以后可以重新构建相同的模型。
    def get_config(self):
        # get_config() 方法是一个用于获取模型配置信息的方法。它通常返回一个字典，这个字典包含了模型的各种配置参数。
        config = super().get_config() 
        config.update( # 这部分代码更新了基础配置字典，加入了当前类实例特有的配置项。
            {
                "include_rescaling": self.include_rescaling,# 是否包含输入的归一化处理。
                "width_coefficient": self.width_coefficient, # 用于控制模型宽度和深度的系数。
                "depth_coefficient": self.depth_coefficient,
                "dropout_rate": self.dropout_rate, # 用于控制模型的正则化程度。
                "drop_connect_rate": self.drop_connect_rate,
                "depth_divisor": self.depth_divisor, # 用于调整通道数的除数。
                "activation": self.activation, # 激活函数。
                "input_tensor": self.input_tensor, # 输入张量和形状。
                "input_shape": self.input_shape[1:],
                "trainable": self.trainable,# 权重是否冻结
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes, # 各个层级的配置信息。
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_strides": self.stackwise_strides,
                "stackwise_squeeze_and_excite_ratios": (
                    self.stackwise_squeeze_and_excite_ratios
                ),
            }
        )
        return config
    # presets 是一个类属性，用于存储预设的配置信息。这些预设通常包含了一些预先定义好的模型配置，方便用户
    # 快速选择和使用
    @classproperty 
    def presets(cls):
        return copy.deepcopy(backbone_presets)

# 创建一个 EfficientNetV1Backbone 实例
model = EfficientNetV1Backbone(
    include_rescaling=True,
    width_coefficient=1.0,
    depth_coefficient=1.0,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation="swish",
    input_tensor=None,
    input_shape=(224, 224, 3),
    trainable=True,
    stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
    stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192, 320],
    stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320, 1280],
    stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
    stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
    stackwise_squeeze_and_excite_ratios=[0.25] * 7,
)
# 查看预设配置
# print(EfficientNetV1Backbone.presets)
# metadata：
#     description：描述了该模型的架构特点，指出这是一个 EfficientNet B-style 架构，具有 7 个卷积块，
#     并且 width_coefficient 和 depth_coefficient 均为 1.0。
#     params：模型的参数数量，这里是 4,050,716 个参数。
#     official_name：官方命名，这里是 EfficientNetV1。
#     path：模型文件所在的路径，这里是 efficientnetv1。
# kaggle_handle：
#     指定了模型文件在 Kaggle 上的存储位置，这里是 gs://keras-cv-kaggle/efficientnetv1_b0。

backbone_presets_no_weights = {
    "efficientnetv1_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 4050716,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b0",
    },
    "efficientnetv1_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 6576704,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b1",
    },
    "efficientnetv1_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 7770034,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b2",
    },
    "efficientnetv1_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 10785960,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b3",
    },
    "efficientnetv1_b4": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.4` and `depth_coefficient=1.8`."
            ),
            "params": 17676984,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b4",
    },
    "efficientnetv1_b5": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.6` and `depth_coefficient=2.2`."
            ),
            "params": 28517360,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b5",
    },
    "efficientnetv1_b6": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.8` and `depth_coefficient=2.6`."
            ),
            "params": 40965800,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b6",
    },
    "efficientnetv1_b7": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=2.0` and `depth_coefficient=3.1`."
            ),
            "params": 64105488,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b7",
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}