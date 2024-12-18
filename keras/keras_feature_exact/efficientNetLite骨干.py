# 核初始化
def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )
# 确保通道数是depth_divisor的倍数
def round_filters(filters, depth_divisor, width_coefficient):
    filters *= width_coefficient
    new_filters = max(
        depth_divisor,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)
# 堆叠块的深度
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def apply_efficient_net_lite_block(
    inputs,
    activation="relu6",
    dropout_rate=0.0,
    name=None,
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
):
    # 设置默认的块前缀
    if name is None:
        name = f"block_{keras.backend.get_uid('block_')}_"

    # 扩张阶段
    filters = filters_in * expand_ratio # 内部通道数
    # 如果扩张系数!=1,用扩张点卷积切换通道数
    if expand_ratio != 1:
        x = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name=name + "expand_conv",
        )(inputs)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, name=name + "expand_bn"
        )(x)
        x = keras.layers.Activation(
            activation, name=name + "expand_activation"
        )(x)
    else: # 扩张系数==1,不改变,x是中间变量
        x = inputs

    # 设置正确的填充
    if strides == 2:
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, kernel_size),
            name=name + "dwconv_pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    # 深度卷积,用于提取空间特征
    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=conv_kernel_initializer(),
        name=name + "dwconv",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(axis=BN_AXIS, name=name + "bn")(x)
    x = keras.layers.Activation(activation, name=name + "activation")(x)

    # 输出阶段,点卷积切换通道
    x = keras.layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=conv_kernel_initializer(),
        name=name + "project_conv",
    )(x)
    # 在最后一个轴做批次标准化
    x = keras.layers.BatchNormalization(axis=BN_AXIS, name=name + "project_bn")(
        x
    )
    #残差连接,条件:步长==1,并且输入和输出通道数相同
    if strides == 1 and filters_in == filters_out:
        # 如果有dropout,noise_shape=(None, 1, 1, 1)
        # 对于特征图的空间一致性要求较高的任务：比如物体检测、分割等任务，通常需要特征图在空间上具有一致性，
        # 这时使用 noise_shape=(None, 1, 1, 1) 更合适。
        if dropout_rate > 0:
            x = keras.layers.Dropout(
                dropout_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = keras.layers.Add(name=name + "add")([x, inputs])
    # 如果用了残差,返回的就是残差后的结果,否则,返回的是bn后的结果
    return x
# keras_cv_export注解提供了从外部导入本类的路径
# keras.saving.register_keras_serializable用于注册一个类为可序列化的 Keras 对象。这意味着该类
# 可以被 Keras 自动保存和加载，从而支持模型的持久化操作。
@keras_cv_export("keras_cv.models.EfficientNetLiteBackbone")
@keras.saving.register_keras_serializable(package="keras_cv.models")
class EfficientNetLiteBackbone(Backbone):
    def __init__(
        self,
        *,
        include_rescaling, # 是否在内部归一化数据
        width_coefficient, # 宽度系数
        depth_coefficient, # 深度系数
        # 配置列表
        stackwise_kernel_sizes,  # 核大小
        stackwise_num_repeats, # 重复提取块的次数
        stackwise_input_filters, # 输入通道数
        stackwise_output_filters, # 输出通道数
        stackwise_expansion_ratios, # 扩张比例
        stackwise_strides, # 步长
        dropout_rate=0.2, # 在最终分类层之前的丢弃率。
        drop_connect_rate=0.2, # 在残差连接处的丢弃率。默认值设置为 0.2。
        depth_divisor=8, # 单位宽度
        input_shape=(None, None, 3), # 输入形状
        input_tensor=None, # 输入
        activation="relu6", # 激活函数
        **kwargs,
    ):
        # 模型输入
        img_input = utils.parse_model_inputs(input_shape, input_tensor)
        # 这里x会作为中间变量,做函数式的处理
        x = img_input
        if include_rescaling:
            x = keras.layers.Rescaling(1.0 / 255.0)(x) # 0--1
        # 填充,确保偶数尺寸时,也能正确填充
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, 3), name="stem_conv_pad"
        )(x)
        # 第一次下采样(112,112,3)
        x = keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="stem_conv",
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(axis=BN_AXIS, name="stem_bn")(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)

        # 处理块索引
        block_id = 0
        blocks = float(sum(stackwise_num_repeats)) # 处理块的总数
        # 对应金字塔层级的特征图列表
        pyramid_level_inputs = []
        # 遍历每一个层级的卷积
        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i] # 当前层级的重复次数
            input_filters = stackwise_input_filters[i] # 当前层级的输入通道数
            output_filters = stackwise_output_filters[i]
            # 规范化输入输出通道(8的倍数)
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                depth_divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                depth_divisor=depth_divisor,
            )
            # 如果是第一个和最后一个层级
            if i == 0 or i == (len(stackwise_kernel_sizes) - 1):
                repeats = num_repeats
            else: # 其他情况
                repeats = round_repeats(
                    repeats=num_repeats,
                    depth_coefficient=depth_coefficient,
                )
            strides = stackwise_strides[i] # 当前使用相同配置的卷积步长
            # 遍历当前层级中的每个提取块
            for j in range(repeats):
                # 如果不是第一个块的话,步长==1,输入和输出通道数相同
                if j > 0:
                    strides = 1
                    input_filters = output_filters
                # 第一次下采样时,会把之前的特征图加入FPN
                if strides != 1:
                    pyramid_level_inputs.append(utils.get_tensor_input_name(x))
            
                # a...,这里预设的是块前缀
                letter_identifier = chr(j + 97)
                # 应用特征提取块
                x = apply_efficient_net_lite_block(
                    inputs=x,
                    filters_in=input_filters,
                    filters_out=output_filters,
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    expand_ratio=stackwise_expansion_ratios[i],
                    activation=activation,
                    dropout_rate=drop_connect_rate * block_id / blocks,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                block_id += 1 # 块索引

        # 经过所有提取块处理之后,要切换到的输出通道数,点卷积切换通道
        x = keras.layers.Conv2D(
            1280,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="top_conv",
        )(x)
        # 批次激活块,激活函数用于给线性的卷积增加非线性能力
        x = keras.layers.BatchNormalization(axis=BN_AXIS, name="top_bn")(x)
        x = keras.layers.Activation(activation, name="top_activation")(x)
        # FPN
        pyramid_level_inputs.append(utils.get_tensor_input_name(x))
        # 这个调用父类Model的方法构建模型
        super().__init__(inputs=img_input, outputs=x, **kwargs)
        # 保存这些设置为实例属性
        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.input_tensor = input_tensor
        # 金字塔层级的提取模块,字典类型,idx-->name
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
        # 配置列表
        self.stackwise_kernel_sizes = stackwise_kernel_sizes
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_input_filters = stackwise_input_filters
        self.stackwise_output_filters = stackwise_output_filters
        self.stackwise_expansion_ratios = stackwise_expansion_ratios
        self.stackwise_strides = stackwise_strides
    # 这个用于设置配置字典,以便于之后的序列化
    def get_config(self):
        # 获取父类的配置字典
        config = super().get_config()
        config.update( # 加入子类特有的配置属性
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "dropout_rate": self.dropout_rate,
                "drop_connect_rate": self.drop_connect_rate,
                "depth_divisor": self.depth_divisor,
                "activation": self.activation,
                "input_tensor": self.input_tensor,
                "input_shape": self.input_shape[1:],
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_strides": self.stackwise_strides,
            }
        )
        return config
    # 类属性,预设配置
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
model = EfficientNetLiteBackbone(
        input_shape=(224,224,3),
        stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
        stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1], # 相同卷积配置的提取深度
        stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
        stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
        stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6], # 扩张系数
        stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
        width_coefficient=1.0,
        depth_coefficient=1.0,
        include_rescaling=False,
    )
model=keras_cv.models.EfficientNetLiteBackbone.from_preset('efficientnetlite_b0')
model.pyramid_level_inputs
@keras_cv_export("keras_cv.models.EfficientNetLiteB0Backbone")
class EfficientNetLiteB0Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # 把传入的参数打包到kwargs字典中
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b0", **kwargs
        )
    @classproperty
    def presets(cls):
        return {}
    @classproperty
    def presets_with_weights(cls):
        return {}