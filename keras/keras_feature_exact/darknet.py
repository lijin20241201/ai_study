def DarknetConvBlock(
    filters, kernel_size, strides, use_bias=False, activation="silu", name=None
):
    #块前缀
    if name is None:
        name = f"conv_block{keras.backend.get_uid('conv_block')}"
    # 栈列表,标准卷积或点卷积,批次标准化
    model_layers = [
        keras.layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding="same",
            use_bias=use_bias,
            name=name + "_conv",
        ),
        keras.layers.BatchNormalization(name=name + "_bn"),
    ]
    # 激活函数
    if activation == "silu":
        model_layers.append(
            keras.layers.Lambda(lambda x: keras.activations.silu(x))
        )
    elif activation == "relu":
        model_layers.append(keras.layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(keras.layers.LeakyReLU(0.1))
    # 返回序列化栈
    return keras.Sequential(model_layers, name=name)
def DarknetConvBlockDepthwise(
    filters, kernel_size, strides, activation="silu", name=None
):
    # 设置块默认前缀
    if name is None:
        name = f"conv_block{keras.backend.get_uid('conv_block')}"
    # 深度卷积,批次标准化,栈列表
    model_layers = [
        keras.layers.DepthwiseConv2D(
            kernel_size, strides, padding="same", use_bias=False
        ),
        keras.layers.BatchNormalization(),
    ]
    # 激活函数
    if activation == "silu":
        model_layers.append(
            keras.layers.Lambda(lambda x: keras.activations.swish(x))
        )
    elif activation == "relu":
        model_layers.append(keras.layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(keras.layers.LeakyReLU(0.1))
    # 点卷积块
    model_layers.append(
        DarknetConvBlock(
            filters, kernel_size=1, strides=1, activation=activation
        )
    )
    return keras.Sequential(model_layers, name=name)
# 将输入张量 x 沿着高度和宽度维度分成四个部分：
# 左上象限：表示取偶数行和偶数列。
# 右上象限：表示取奇数行和偶数列
# 左下象限：表示取偶数行和奇数列。
# 右下象限：表示取奇数行和奇数列。
# 使用 keras.layers.Concatenate 层沿着通道轴将这四个部分拼接起来。结果是一个新的
# 张量，其高度和宽度分别是原始张量的一半，但通道数是原来的四倍。
def Focus(name=None):
    def apply(x): # 闭包
        return keras.layers.Concatenate(name=name)(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
        )
    return apply
# 空间金字塔池化瓶颈
def SpatialPyramidPoolingBottleneck(
    filters, # c
    hidden_filters=None, # c/2
    kernel_sizes=(5, 9, 13),
    activation="silu",
    name=None,
):
    # 设置块前缀
    if name is None:
        name = f"spp{keras.backend.get_uid('spp')}"
    # 设置默认的hidden_filters
    if hidden_filters is None:
        hidden_filters = filters
    
    def apply(x):
        # 点卷积切换通道到c/2
        x = DarknetConvBlock(
            hidden_filters, # c/2
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv1",
        )(x)
        x = [x] # 设置列表
        # 遍历不同的核大小,把不同池化大小的处理结果加入列表
        for kernel_size in kernel_sizes:
            x.append(
                keras.layers.MaxPooling2D(
                    kernel_size,
                    strides=1,
                    padding="same",
                    name=f"{name}_maxpool_{kernel_size}",
                )(x[0])
            )
        # 在特征轴合并特征c/2-->2c
        x = keras.layers.Concatenate(name=f"{name}_concat")(x)
        # 点卷积切换通道到c
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv2",
        )(x)
        return x
    return apply
# 注解表示可序列化
@keras.saving.register_keras_serializable(package="keras_cv")
class CrossStagePartial(keras.layers.Layer): # csp
    def __init__(
        self,
        filters,
        num_bottlenecks,
        residual=True,
        use_depthwise=False,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters # 通道,c
        self.num_bottlenecks = num_bottlenecks # 瓶颈数
        self.residual = residual # 残差前段
        self.use_depthwise = use_depthwise # 是否用深度卷积
        self.activation = activation # 激活函数
        hidden_channels = filters // 2 # c/2
        # 设置提取块
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )
        # 点卷积(切换通道到c/2)
        self.darknet_conv1 = DarknetConvBlock(
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        # 点卷积(切换通道到c/2)
        self.darknet_conv2 = DarknetConvBlock(
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
        )
        # 重复瓶颈,num_瓶颈次数
        self.bottleneck_convs = []
        # 一定深度的瓶颈块
        for _ in range(num_bottlenecks):
            # 点卷积
            self.bottleneck_convs.append(
                DarknetConvBlock(
                    hidden_channels, # c/2
                    kernel_size=1,
                    strides=1,
                    activation=activation,
                )
            )
            # 提取块, 标准卷积
            self.bottleneck_convs.append(
                ConvBlock(
                    hidden_channels, # c/2
                    kernel_size=3,
                    strides=1,
                    activation=activation,
                )
            )
        # 如果设置了残差
        if self.residual:
            self.add = keras.layers.Add()
        self.concatenate = keras.layers.Concatenate()
        # 点卷积ConvBlock(切换通道到c)
        self.darknet_conv3 = DarknetConvBlock(
            filters, kernel_size=1, strides=1, activation=activation
        )
    # 前向传播
    def call(self, x):
        x1 = self.darknet_conv1(x) # c-->c/2
        x2 = self.darknet_conv2(x) # c-->c/2
        # 循环的处理x1,每次的residual都是上次残差后的x1
        for i in range(self.num_bottlenecks):
            residual = x1 # 残差连接前段
            x1 = self.bottleneck_convs[2 * i](x1) # 先切换通道到c/2
            x1 = self.bottleneck_convs[2 * i + 1](x1) # 之后深度卷积
            # 残差连接
            if self.residual:
                x1 = self.add([residual, x1]) # 残差连接 
        # 将另一个没有经过瓶颈处理的和经过多个瓶颈处理的特征图合并特征,通道变成c
        x1 = self.concatenate([x1, x2])
        # 经过最后的混合通道特征的点卷积块
        x = self.darknet_conv3(x1)
        return x
    # 获取配置字典的方法
    def get_config(self):
        config = { # 子类的特有的配置
            "filters": self.filters,
            "num_bottlenecks": self.num_bottlenecks,
            "residual": self.residual,
            "use_depthwise": self.use_depthwise,
            "activation": self.activation,
        }
        base_config = super().get_config() # 父类的配置字典
        return dict(list(base_config.items()) + list(config.items())) # 返回合并后的字典
# CSPDarkNet骨干
# 导入当前类的路径
@keras_cv_export("keras_cv.models.CSPDarkNetBackbone")
class CSPDarkNetBackbone(Backbone):
    def __init__(
        self,
        *,
        stackwise_channels, # 栈通道列表
        stackwise_depth, # 栈深度列表
        include_rescaling, # 是否归一化
        use_depthwise=False, # 如果为True，则使用深度卷积代替标准卷积
        input_shape=(None, None, 3),# 输入数据的形状。
        input_tensor=None,# 可选的输入张量
        **kwargs,
    ):
        # 根据use_depthwise判断用那种提取块
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )
        base_channels = stackwise_channels[0] // 2
        # 模型输入
        inputs = utils.parse_model_inputs(input_shape, input_tensor) # (224,224,3)
        # print(inputs.shape)
        x = inputs # 中间变量
        if include_rescaling: 
            x = keras.layers.Rescaling(1 / 255.0)(x) # 归一化
        # 焦点处理,空间尺寸会减半,通道会是原来的4倍
        # Focus层用于重新组织输入张量，以便模型可以在更早的阶段捕获更多细节信息。
        x = Focus(name="stem_focus")(x) # (112,112,12)
        # print(x.shape)
        # 标准卷积块(112,112,c/2)
        x = DarknetConvBlock(
            base_channels, kernel_size=3, strides=1, name="stem_conv"
        )(x)
        # print(x.shape)
        # 层级
        pyramid_level_inputs = {}
        # 遍历配置中的每一个相同配置的栈
        for index, (channels, depth) in enumerate(
            zip(stackwise_channels, stackwise_depth)
        ):
            # 提取块,核大小3,步长2,下采样
            # 使用选定的 ConvBlock 进行一次步长为2的卷积操作，实现下采样。
            x = ConvBlock(
                channels,
                kernel_size=3,
                strides=2,
                name=f"dark{index + 2}_conv",
            )(x)
            # 如果是最后一个阶段，则添加一个SPP,以获取不同尺度的特征。
            # SPP 层允许模型从不同尺度的特征图中提取信息，这对于物体检测尤其有用，因为它可以帮助模型
            # 处理不同大小的目标。
            if index == len(stackwise_depth) - 1:
                x = SpatialPyramidPoolingBottleneck(
                    channels,
                    hidden_filters=channels // 2,
                    name=f"dark{index + 2}_spp",
                )(x)
            # 添加一个 CrossStagePartial 层（CSP），这是一系列带有残差连接的瓶颈块，用于增加网络的深度
            # 而不显著增加计算成本。
            # CSP 结构是一种创新的设计，旨在通过引入额外的路径来改善信息流动，同时减少参数数量。它通常包括
            # 一系列带有残差连接的卷积层。
            x = CrossStagePartial(
                channels,
                num_bottlenecks=depth,
                use_depthwise=use_depthwise,
                residual=(index != len(stackwise_depth) - 1),
                name=f"dark{index + 2}_csp",
            )(x)
            pyramid_level_inputs[f"P{index + 2}"] = utils.get_tensor_input_name(
                x
            )
            # print(x.shape)
        # 调用父类的构造模型方法
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        # 设置实例属性
        self.pyramid_level_inputs = pyramid_level_inputs 
        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.include_rescaling = include_rescaling
        self.use_depthwise = use_depthwise
        self.input_tensor = input_tensor
    # 获取配置的方法
    def get_config(self):
        config = super().get_config() # 获取父类的配置字典
        config.update( # 更新子类独有的设置到配置字典
            {
                "stackwise_channels": self.stackwise_channels, # 通道列表
                "stackwise_depth": self.stackwise_depth, #块深度
                "include_rescaling": self.include_rescaling, # 是否归一化
                "use_depthwise": self.use_depthwise, # 是否用深度卷积块
                "input_shape": self.input_shape[1:], # (h,w,c)
                "input_tensor": self.input_tensor, 
            }
        )
        return config
    # 类属性,预设骨干配置
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    # 类属性,带预训练权重的预设配置
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
