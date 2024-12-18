BN_AXIS = 3
BN_EPSILON = 1.001e-5
# 基本残差块
def apply_basic_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    conv_shortcut=False,
    name=None,
):
    # 设置默认块前缀
    if name is None:
        name = f"v2_basic_block_{keras.backend.get_uid('v2_basic_block')}"
    # 用批次激活在前的的情况,预激活批次激活块
    use_preactivation = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)

    use_preactivation = keras.layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)
    # 首先，根据 dilation 参数来确定最终使用的步长 s
    # 如果 dilation（膨胀率）为 1，则步长 s 设置为 stride 的值；
    # 否则，如果 dilation 不为 1，则步长 s 被固定为 1。
    s = stride if dilation == 1 else 1
    # 如果 conv_shortcut 为 True，则使用 1x1 卷积来创建残差前段，这通常用于改变输入
    # 特征图的维度（即通道数），以便与主路径的输出维度匹配。
    # 这里 strides=s 表示捷径连接的步长取决于前面设置的 s 值。
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            filters, 1, strides=s, name=name + "_0_conv"
        )(use_preactivation)
    # 如果 conv_shortcut 为 False，则需要根据步长 s 是否大于 1 来决定shortcut：
    # 当 s > 1 时，意味着需要降低特征图的尺寸，这里使用 1x1 的最大池化层（MaxPooling2D）来实现这一点。
    # 当 s == 1 时，不需要改变特征图的尺寸，因此shortcut直接使用输入 x
    # 当需要改变维度或步长时，使用 1x1 卷积；
    # 当只需要改变步长而不改变维度时，使用最大池化；
    # 当既不需要改变维度也不需要改变步长时，捷径连接直接使用输入 x。
    # 1x1的pool_size不会取窗口内的最大值,因为步长是2,只会下采样
    else:
        shortcut = (
            keras.layers.MaxPooling2D(
                1, strides=s, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )
    # 标准卷积处理
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=1,
        use_bias=False,
        name=name + "_1_conv",
    )(use_preactivation)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)
    # 标准卷积处理
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    # 残差
    x = keras.layers.Add(name=name + "_out")([shortcut, x])
    return x
x=tf.constant([[[[1], [2],  [3],  [4]],
[[5],  [6], [7],  [8]],
[[9], [10], [11], [12]],
[[13], [14], [15], [16]]]])
x=keras.layers.MaxPooling2D(
                1, strides=2
            )(x)

def apply_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    conv_shortcut=False,
    name=None,
):
    # 设置默认块前缀
    if name is None:
        name = f"v2_block_{keras.backend.get_uid('v2_block')}"
    # 使用批次激活块在前的预激活
    use_preactivation = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)
    use_preactivation = keras.layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)
    # 首先，根据 dilation 参数来确定最终使用的步长 s
    # 如果 dilation（膨胀率）为 1，则步长 s 设置为 stride 的值；
    # 否则，如果 dilation 不为 1，则步长 s 被固定为 1。
    s = stride if dilation == 1 else 1
    # 当需要改变通道数或步长时，使用 1x1 卷积；
    # 当只需要改变步长而不改变维度时，使用最大池化；
    # 当既不需要改变维度也不需要改变步长时，捷径连接直接使用输入 x。
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            4 * filters,
            1,
            strides=s,
            name=name + "_0_conv",
        )(use_preactivation)
    else:
        shortcut = (
            keras.layers.MaxPooling2D(
                1, strides=stride, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )
    # 点卷积切换到狭窄通道
    x = keras.layers.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + "_1_conv"
    )(use_preactivation)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)
    # 普通卷积处理,根据s的值决定是否下采样
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        use_bias=False,
        padding="same",
        dilation_rate=dilation,
        name=name + "_2_conv",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_2_relu")(x)
    # 点卷积切换通道到宽通道
    x = keras.layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = keras.layers.Add(name=name + "_out")([shortcut, x]) # 残差
    return x
# 应用提取块
def apply_stack(
    x,
    filters,
    blocks,
    stride=2,
    dilations=1,
    name=None,
    block_type="block",
    first_shortcut=True,
):
    # 预设块前缀
    if name is None:
        name = "v2_stack"
    # 根据block_type选择使用的提取块
    if block_type == "basic_block":
        block_fn = apply_basic_block # 基本的卷积块
    elif block_type == "block":
        block_fn = apply_block # 瓶颈块
    else:
        raise ValueError(
            """`block_type` must be either "basic_block" or "block". """
            f"Received block_type={block_type}."
        )
    # 第一次: (None, 56, 56, 256) 1 1
    # 第2-n-1次: (None, 56, 56, 256) 1 1
    # 第n次: (None, 56, 56, 256) 1 1
    # 第一次: (None, 56, 56, 512) 1 2
    # 第2-n-1次: (None, 56, 56, 512) 1 2
    # 第n次: (None, 28, 28, 512) 1 2
    # 第一次: (None, 28, 28, 1024) 1 2
    # 第2-n-1次: (None, 28, 28, 1024) 1 2
    # 第n次: (None, 14, 14, 1024) 1 2
    # 第一次: (None, 14, 14, 2048) 1 2
    # 第2-n-1次: (None, 14, 14, 2048) 1 2
    # 第n次: (None, 7, 7, 2048) 1 2
    # 第一次特征处理,没传入步长,步长是1,只是特征提取,没下采样,所以在块内只会设置残差前段切换通道
    x = block_fn(
        x, filters, conv_shortcut=first_shortcut, name=name + "_block1"
    )
    # print('第一次:',x.shape,dilations,stride)
    # 同配置的特征提取中的2--n-1深度的处理,这时步长是1,特征提取
    for i in range(2, blocks):
        x = block_fn(
            x, filters, dilation=dilations, name=name + "_block" + str(i)
        )
    # print('第2-n-1次:',x.shape,dilations,stride)
    # 最后一个块处理,这时传入了步长,所以最后才进行了下采样
    x = block_fn(
        x,
        filters,
        stride=stride,
        dilation=dilations,
        name=name + "_block" + str(blocks),
    )
    # print('第n次:',x.shape,dilations,stride)
    return x
@keras_cv_export("keras_cv.models.ResNetV2Backbone")
class ResNetV2Backbone(Backbone): # resnetv2
    def __init__(
        self,
        *,
        stackwise_filters, # 通道数
        stackwise_blocks, # 块深度
        stackwise_strides, # 步长
        include_rescaling, # 是否包含归一化
        stackwise_dilations=None, # 膨胀率
        input_shape=(None, None, 3), 
        input_tensor=None,
        block_type="block", # 块类型
        **kwargs,
    ):
        # 输入
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs # 中间变量x,(224,224,3)
        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x) # 归一化
        # 第一个标准卷积下采样,核大小7,(112,112,3)
        x = keras.layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=True,
            padding="same",
            name="conv1_conv",
        )(x)
        # 最大池化,(56,56,3)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)
        
        num_stacks = len(stackwise_filters) # 层级
        # 膨胀率
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks
        # 金字塔层级的特征提取
        pyramid_level_inputs = {}
        # 遍历每个层级
        for stack_index in range(num_stacks):
            # 相同配置的模块处理
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index], # 当前层级的通道数
                blocks=stackwise_blocks[stack_index], # 深度
                stride=stackwise_strides[stack_index], # 步长
                dilations=stackwise_dilations[stack_index], # 膨胀率
                block_type=block_type, # 块类型
                # 如果block_type == "block",first_shortcut=True
                # 如果block_type不是"block",stack_index > 0,就是不能是第一次
                first_shortcut=(block_type == "block" or stack_index > 0),
                name=f"v2_stack_{stack_index}",
            )
            # 金字塔特征图
            pyramid_level_inputs[f"P{stack_index + 2}"] = (
                utils.get_tensor_input_name(x)
            )
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="post_bn"
        )(x)
        x = keras.layers.Activation("relu", name="post_relu")(x)
        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # 设置实例属性
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.stackwise_dilations = stackwise_dilations
        self.input_tensor = input_tensor
        self.block_type = block_type
    # 获取配置的方法,用于序列化
    def get_config(self):
        config = super().get_config() # 获取父类配置字典
        config.update( # 把子类的配置加入配置字典
            {
                "stackwise_filters": self.stackwise_filters,
                "stackwise_blocks": self.stackwise_blocks,
                "stackwise_strides": self.stackwise_strides,
                "include_rescaling": self.include_rescaling, # 是否归一化
                # (h,w,c)
                "input_shape": self.input_shape[1:],
                "stackwise_dilations": self.stackwise_dilations, # 膨胀率
                "input_tensor": self.input_tensor, 
                "block_type": self.block_type,
            }
        )
        return config
    # 类属性,骨干预设
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    # 类属性,骨干预设(包含预训练权重)
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
# 骨干预设没有预训练权重
backbone_presets_no_weights = {
    "resnet18_v2": { # resnet18_v2
        "metadata": { # 元数据
            "description": ( # 描述
                "ResNet model with 18 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 11183488, # 参数
            "official_name": "ResNetV2", # 官方名称
            "path": "resnet_v2", # path
        },
        # kaggle上的模型配置路径
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet18_v2/2",
    },
    "resnet34_v2": {
        "metadata": {
            "description": (
                "ResNet model with 34 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 21299072,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet34_v2/2",
    },
    "resnet50_v2": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 23564800,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet50_v2/2",
    },
    "resnet101_v2": {
        "metadata": {
            "description": (
                "ResNet model with 101 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 42626560,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet101_v2/2",
    },
    "resnet152_v2": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 58331648,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet152_v2/2",
    },
}
# 带预训练权重的预设
backbone_presets_with_weights = {
    "resnet50_v2_imagenet": { # 键值对
        "metadata": { # 元数据
            "description": ( # 描述
                "ResNet model with 50 layers where the batch normalization and "
                "ReLU activation precede the convolution layers (v2 style). "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 23564800, # 参数
            "official_name": "ResNetV2", 
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet50_v2_imagenet/2",
    },
}
# 导入路径
@keras_cv_export("keras_cv.models.ResNet18V2Backbone")
class ResNet18V2Backbone(ResNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # kwargs是字典结构,把传入参数加入kwargs字典中
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        # 根据预设的键和kwargs中传入的配置构建resnet18_v2模型
        return ResNetV2Backbone.from_preset("resnet18_v2", **kwargs)
    # 类属性
    @classproperty
    def presets(cls):
        return {}
    @classproperty
    def presets_with_weights(cls):
        return {}

rsnet=ResNet18V2Backbone(input_shape=(224,224,3))
rsnet.summary()
[rsnet.get_layer('conv1_conv').output]+[rsnet.get_layer(i).output for i in rsnet.pyramid_level_inputs.values()]
