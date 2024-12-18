BN_AXIS = 3
BN_EPSILON = 1.001e-5
# 卷积块
def apply_conv_block(x, growth_rate, name=None):
    # 设置默认块名前缀
    if name is None:
        name = f"conv_block_{keras.backend.get_uid('conv_block')}"
    shortcut = x # 残差前段
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_0_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_0_relu")(x)
    # 点卷积 c-->4c
    x = keras.layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv"
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_1_relu")(x)
    # 标准卷积 4c-->c
    x = keras.layers.Conv2D(
        growth_rate,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    # 在最后一个轴合并特征 c-->2c
    x = keras.layers.Concatenate(axis=BN_AXIS, name=f"{name}_concat")(
        [shortcut, x]
    )
    return x
# 堆叠卷积块
def apply_dense_block(x, num_repeats, growth_rate, name=None):
    # 设置默认块前缀
    if name is None:
        name = f"dense_block_{keras.backend.get_uid('dense_block')}"
    for i in range(num_repeats):
        x = apply_conv_block(x, growth_rate, name=f"{name}_block_{i}")
        # print(x.shape)
    return x
# 应用过度块,compression_ratio:压缩比
def apply_transition_block(x, compression_ratio, name=None):
    # 块前缀名
    if name is None:
        name = f"transition_block_{keras.backend.get_uid('transition_block')}"
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu")(x)
    # 点卷积,2c-->c
    x = keras.layers.Conv2D(
        int(x.shape[BN_AXIS] * compression_ratio),
        1,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    # 平均池化
    x = keras.layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x
# 注解,导入类的方式
@keras_cv_export("keras_cv.models.DenseNetBackbone")
class DenseNetBackbone(Backbone): # densenet
    def __init__(
        self,
        *,
        stackwise_num_repeats, # 深度
        include_rescaling,  # 是否归一化
        input_shape=(None, None, 3), 
        input_tensor=None,
        compression_ratio=0.5,# 压缩比
        growth_rate=32, # 增长速度
        **kwargs,
    ):
        # 模型输入(224,224,3)
        inputs = utils.parse_model_inputs(input_shape, input_tensor) 
        x = inputs # 中间变量
        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x) # 归一化
        # 下采样(112,112,3)
        x = keras.layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        # 最大池化,下采样(56,56,3)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1"
        )(x)
        # 金字塔层级特征
        pyramid_level_inputs = {}
        # 遍历每个层级
        for stack_index in range(len(stackwise_num_repeats) - 1):
            index = stack_index + 2
            # 堆叠卷积块
            x = apply_dense_block(
                x,
                stackwise_num_repeats[stack_index], # 当前层级的提取深度
                growth_rate, 
                name=f"conv{index}",
            )
            # P1-->name
            pyramid_level_inputs[f"P{index}"] = utils.get_tensor_input_name(x)
            # 点卷积切换通道,平均池化,下采样
            x = apply_transition_block(
                x, compression_ratio, name=f"pool{index}"
            )
        # 最后一个层级
        x = apply_dense_block(
            x,
            stackwise_num_repeats[-1],
            growth_rate,
            name=f"conv{len(stackwise_num_repeats) + 1}",
        )
        # pn-->name
        pyramid_level_inputs[f"P{len(stackwise_num_repeats) + 1}"] = (
            utils.get_tensor_input_name(x)
        )
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="bn"
        )(x)
        x = keras.layers.Activation("relu", name="relu")(x)
        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        # 设置实例属性
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_num_repeats = stackwise_num_repeats
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.compression_ratio = compression_ratio
        self.growth_rate = growth_rate
    # 获取子类配置的方法
    def get_config(self):
        config = super().get_config() # 获取父类配置方法
        config.update( 
            {
                "stackwise_num_repeats": self.stackwise_num_repeats, # 深度
                "include_rescaling": self.include_rescaling, 
                "input_shape": self.input_shape[1:], # (h,w,c)
                "input_tensor": self.input_tensor,
                "compression_ratio": self.compression_ratio,  # 压缩比
                "growth_rate": self.growth_rate,
            }
        )
        return config
    # 类属性,骨干预设
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    # 类属性,带权重的预设
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
from keras_cv.models import DenseNetBackbone
dense=DenseNetBackbone.from_preset('densenet169_imagenet',(224,224, 3))
[dense.get_layer(i).output for i in dense.pyramid_level_inputs.values()]
dense.get_config()
# {'name': 'dense_net_backbone',
#  'trainable': True,
#  'stackwise_num_repeats': [6, 12, 32, 32],
#  'include_rescaling': True,
#  'input_shape': (None, None, 3),
#  'input_tensor': None,
#  'compression_ratio': 0.5,
#  'growth_rate': 32}
input_data = tf.ones(shape=(1, 224, 224, 3))
# 使用自定义配置随机初始化backbone
model = DenseNetBackbone(
    stackwise_num_repeats=[6, 12, 32, 32],
    include_rescaling=False,
    input_shape=(224,224,3)
)
output = model(input_data)
