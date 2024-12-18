import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import copy
from keras_cv.src import layers as cv_layers
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty
CHANNEL_AXIS = -1
BN_EPSILON = 1e-3
BN_MOMENTUM = 0.999
# 这里的 x + 3.0 操作是为了将输入向右移动，使得当 x 在 -3 到 3 之间变化时，加上 3 后的值会落在 0 到 6 之间。
# 接下来，通过 ReLU(6.0) 层，任何小于 0 的值都会被截断为 0，而大于 6 的值会被截断为6 最后，结果乘以 1.0 / 6.0 
# 是为了将输出缩放到 [0, 1] 范围内。
def apply_hard_sigmoid(x):
    activation = keras.layers.ReLU(6.0)
    return activation(x + 3.0) * (1.0 / 6.0)
class HardSigmoidActivation(keras.layers.Layer):
    def __init__(self):
        super().__init__() 
    def call(self, x):
        return apply_hard_sigmoid(x)
    def get_config(self): # 获取配置方法
        return super().get_config()
# 规范化通道数
def adjust_channels(x, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%.
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x
def apply_hard_swish(x):
    return keras.layers.Multiply()([x, apply_hard_sigmoid(x)])
# 倒残差块
def apply_inverted_res_block(
    x,
    expansion,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    expansion_index,
):

    # 激活函数
    if isinstance(activation, str):
        if activation == "hard_swish":
            activation = apply_hard_swish
        else:
            activation = keras.activations.get(activation)
    shortcut = x
    prefix = "expanded_conv_" # 前缀
    infilters = x.shape[CHANNEL_AXIS] # 特征轴大小
    
    if expansion_index > 0:
        # 前缀
        prefix = f"expanded_conv_{expansion_index}_"
        # 点卷积切换到扩张通道
        x = keras.layers.Conv2D(
            adjust_channels(infilters * expansion), #通道
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=prefix + "expand_BatchNorm",
        )(x)
        x = activation(x)
    # 如果步长是2,设置填充
    if stride == 2:
        # 填充
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, kernel_size),
            name=prefix + "depthwise_pad",
        )(x)
    # 深度卷积
    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "depthwise_BatchNorm",
    )(x)
    x = activation(x)
    # 如果设置了se_ratio
    if se_ratio:
        se_filters = adjust_channels(infilters * expansion) # 通道
        # se模块
        x = cv_layers.SqueezeAndExcite2D(
            filters=se_filters,
            bottleneck_filters=adjust_channels(se_filters * se_ratio),
            squeeze_activation="relu",
            excite_activation=HardSigmoidActivation(),
        )(x)
    # 点卷积切换到收缩通道
    x = keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    # 批次标准化
    x = keras.layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "project_BatchNorm",
    )(x)
    # 残差条件:步长为1,并且输入输出通道数相同
    if stride == 1 and infilters == filters:
        x = keras.layers.Add(name=prefix + "Add")([shortcut, x])
    return x
# MobileNetV3Backbone
@keras_cv_export("keras_cv.models.MobileNetV3Backbone")
class MobileNetV3Backbone(Backbone):
    def __init__(
        self,
        *,
        stackwise_expansion, # 扩张系数
        stackwise_filters, # 通道
        stackwise_kernel_size, # 核大小
        stackwise_stride, # 步长
        stackwise_se_ratio, # se比例
        stackwise_activation, # 激活函数
        include_rescaling, # 是否归一化
        input_shape=(None, None, 3), # 输入形状
        input_tensor=None,
        alpha=1.0,
        **kwargs,
    ):
        # 输入数据(224,224,3)
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs # 中间变量x
        if include_rescaling: 
            x = keras.layers.Rescaling(scale=1 / 255)(x) # 归一化
        # 下采样(112,112,3)
        x = keras.layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv",
        )(x)
        # 批次标准化
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv_BatchNorm",
        )(x)
        # 激活函数
        x = apply_hard_swish(x)
        pyramid_level_inputs = [] # FPN
        # 遍历每个层级
        for stack_index in range(len(stackwise_filters)):
            # 每次下采样的话,就加入特征图层级
            if stackwise_stride[stack_index] != 1:
                pyramid_level_inputs.append(utils.get_tensor_input_name(x))
            # 倒残差块
            x = apply_inverted_res_block(
                x,
                expansion=stackwise_expansion[stack_index], # 扩张比例
                filters=adjust_channels( # 通道
                    (stackwise_filters[stack_index]) * alpha
                ),
                kernel_size=stackwise_kernel_size[stack_index], # 核大小
                stride=stackwise_stride[stack_index], # 步长
                se_ratio=stackwise_se_ratio[stack_index], 
                activation=stackwise_activation[stack_index],
                expansion_index=stack_index,
            )
        
        pyramid_level_inputs.append(utils.get_tensor_input_name(x))
        # 最后的输出层的通道大小
        last_conv_ch = adjust_channels(x.shape[CHANNEL_AXIS] * 6)
        # 点卷积切换通道
        x = keras.layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="Conv_1",
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv_1_BatchNorm",
        )(x)
        x = apply_hard_swish(x)
        
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        # 金字塔层级特征
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
        # 设置实例属性
        self.stackwise_expansion = stackwise_expansion
        self.stackwise_filters = stackwise_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_stride = stackwise_stride
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.alpha = alpha
    # 获取配置的方法
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_filters": self.stackwise_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_stride": self.stackwise_stride,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "alpha": self.alpha,
            }
        )
        return config
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
from keras_cv.models import MobileNetV3Backbone
m1=MobileNetV3Backbone.from_preset('mobilenet_v3_large_imagenet',input_shape=(224,224, 3))
m1.summary()
[m1.get_layer(i).output for i in m1.pyramid_level_inputs.values()]
[<KerasTensor shape=(None, 112, 112, 16), dtype=float32, sparse=False, name=keras_tensor_13>,
 <KerasTensor shape=(None, 56, 56, 24), dtype=float32, sparse=False, name=keras_tensor_31>,
 <KerasTensor shape=(None, 28, 28, 40), dtype=float32, sparse=False, name=keras_tensor_64>,
 <KerasTensor shape=(None, 14, 14, 112), dtype=float32, sparse=False, name=keras_tensor_157>,
 <KerasTensor shape=(None, 7, 7, 160), dtype=float32, sparse=False, name=keras_tensor_208>]
m1.get_config()
# {'name': 'mobile_net_v3_large_backbone',
#  'trainable': True,
#  'stackwise_expansion': [1, 4, 3, 3, 3, 3, 6, 2.5, 2.3, 2.3, 6, 6, 6, 6, 6],
#  'stackwise_filters': [16,
#   24,
#   24,
#   40,
#   40,
#   40,
#   80,
#   80,
#   80,
#   80,
#   112,
#   112,
#   160,
#   160,
#   160],
#  'stackwise_kernel_size': [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5],
#  'stackwise_stride': [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
#  'stackwise_se_ratio': [None,
#   None,
#   None,
#   0.25,
#   0.25,
#   0.25,
#   None,
#   None,
#   None,
#   None,
#   0.25,
#   0.25,
#   0.25,
#   0.25,
#   0.25],
#  'stackwise_activation': ['relu',
#   'relu',
#   'relu',
#   'relu',
#   'relu',
#   'relu',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish',
#   'hard_swish'],
#  'include_rescaling': True,
#  'input_shape': (224, 224, 3),
#  'input_tensor': None,
#  'alpha': 1.0}
