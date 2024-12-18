import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import keras_cv
import numpy as np
import copy
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty
import tensorflow as tf
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNetBackbone,
)
# 普通的卷积残差块
def apply_basic_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    # 预设块名称前缀
    if name is None:
        name = f"v1_basic_block_{keras.backend.get_uid('v1_basic_block_')}"
    # 设置残差连接前段
    # 如果conv_shortcut为True,用点卷积切换通道,之后批次标准化,这时一般要下采样
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else: # 否则不变
        shortcut = x
    # 普通卷积,strides=2时,下采样
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=stride,
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)
    # 第二个普通卷积,步长为1
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = keras.layers.BatchNormalization( # 批次标准化
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    # 注意:残差连接前的两个残差块,都只是批次标准化处理,并没用激活函数
    # 这是因为激活函数会破坏残差的线性,因为卷积是线性的
    x = keras.layers.Add(name=name + "_add")([shortcut, x])
    # 之后经过激活函数处理
    x = keras.layers.Activation("relu", name=name + "_out")(x)
    return x
# 特殊的卷积提取块(宽--窄--宽)
def apply_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    # 预设块前缀 v1_block_1
    if name is None:
        name = f"v1_block_{keras.backend.get_uid('v1_block')}"
    # 如果设置了conv_shortcut=True,用点卷积切换通道(4c),之后批次标准化,这时一般要下采样
    # 这是设置残差前段
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else: # 否则,残差前段=x(传入数据)
        shortcut = x
    # 点卷积切换通道,strides=2时,下采样
    x = keras.layers.Conv2D(
        filters, 1, strides=stride, use_bias=False, name=name + "_1_conv"
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)
    # 普通卷积,步长采用默认1
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    # 批次激活块
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_2_relu")(x)
    # 点卷积切换通道到4c
    x = keras.layers.Conv2D(
        4 * filters, 1, use_bias=False, name=name + "_3_conv"
    )(x)
    x = keras.layers.BatchNormalization( # 批次标准化
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_3_bn"
    )(x)
    # 残差连接,残差前不用激活函数,因为会破坏残差的线性
    x = keras.layers.Add(name=name + "_add")([shortcut, x])
    # 残差后用激活函数(这时通道是4c)
    x = keras.layers.Activation("relu", name=name + "_out")(x)
    return x
# 堆叠的残差块
def apply_stack(
    x,
    filters,
    blocks,
    stride=2,
    name=None,
    block_type="block",
    first_shortcut=True,
):
    # 设置默认名称前缀
    if name is None:
        name = "v1_stack"
    # 根据block_type的类型使用不同的提取块函数
    if block_type == "basic_block":
        block_fn = apply_basic_block # 基本卷积残差块
    elif block_type == "block":
        block_fn = apply_block # 特殊的卷积残差块
    else:
        raise ValueError(
            """`block_type` must be either "basic_block" or "block". """
            f"Received block_type={block_type}."
        )
    # 第一次特征提取,通常要下采样
    x = block_fn(
        x,
        filters,
        stride=stride,
        name=name + "_block1",
        conv_shortcut=first_shortcut,
    )
    # 之后的特征提取,步长一般是1,不进行下采样,只是残差
    for i in range(2, blocks + 1):
        x = block_fn(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x
# keras_cv_export:导入当前类的路径
@keras_cv_export("keras_cv.models.ResNetBackbone")
class ResNetBackbone(Backbone): # resnet骨干
    def __init__(
        self,
        *,
        stackwise_filters, # 通道
        stackwise_blocks,
        stackwise_strides, # 步长列表
        include_rescaling, # 是否内部归一化
        input_shape=(None, None, 3), # 输入形状
        input_tensor=None, # 输入的数据
        block_type="block",
        **kwargs,
    ):
        # 模型输入
        inputs = utils.parse_model_inputs(input_shape, input_tensor) # (224,224,3)
        x = inputs # 中间变量
        # 如果要内部归一化
        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x) # 归一化
        # 第一次下采样(112,112,3)
        x = keras.layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)
        # 批次激活块
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        # 最大池化(56,56,3)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)
        # 不同层级
        num_stacks = len(stackwise_filters)
        # 对应金字塔层级的特征图
        pyramid_level_inputs = {}
        # 遍历不同层级
        for stack_index in range(num_stacks):
            # 应用特征提取模块
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index], # 相同配置的块深度
                stride=stackwise_strides[stack_index],
                block_type=block_type, # 提取块的类型,根据这个选是用基本的卷积块,还是瓶颈块
                # 你看变量名称会坑死你,其实这个是说第一次如果要下采样的话,那残差前段也要跟着下采样
                # 不然你无法残差,条件就是如果block_type == "block"(特殊的卷积残差块)或者
                # stack_index > 0(基本卷积残差块)
                first_shortcut=(block_type == "block" or stack_index > 0),
                name=f"v2_stack_{stack_index}",
            )
            # 对应金字塔层级特征图
            pyramid_level_inputs[f"P{stack_index + 2}"] = (
                utils.get_tensor_input_name(x)
            )

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        # 设置实例属性
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.block_type = block_type
    
    def get_config(self):
        config = super().get_config() # 获取父类的配置字典
        config.update( # 更新字典,加入了子类的配置
            {
                "stackwise_filters": self.stackwise_filters,
                "stackwise_blocks": self.stackwise_blocks,
                "stackwise_strides": self.stackwise_strides,
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "block_type": self.block_type,
            }
        )
        return config
    # 类属性(返回预设的配置)
    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)
    # 类属性(包含权重的配置)
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
# 使用自定义配置随机初始化backbone
model = ResNetBackbone(
    input_shape=(224,224,3),
    stackwise_filters=[64, 128, 256, 512], # 通道数
    stackwise_blocks=[2, 2, 2, 2], # 块深度
    stackwise_strides=[1, 2, 2, 2], # 步长
    include_rescaling=False,
)
len(model.layers)
model.pyramid_level_inputs
[model.get_layer(i).output for i in model.pyramid_level_inputs.values()]
model.summary()
input_data = tf.ones(shape=(8, 224, 224, 3))
output = model(input_data)
output.shape
backbone_presets_no_weights = {
    "resnet18": {
        "metadata": {
            "description": (
                "ResNet model with 18 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 11186112,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet18/2",
    },
    "resnet34": {
        "metadata": {
            "description": (
                "ResNet model with 34 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 21301696,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet34/2",
    },
    "resnet50": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 23561152,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet50/2",
    },
    "resnet101": {
        "metadata": {
            "description": (
                "ResNet model with 101 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 42605504,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet101/2",
    },
    "resnet152": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 58295232,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet152/2",
    },
}

backbone_presets_with_weights = {
    "resnet50_imagenet": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style). "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 23561152,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet50_imagenet/2",
    },
}
# 预设的骨干配置字典
backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
ALIAS_DOCSTRING = """ResNetBackbone (V1) model with {num_layers} layers.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    The difference in ResNetV1 and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNetV1 where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = ResNet{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501 这个注释是不让校验工具报错
# 注解,导入类的路径
@keras_cv_export("keras_cv.models.ResNet18Backbone")
class ResNet18Backbone(ResNetBackbone):
    def __new__( 
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # 把传入参数更新到kwargs里
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        # 获取resnet18骨干网络
        return ResNetBackbone.from_preset("resnet18", **kwargs)

    @classproperty
    def presets(cls):
        return {}

    @classproperty
    def presets_with_weights(cls):
        return {}
model1=ResNet18Backbone(input_shape=(224,224, 3))
model1.summary()
model1.pyramid_level_inputs
@keras_cv_export("keras_cv.models.ResNet34Backbone")
class ResNet34Backbone(ResNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetBackbone.from_preset("resnet34", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}
@keras_cv_export("keras_cv.models.ResNet50Backbone")
class ResNet50Backbone(ResNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetBackbone.from_preset("resnet50", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "resnet50_imagenet": copy.deepcopy(
                backbone_presets["resnet50_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets

@keras_cv_export("keras_cv.models.ResNet101Backbone")
class ResNet101Backbone(ResNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetBackbone.from_preset("resnet101", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}
@keras_cv_export("keras_cv.models.ResNet152Backbone")
class ResNet152Backbone(ResNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetBackbone.from_preset("resnet152", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}
from keras.applications.resnet import ResNet50
aa=ResNet50(include_top=False,input_shape=(224,244,3))
aa.summary()
BN_AXIS = 3
BN_EPSILON = 1.001e-5
model2=ResNet152Backbone(input_shape=(224,224,3))
len(model2.layers)
model2.get_config()
model2=ResNet152Backbone(input_shape=(224,224,3))
len(model2.layers)
[model2.get_layer(i).output for i in model2.pyramid_level_inputs.values()]
model2.get_config()
print(ALIAS_DOCSTRING.format(num_layers=18))
print(ResNet18Backbone.__doc__)
#这些代码片段确实是在为 __doc__ 属性设置文档字符串。setattr 函数被用来动态地修改类的属性。在这个例子中，__doc__ 
# 属性被设置成了一个格式化的字符串，这个字符串包含了有关 ResNet 模型变体的信息，特别是模型的层数。
setattr(ResNet18Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=18))
setattr(ResNet152Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=152))
