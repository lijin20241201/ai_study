import numpy as np
from keras_cv.src.backend import ops
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
import tensorflow as tf
import keras_cv
images = np.ones(shape=(1, 96, 96, 3))
labels = np.zeros(shape=(1, 96, 96, 1))
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
[backbone.get_layer(i).output for i in backbone.pyramid_level_inputs.values()]
backbone.get_config()
# {'name': 'mi_t_backbone',
#  'trainable': True,
#  'depths': [2, 2, 2, 2],
#  'embedding_dims': [32, 64, 160, 256],
#  'include_rescaling': True,
#  'input_shape': (224, 224, 3),
#  'input_tensor': None}

# 实现MixTransformer架构的Keras模型
# 导入类的路径
@keras_cv_export("keras_cv.models.MiTBackbone")
class MiTBackbone(Backbone):
    def __init__(
        self,
        include_rescaling, # 是否归一花
        depths, # 深度
        input_shape=(224, 224, 3), # 输入形状
        input_tensor=None, # 输入数据
        embedding_dims=None, # 嵌入维度
        **kwargs, 
    ):
        drop_path_rate = 0.1  # drop率
        # dropout列表
        dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8] # 头数
        blockwise_sr_ratios = [8, 4, 2, 1] # sr比率
        num_stages = 4  # 阶段
        cur = 0  
        patch_embedding_layers = [] # 补丁嵌入层列表
        transformer_blocks = [] # transformer模块
        layer_norms = []  # 层标准化
        # 阶段
        for i in range(num_stages):
            # 补丁嵌入层,重叠块嵌入
            patch_embed_layer = cv_layers.OverlappingPatchingAndEmbedding(
                project_dim=embedding_dims[i], # 块通道数
                patch_size=7 if i == 0 else 3, # 块大小
                stride=4 if i == 0 else 2, # 步长
                name=f"patch_and_embed_{i}", # 块嵌入
            )
            patch_embedding_layers.append(patch_embed_layer)
            # 分层transformer编码器,多个层
            transformer_block = [
                # 分层transformer encoder
                cv_layers.HierarchicalTransformerEncoder( 
                    project_dim=embedding_dims[i], # 通道
                    num_heads=blockwise_num_heads[i], # 头数
                    sr_ratio=blockwise_sr_ratios[i], # sr比率
                    drop_prob=dpr[cur + k], #  每一层的dropout
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i])
            ]
            # transformerrs块列表
            transformer_blocks.append(transformer_block)
            cur += depths[i] # 改变块索引
            # 层标准化
            layer_norms.append(keras.layers.LayerNormalization())
        # 模型输入
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs # 中间变量
        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x) # 归一化

        pyramid_level_inputs = []
        for i in range(num_stages):
            # Compute new height/width after the `proj`
            # call in `OverlappingPatchingAndEmbedding`
            stride = 4 if i == 0 else 2
            new_height, new_width = (
                int(ops.shape(x)[1] / stride),
                int(ops.shape(x)[2] / stride),
            )
            # 金字塔层级特征,嵌入
            x = patch_embedding_layers[i](x)
            # transformer处理
            for blk in transformer_blocks[i]:
                x = blk(x)
            x = layer_norms[i](x) # 标准化
            x = keras.layers.Reshape(  # 变形
                (new_height, new_width, -1), name=f"output_level_{i}"
            )(x)
            # 金字塔层级特征
            pyramid_level_inputs.append(utils.get_tensor_input_name(x))
        # 构建模型
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        # 设置配置属性
        self.depths = depths
        self.embedding_dims = embedding_dims
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        # 金字塔层级特征
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
    # 获取配置
    def get_config(self):
        config = super().get_config() # 获取父类配置
        config.update( # 更新配置,加入子类特有的配置
            {
                "depths": self.depths,
                "embedding_dims": self.embedding_dims,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config
    # 类属性
    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
    @classproperty
    def presets_with_weights(cls):
        return copy.deepcopy(backbone_presets_with_weights)
