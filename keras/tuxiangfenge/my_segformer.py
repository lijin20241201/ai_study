from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
from collections import OrderedDict
from typing import Mapping
from packaging import version
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
# logger = logging.get_logger(__name__)
import keras
import tensorflow as tf
from IPython.display import Image, display
from keras.utils import load_img
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import math
from typing import Optional, Tuple, Union
from transformers.tf_utils import shape_list, stable_softmax
from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from transformers.modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
class SegformerConfig(PretrainedConfig):
    model_type = "segformer"
    def __init__(
        self,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128,320,512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=768,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "reshape_last_stage" in kwargs and kwargs["reshape_last_stage"] is False:
            warnings.warn(
                "Reshape_last_stage is set to False in this config. This argument is deprecated and will soon be"
                " removed, as the behaviour will default to that of reshape_last_stage = True.",
                FutureWarning,
            )

        self.num_channels = num_channels 
        self.num_labels=1
        self.num_encoder_blocks = num_encoder_blocks 
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 实现 Segformer 模型中的 DropPath（随机深度）机制。DropPath 是一种正则化技术，
# 以减少过拟合并提高模型的泛化能力。在训练过程中,在这里的作用是对某些样本的输出随机置0
class TFSegformerDropPath(keras.layers.Layer):
    # 构造函数接受一个 drop_path 参数
    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path
    # call 函数实现了 DropPath 的逻辑。它根据是否处于训练模式 (training) 来决定是否应用 DropPath。
    # 尽管 DropPath 和 Dropout 在实现随机置零方面有类似之处，但它们针对的对象和应用场景不同。DropPath 
    # 主要用于残差网络中的路径，而 Dropout 通常应用于单个神经元。缩放因子 1/keep_prob 的作用是为了在训练
    # 过程中保持路径输出的期望值不变，从而确保模型在训练和测试时表现一致。
    def call(self, x: tf.Tensor, training=None):
        if training: # 训练模式
            keep_prob = 1 - self.drop_path # 计算保留概率 
            # (b,1,1) ,*是复制多少次的意思
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            # shape = (tf.shape(x)[0],) + (1,) * (tf.rank(x)-1)  # 使用 tf.rank 获取动态秩
            # 构造随机张量 random_tensor：形状为 (batch_size, 1, ..., 1)，并在每个样本上均匀采样。
            # 均匀分布是一种概率分布，其中所有区间内的事件发生的概率相等。
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            # 将 random_tensor向下取整，得到一个二值张量（0 或 1）,floor向下取整,即便是
            # 0.99,都会变成0
            random_tensor = tf.floor(random_tensor)
            # 返回经过缩放后的张量：(x / keep_prob) * random_tensor,缩放因子用来保持输出
            # 的期望值不变,会对批次内的整个样本随机置0
            return (x / keep_prob) * random_tensor
        # 非训练模式：直接返回输入张量 x。
        return x
# 构造重叠的补丁嵌入(patch_size比stride大,所以有重叠块)
class TFSegformerOverlapPatchEmbeddings(keras.layers.Layer):
    def __init__(self, patch_size, stride, num_channels, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.proj = keras.layers.Conv2D( # 不填充
            filters=hidden_size, kernel_size=patch_size, strides=stride, padding="VALID", name="proj"
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")
        self.num_channels = num_channels
        self.hidden_size = hidden_size
    # pixel_values:(b,h,w,c)
    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        # 首先对输入图像进行零填充。然后通过卷积层投影补丁嵌入。
        # 在输入图像的边缘进行零填充，以确保卷积操作后得到的补丁具有重叠。
        # 使用卷积层对填充后的输入图像进行卷积操作，生成补丁嵌入。
        embeddings = self.proj(self.padding(pixel_values)) # (b,h,w,d)
        # 计算补丁嵌入的高度、宽度和隐藏维度。
        height = shape_list(embeddings)[1] 
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        # (b,h,w,d)-->(b,h*w,d)
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        # 对展平后的补丁嵌入进行层规范化。
        embeddings = self.layer_norm(embeddings) # 层标准化 (b,s,d)
        return embeddings, height, width
    def build(self, input_shape=None):
        if self.built: # 这个bulit应该是个标记,用来标记是否已经构建
            return
        self.built = True
        # 如果实例有proj属性
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                # 这个参数告诉卷积层 self.proj 输入张量的预期形状。(None,None,None,c)
                # 假设我们有一个输入张量 pixel_values，其形状为 [b, h, w,c]。
                # 当我们第一次调用 self.proj 层时，如果它还没有被构建，就会调用 self.proj.build(
                # [None, None, None, self.num_channels]) 来创建权重矩阵。
                self.proj.build([None, None, None, self.num_channels])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 这个参数告诉标准化层输入张量的预期形状,(None,None,d)
                self.layer_norm.build([None, None, self.hidden_size])
# 自注意力
class TFSegformerEfficientSelfAttention(keras.layers.Layer):
    # SegFormer高效自注意力机制
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size # d
        self.num_attention_heads = num_attention_heads # h
        # d必须能被h整除
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        self.attention_head_size = self.hidden_size // self.num_attention_heads # dk
        self.all_head_size = self.num_attention_heads * self.attention_head_size # h*dk
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size) # scale
        self.query = keras.layers.Dense(self.all_head_size, name="query") # 线性投影
        self.key = keras.layers.Dense(self.all_head_size, name="key")
        self.value = keras.layers.Dense(self.all_head_size, name="value")
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.sr_ratio = sequence_reduction_ratio 
        # 这是query和key,value具有的token数目不同,query的相对具体,key,value的相对抽象,就是被下采样了
        if sequence_reduction_ratio > 1:
            self.sr = keras.layers.Conv2D(
                filters=hidden_size, kernel_size=sequence_reduction_ratio, strides=sequence_reduction_ratio, name="sr"
            )
            # 层标准化
            self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        batch_size = shape_list(tensor)[0] # b
        # (b,s,d)-->(b,s,h,dk)
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # (b,s,h,dk)-->(b,h,s,dk)
        return tf.transpose(tensor, perm=[0, 2, 1, 3])
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # print(f'{hidden_states.shape =}') [1, 625, 32]
        batch_size = shape_list(hidden_states)[0] # b
        num_channels = shape_list(hidden_states)[2] # c
        # (b,h,s,dk)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # 当sr_ratio > 1时,query具有的补丁数和key,value不同
        # 在这里sr_ratio为[8, 4, 2, 1],这是因为,每次嵌入的token不断下采样,去query的key都是
        # 经过下采样后的
        # 通过将所有键的尺寸统一到最后一个特征图的尺寸(8,8)，SegFormer 可以确保不同层次的信息在语义层面的一
        # 致性，并且可以更容易地识别重要特征。这种方法不仅简化了模型的设计，还提高了计算效率，尤其是在
        # 处理高分辨率图像时。
        if self.sr_ratio > 1:
            # (b,s,d)-->(b,h,w,c)
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            # 应用序列缩减 
            hidden_states = self.sr(hidden_states)
            # (b,h,w,d)-->(b,s,d)
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            hidden_states = self.layer_norm(hidden_states) # 层标准化
        # (b,h,s_k,dk)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states)) # (b,h,s_v,dk)
        # (1, 1, 625, 32) (1, 1, 9, 32) (1, 1, 9, 32)
        # print(query_layer.shape,key_layer.shape,value_layer.shape)
        # (b,h,s_q,dk)@(b,h,dk,s_k)-->(b,h,s_q,s_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype) 
        attention_scores = tf.divide(attention_scores, scale) # 缩放
        # 在s_k所在的轴归一化,这样s_q上指定的token就和s_k上所有有效token
        # 形成了相似性,这些加起来是1
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        #  对注意力矩阵中的值随机置0
        # 在实际应用中，Dropout 的效果通常是在训练过程中增强模型的鲁棒性，使模型能够在测试时更好地处理新的数据。
        # 尽管某些高得分的注意力权重可能会被置为 0，但这有助于模型学习更广泛的特征组合，并且在测试时能够更好地
        # 应对不同的输入情况。
        attention_probs = self.dropout(attention_probs, training=training)
        # print(f'{attention_probs.shape =}') [1, 1, 625, 9]
        # (b,h,s_q,s_k)@(b,h,s_v,dk)-->(b,h,s_q,dk)
        # 这里主要是s_k轴上的分数和s_v上的每个token的表示做加权和,
        context_layer = tf.matmul(attention_probs, value_layer)
        # (b,h,s_q,dk)-->(b,s_q,h,dk)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        # (b,s_q,h,dk)-->(b,s_q,d)
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))
        # 注意力输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    def build(self, input_shape=None):
        if self.built: # 如果已经构建,return 
            return
        self.built = True
        # 如果实例有query属性
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                # 告诉query层,接受的输入形状是(None,None,d)
                self.query.build([None, None, self.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.hidden_size])
        if getattr(self, "sr", None) is not None:
            with tf.name_scope(self.sr.name):
                # (b,h,w,c)
                self.sr.build([None, None, None, self.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])

# 注意力机制的最后一个线性投影层
class TFSegformerSelfOutput(keras.layers.Layer):
    def __init__(self, config: SegformerConfig, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(hidden_states) # (b,s,d)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])

class TFSegformerAttention(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self = TFSegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            name="self",
        )
        self.dense_output = TFSegformerSelfOutput(config, hidden_size=hidden_size, name="output")

    def call(
        self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # (b,s,d)
        self_outputs = self.self(hidden_states, height=height, width=width, output_attentions=output_attentions)
        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  
        return outputs
    def build(self, input_shape=None):
        if self.built: # 如果已经构建
            return
        self.built = True
        # tf.name_scope 是 TensorFlow 中用于给操作或变量命名的一个工具。在构建计算图时，使用 with 
        # tf.name_scope 可以创建一个命名空间，这样可以使得在 TensorBoard 中的可视化更加清晰，同时
        # 也可以避免名字冲突的问题。
        # build 方法是在定义一个自定义层时用来初始化层内部的变量（如权重）。在这个方法中调用 build(None)
        # 表示即使没有具体的输入形状信息，也要构建这个子层。
        # 这里需要注意的是，如果你的子层确实需要知道输入的形状来正确地初始化其内部变量，那么传递 None 可能会导致
        # 问题。因此，最好确保 build 方法能够处理这种情况，或者确保在调用 build 时传入正确的 input_shape。
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)

# 先变形成卷积数据输入形状,深度卷积后再变回来
class TFSegformerDWConv(keras.layers.Layer):
    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_convolution = keras.layers.Conv2D(
            filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, name="dwconv"
        )
        self.dim = dim
    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor:
        batch_size = shape_list(hidden_states)[0] # b
        num_channels = shape_list(hidden_states)[-1] # c
        # (b,s,d)-->(b,h,w,d)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        # 深度卷积
        hidden_states = self.depthwise_convolution(hidden_states)
        new_height = shape_list(hidden_states)[1] # h
        new_width = shape_list(hidden_states)[2] # w
        num_channels = shape_list(hidden_states)[3] # c
        # (b,h,w,c)-->(b,h*w,c)
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        return hidden_states
    def build(self, input_shape=None):
        if self.built:  # built是个标记,用来标记构建状态
            return
        self.built = True
        if getattr(self, "depthwise_convolution", None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                # 说明期望的传入数据形状是(None, None, None, c)
                self.depthwise_convolution.build([None, None, None, self.dim])

# 前馈层
class TFSegformerMixFFN(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        **kwargs,
    ):
        # 初始化时设置了几个关键组件，包括两个 Dense 层（dense1 和 dense2），一个深度可分离卷积层
        # （depthwise_convolution），以及一个 Dropout 层。
        super().__init__(**kwargs)
        out_features = out_features or in_features
        self.dense1 = keras.layers.Dense(hidden_features, name="dense1")
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name="dwconv")
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = keras.layers.Dense(out_features, name="dense2")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_features = hidden_features
        self.in_features = in_features
    # 这个方法实现了数据流过该层的过程。首先通过 dense1 层进行线性变换，然后通过 depthwise_convolution 
    # 层进行深度可分离卷积，接着应用激活函数和 dropout，最后再次通过 dense2 层进行线性变换并应用 dropout。
    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 投影
        hidden_states = self.dense1(hidden_states)
        # 深度卷积
        hidden_states = self.depthwise_convolution(hidden_states, height=height, width=width)
        hidden_states = self.intermediate_act_fn(hidden_states) # 激活函数处理
        hidden_states = self.dropout(hidden_states, training=training) #dropout
        hidden_states = self.dense2(hidden_states) # 投影
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states
    # build方法用于初始化层内部的权重。这里对每个子层分别调用了 build 方法，并使用 tf.name_scope
    # 为它们创建了命名空间,input_shape 参数在这里被忽略，因为子层的构建是基于类属性 in_features 
    # 和 hidden_features 的。
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense1", None) is not None:
            with tf.name_scope(self.dense1.name):
                self.dense1.build([None, None, self.in_features])
        if getattr(self, "depthwise_convolution", None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                self.depthwise_convolution.build(None)
        if getattr(self, "dense2", None) is not None:
            with tf.name_scope(self.dense2.name):
                self.dense2.build([None, None, self.hidden_features])

# 相当于transformer编码器层
class TFSegformerLayer(keras.layers.Layer):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
        **kwargs,
    ):
        # 初始化时设置了 Layer Normalization 层、多头注意力机制层、随机深度层和混合前馈网络层。
        super().__init__(**kwargs)
        # 标准化层
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_1")
        self.attention = TFSegformerAttention( # 注意力
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            name="attention",
        )
        # 随机深度
        # Activation("linear") 指的是线性激活函数，它实际上等同于恒等函数（identity function），即输出等于输入。
        self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else keras.layers.Activation("linear")
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_2")
        mlp_hidden_size = int(hidden_size * mlp_ratio) 
        # 前馈层
        self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name="mlp")
        self.hidden_size = hidden_size # d
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple:
        # 首先对输入数据进行 Layer Normalization 处理，然后通过注意力机制层。
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # 先标准化 
            height=height,
            width=width,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:] # 注意力矩阵
        # 随机深度(对样本的操作,会把一个批次内的某些样本的输出整个置0)
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states # 注意力前后残差
        # 前馈层输出
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height=height, width=width)
        # 随机深度(对样本的操作,会把一个批次内的某些样本的输出整个置0)
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states # 前馈前后残差
        outputs = (layer_output,) + outputs # 编码器输出+(注意力矩阵)
        return outputs # (b,s,d)
    def build(self, input_shape=None):
        if self.built: # 如果已经构建,返回
            return
        self.built = True
        if getattr(self, "layer_norm_1", None) is not None:
            with tf.name_scope(self.layer_norm_1.name):
                self.layer_norm_1.build([None, None, self.hidden_size])
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "layer_norm_2", None) is not None:
            with tf.name_scope(self.layer_norm_2.name):
                self.layer_norm_2.build([None, None, self.hidden_size])
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)

class TFSegformerEncoder(keras.layers.Layer): # encoder
    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs) # 调用父类（Layer）的构造函数。
        self.config = config # 设置配置对象 config，这个对象包含了模型的各种超参数。
        # 生成一个从 0 到 config.drop_path_rate 的线性递增序列，长度为所有编码器块层数的总和。
        # 将生成的dropout率存储在列表 drop_path_decays 中
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, config.drop_path_rate, sum(config.depths))]
        # 保存每个编码器块之前的嵌入
        embeddings = [] 
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                # 创建一个重叠补丁嵌入层，该层将输入图像分割成补丁，并进行嵌入。
                TFSegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    name=f"patch_embeddings.{i}",
                )
            )
        self.embeddings = embeddings # 变成实例属性
        blocks = [] # 用来存储编码器块,每个编码器块对应几个编码器层
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            # 遍历每个编码器块对应的层数,每个层都是一个编码器层
            for j in range(config.depths[i]):
                layers.append(
                    TFSegformerLayer( # 相当于一个编码器层
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        name=f"block.{i}.{j}",
                    )
                )
            blocks.append(layers)
        self.block = blocks
        # 对应每个编码器块的层标准化
        self.layer_norms = [
            keras.layers.LayerNormalization(epsilon=1e-05, name=f"layer_norm.{i}")
            for i in range(config.num_encoder_blocks)
        ]
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 初始化存储编码器块输出的元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = shape_list(pixel_values)[0] # b
        hidden_states = pixel_values 
        # 遍历每个编码器块对应的嵌入，编码器，标准化
        for idx, x in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            # 对应当前编码器的初始输入，一个block可能由几个编码器层组成,标准化层
            embedding_layer, block_layer, norm_layer = x # 拆包
            # 先是把4维的数据转换成嵌入形式
            hidden_states, height, width = embedding_layer(hidden_states)
            # 每个块由多个层组成，即编码器层列表
            for i, blk in enumerate(block_layer):
                layer_outputs = blk( # 编码器层的输出
                    hidden_states,
                    height=height,
                    width=width,
                    output_attentions=output_attentions,
                    training=training,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # 层标准化
            hidden_states = norm_layer(hidden_states)
            # 如果不是最后的一个编码器块或者设置了对最后一个块的输出变形
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = shape_list(hidden_states)[-1] # d
                # (b,s,d)--.(b,h,w,c)
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            if output_hidden_states: # 如果要输出每个编码器块的输出
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict: # 如果设置不返回字典
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
    def build(self, input_shape=None):
        if self.built: # 如果已经构建
            return
        self.built = True
        # 根据实例的layer_norms,block,embeddings设置权重w
        if getattr(self, "layer_norms", None) is not None:
            for layer, shape in zip(self.layer_norms, self.config.hidden_sizes):
                with tf.name_scope(layer.name):
                    layer.build([None, None, shape])
        if getattr(self, "block", None) is not None:
            for block in self.block:
                for layer in block:
                    with tf.name_scope(layer.name):
                        layer.build(None)
        if getattr(self, "embeddings", None) is not None:
            for layer in self.embeddings:
                with tf.name_scope(layer.name):
                    layer.build(None)

@keras_serializable # 可序列化注解,主要是把输出变形成(b,c,h,w形式)
class TFSegformerMainLayer(keras.layers.Layer):
    config_class = SegformerConfig # 配置类
    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 分层transformer encoder
        self.encoder = TFSegformerEncoder(config, name="encoder")
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 是否输出注意力矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 是否输出编码器块的输出
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 是否返回字典形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 这里传入数据形状就是(b,h,w,c)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1)) # 转换格式成(b,h,w,c)
        encoder_outputs = self.encoder( # 编码器输出
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = encoder_outputs[0]
        # (b,h,w,c)-->(b,c,h,w)
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])
        # 转换其他编码器块的输出为(b,c,h,w)
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple(tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0])
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]
        
        return TFBaseModelOutput(
            last_hidden_state=sequence_output, # (b,c,h,w)
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

class TFSegformerPreTrainedModel(TFPreTrainedModel):
    config_class = SegformerConfig # 配置类
   #  通常用于在模型的状态字典中识别特定的键名，以便于加载和保存权重。
    base_model_prefix = "segformer"
    # 指定了模型的主要输入名称，这里设置为 "pixel_values"，这表明该模型期望接收像素值作为输入。
    main_input_name = "pixel_values"
    # 输入签名: input_signature 是一个属性，它返回一个字典，描述了模型期望接收的输入张量的形状和数据类型。
    # 这里的输入是一个形状为 (None,c, 512, 512) 的浮点数张量，其中 None 表示批次大小可以动态变化，
    # self.config.num_channels 是输入图像的通道数
    # 确保你在调整输入签名后，模型内部也能够正确处理新的输入尺寸。特别是如果你的模型中有对输入尺寸有特定要求的操
    # 作，比如卷积层的滤波器大小等，那么需要确保这些操作仍然适用。
    @property
    def input_signature(self):
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels,160,320), dtype=tf.float32)}
class TFSegformerModel(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.segformer = TFSegformerMainLayer(config, name="segformer")
    # @unpack_inputs 装饰器确保了 call 方法中的 pixel_values 总是一个张量，即使输入是一个包含多个张
    # 量的结构。这使得 call 方法内部可以专注于处理单一的张量输入，而不需要关心输入的具体形式。
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        outputs = self.segformer( # 返回编码器的输出(b,c,h,w)
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "segformer", None) is not None:
            with tf.name_scope(self.segformer.name):
                # segformer期望输入的形状不固定
                self.segformer.build(None)

class TFSequenceClassificationLoss:
    # 适合序列分类的损失函数。
    def hf_compute_loss(self, labels, logits):
        # 回归任务：当 logits 为一维张量时，使用均方误差（Mean Squared Error）作为损失函数。
        # 二分类任务：当 logits 的第二个维度为 1 时，同样使用均方误差。
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                # 如果标签是1D，则MeanSquaredError返回标量损失，因此要避免这种情况
                labels = tf.expand_dims(labels, axis=-1)
        else:
            # 多分类任务：当 logits 为多维张量时，使用稀疏类别交叉熵（Sparse Categorical Crossentropy
            # ）作为损失函数。
            # from_logits=True 表示 logits 是未经过 softmax 处理的原始输出。
            # reduction=keras.losses.Reduction.NONE 表示不对损失进行平均或求和，而是返回每个样本的损失值。
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.NONE
            )
        return loss_fn(labels, logits)

class TFSegformerForImageClassification(TFSegformerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels # 几分类
        self.segformer = TFSegformerMainLayer(config, name="segformer") # 编码器
        # 分类器
        self.classifier = keras.layers.Dense(config.num_labels, name="classifier")
        self.config = config
    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        outputs = self.segformer( # 返回的输出形式是(b,c,h,w)
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        batch_size = shape_list(sequence_output)[0] # b
        #(b,c,h,w)-->(b,h,w,c)
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
        # (b,h,w,c)-->(b,h*w,d)
        sequence_output = tf.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))
        # 在索引1的轴取均值,也就是在token序列的轴取均值 (b,d)
        sequence_output = tf.reduce_mean(sequence_output, axis=1) # (1, 256)
        # print(sequence_output.shape)
        # (b,d)-->(b,num_labels)
        logits = self.classifier(sequence_output)
        # 返回的是未平均样本的损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels,logits=logits)
        # print(loss.shape,logits.shape) (1,) (1, 2) 
        # 如果不返回字典,把logits和outputs[1:]拼接
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TFSequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "segformer", None) is not None:
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        if getattr(self, "classifier", None) is not None:
            # 分类器
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])

class TFSegformerMLP(keras.layers.Layer):
    def __init__(self, input_dim: int, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.proj = keras.layers.Dense(config.decoder_hidden_size, name="proj")
        self.input_dim = input_dim
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        height = shape_list(hidden_states)[1] # h
        width = shape_list(hidden_states)[2] # w
        hidden_dim = shape_list(hidden_states)[-1] # d
        # (b,h,w,c)-->(b,h*w,d)
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))
        hidden_states = self.proj(hidden_states) # 投影
        return hidden_states
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.input_dim])

# 实现了用于语义分割任务的解码头部分，它接收来自编码器块的多个特征图，并通过一系列处理步骤来生成最终的分割图
class TFSegformerDecodeHead(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        mlps = [] # 对应每个编码器块,里面对应每个mlp类实例
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config=config, input_dim=config.hidden_sizes[i], name=f"linear_c.{i}")
            mlps.append(mlp)
        self.mlps = mlps
        # 下面3层实现了原实现的ConvModule
        self.linear_fuse = keras.layers.Conv2D( # 点卷积
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse"
        )
        self.batch_norm = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = keras.layers.Activation("relu")
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        # 点卷积
        self.classifier = keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name="classifier")
        self.config = config
    # encoder_hidden_states 是一个包含多个特征图的列表，每个特征图对应编码器块的不同编码阶段
    def call(self, encoder_hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.mlps):
            # 如果最后一个编码器的输出是3维张量
            # 对于每个特征图，如果 reshape_last_stage 为 False 并且特征图的形状为 3D（即 (b, s, d)），
            # 则将其重塑为 4D 形状（即 (b, h, w, d)）。
            if self.config.reshape_last_stage is False and len(shape_list(encoder_hidden_state)) == 3:
                # 开方
                height = tf.math.sqrt(tf.cast(shape_list(encoder_hidden_state)[1], tf.float32)) # h
                height = width = tf.cast(height, tf.int32)
                channel_dim = shape_list(encoder_hidden_state)[-1] # d
                # (b,s,d)-->(b,h,w,d)
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
                encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0,3,1,2])
            # 除了if内的代码,正常的encoder_hidden_state(b,c,h,w)-->(b,h,w,c)
            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            height, width = shape_list(encoder_hidden_state)[1:3] # h,w
            # 投影成(b,s,d),统一通道数
            encoder_hidden_state = mlp(encoder_hidden_state) 
            channel_dim = shape_list(encoder_hidden_state)[-1] # d
            # (b,s,d)-->(b,h,w,d)
            encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
            # encoder_hidden_states[0]这个是第一个特征图,是尺寸最大的
            # (b,c,h0,w0)-->(b,h0,w0,c)
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = shape_list(temp_state)[1:-1] # (h,w)
            # 把每个编码器块的输出改变成第一个特征图的尺寸
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method="bilinear")
            # 保存每个编码器输出的特征图,是个元组,所有的都统一到了同一个尺寸
            all_hidden_states += (encoder_hidden_state,)
        # [TensorShape([1, 64, 64, 256]), TensorShape([1, 64, 64, 256]),
        #  TensorShape([1, 64, 64, 256]), TensorShape([1, 64, 64, 256])]
        # print([i.shape for i in all_hidden_states])
        # all_hidden_states[::-1]会逆序排列,就是抽象的特征在前,细节化的特征在后
        # concat会在最后一个轴合并,之后点卷积切换通道,变成(b,h,w,d)
        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states) # relu处理
        hidden_states = self.dropout(hidden_states, training=training)
        # (b,h,w,num_labels) 点卷积切换通道
        logits = self.classifier(hidden_states)
        return logits
    def build(self, input_shape=None):
        if self.built: # 如果已经构建,返回
            return
        self.built = True
        if getattr(self, "linear_fuse", None) is not None:
            with tf.name_scope(self.linear_fuse.name):
                self.linear_fuse.build(
                    [None, None, None, self.config.decoder_hidden_size * self.config.num_encoder_blocks]
                )
        if getattr(self, "batch_norm", None) is not None:
            with tf.name_scope(self.batch_norm.name):
                self.batch_norm.build([None, None, None, self.config.decoder_hidden_size])
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.config.decoder_hidden_size])
        if getattr(self, "mlps", None) is not None:
            for layer in self.mlps:
                with tf.name_scope(layer.name):
                    layer.build(None)

# 语义分割
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.segformer = TFSegformerMainLayer(config, name="segformer")
        self.decode_head = TFSegformerDecodeHead(config, name="decode_head")
    def hf_compute_loss(self, logits, labels):
        # labels:(b,h,w),logits(b,h,w,c)
        label_interp_shape = shape_list(labels)[1:] # (h,w)
        # 这里的说是上采样,其实就是改变特征图的尺寸
        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # 损失函数:多元交叉熵,from_logits=True,没有用softmax归一化
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        def masked_loss(real, pred):
            unmasked_loss = loss_fct(real, pred) # 不带掩码的损失
            # 获取忽略ignore_index的掩码
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask # 会忽略掉其中一些要忽略的ignore_index
            # 这样获取到的就是平均标签损失
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))
        return masked_loss(labels,upsampled_logits) # 返回损失
    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor],
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFSemanticSegmenterOutput]:
        # 获取是否返回字典格式输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 是否输出所有的编码输出
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果有标签,并且标签值不大于1
        # if labels is not None and not self.config.num_labels > 1:
        #     raise ValueError("The number of labels should be greater than one")
        
        outputs = self.segformer( # (b,c,h,w)
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True, # 输出hidden_states
            return_dict=return_dict,
        )
        # 获取所有编码器块的输出
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 获取最后模型的预测(b,h,w,num_labels)
        logits = self.decode_head(encoder_hidden_states)
        logits = tf.image.resize(logits, size=(160,320), method="bilinear")
        logits =tf.nn.sigmoid(logits)
        # loss = None
        # if labels is not None:
        #     # 计算损失
        #     loss = self.hf_compute_loss(logits=logits, labels=labels)
        # (b,h,w,num_labels)-->(b,num_labels,h,w)
        # logits = tf.transpose(logits, perm=[0, 3, 1, 2])
        if not return_dict: # 如果不返回字典的话
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            # return ((loss,) + output) if loss is not None else output
            return output
        # return TFSemanticSegmenterOutput(
        #     loss=loss, # 损失
        #     logits=logits, # 模型预测
        #     hidden_states=outputs.hidden_states if output_hidden_states else None,
        #     attentions=outputs.attentions,
        # )
        return TFSemanticSegmenterOutput(
            logits=logits, # 模型预测
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "segformer", None) is not None:
            # 期望的输入:任意
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        if getattr(self, "decode_head", None) is not None:
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)

# class SegformerImageProcessor(BaseImageProcessor):
#     model_input_names = ["pixel_values"]
#     @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="4.41.0")
#     @filter_out_non_signature_kwargs(extra=INIT_SERVICE_KWARGS)
#     def __init__(
#         self,
#         do_resize: bool = True,
#         size: Dict[str, int] = None,
#         resample: PILImageResampling = PILImageResampling.BILINEAR,
#         do_rescale: bool = True,
#         rescale_factor: Union[int, float] = 1 / 255,
#         do_normalize: bool = True,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         do_reduce_labels: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         size = size if size is not None else {"height": 512, "width": 512}
#         size = get_size_dict(size)
#         self.do_resize = do_resize
#         self.size = size
#         self.resample = resample
#         self.do_rescale = do_rescale
#         self.rescale_factor = rescale_factor
#         self.do_normalize = do_normalize
#         self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
#         self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
#         self.do_reduce_labels = do_reduce_labels
#     @classmethod
#     def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
#         """
#         Overrides the `from_dict` method from the base class to save support of deprecated `reduce_labels` in old configs
#         """
#         image_processor_dict = image_processor_dict.copy()
#         if "reduce_labels" in image_processor_dict:
#             image_processor_dict["do_reduce_labels"] = image_processor_dict.pop("reduce_labels")
#         return super().from_dict(image_processor_dict, **kwargs)
#     def resize(
#         self,
#         image: np.ndarray,
#         size: Dict[str, int],
#         resample: PILImageResampling = PILImageResampling.BILINEAR,
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ) -> np.ndarray:
#         size = get_size_dict(size)
#         if "height" not in size or "width" not in size:
#             raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
#         output_size = (size["height"], size["width"])
#         return resize(
#             image,
#             size=output_size,
#             resample=resample,
#             data_format=data_format,
#             input_data_format=input_data_format,
#             **kwargs,
#         )
#     def reduce_label(self, label: ImageInput) -> np.ndarray:
#         label = to_numpy_array(label)
#         # Avoid using underflow conversion
#         label[label == 0] = 255
#         label = label - 1
#         label[label == 254] = 255
#         return label

#     def _preprocess(
#         self,
#         image: ImageInput,
#         do_reduce_labels: bool,
#         do_resize: bool,
#         do_rescale: bool,
#         do_normalize: bool,
#         size: Optional[Dict[str, int]] = None,
#         resample: PILImageResampling = None,
#         rescale_factor: Optional[float] = None,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#     ):
#         if do_reduce_labels:
#             image = self.reduce_label(image)

#         if do_resize:
#             image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

#         if do_rescale:
#             image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

#         if do_normalize:
#             image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

#         return image

#     def _preprocess_image(
#         self,
#         image: ImageInput,
#         do_resize: bool = None,
#         size: Dict[str, int] = None,
#         resample: PILImageResampling = None,
#         do_rescale: bool = None,
#         rescale_factor: float = None,
#         do_normalize: bool = None,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#     ) -> np.ndarray:
#         """Preprocesses a single image."""
#         # All transformations expect numpy arrays.
#         image = to_numpy_array(image)
#         if is_scaled_image(image) and do_rescale:
#             logger.warning_once(
#                 "It looks like you are trying to rescale already rescaled images. If the input"
#                 " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
#             )
#         if input_data_format is None:
#             input_data_format = infer_channel_dimension_format(image)
#         image = self._preprocess(
#             image=image,
#             do_reduce_labels=False,
#             do_resize=do_resize,
#             size=size,
#             resample=resample,
#             do_rescale=do_rescale,
#             rescale_factor=rescale_factor,
#             do_normalize=do_normalize,
#             image_mean=image_mean,
#             image_std=image_std,
#             input_data_format=input_data_format,
#         )
#         if data_format is not None:
#             image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
#         return image

#     def _preprocess_mask(
#         self,
#         segmentation_map: ImageInput,
#         do_reduce_labels: bool = None,
#         do_resize: bool = None,
#         size: Dict[str, int] = None,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#     ) -> np.ndarray:
#         """Preprocesses a single mask."""
#         segmentation_map = to_numpy_array(segmentation_map)
#         # Add channel dimension if missing - needed for certain transformations
#         if segmentation_map.ndim == 2:
#             added_channel_dim = True
#             segmentation_map = segmentation_map[None, ...]
#             input_data_format = ChannelDimension.FIRST
#         else:
#             added_channel_dim = False
#             if input_data_format is None:
#                 input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
#         # reduce zero label if needed
#         segmentation_map = self._preprocess(
#             image=segmentation_map,
#             do_reduce_labels=do_reduce_labels,
#             do_resize=do_resize,
#             resample=PILImageResampling.NEAREST,
#             size=size,
#             do_rescale=False,
#             do_normalize=False,
#             input_data_format=input_data_format,
#         )
#         # Remove extra channel dimension if added for processing
#         if added_channel_dim:
#             segmentation_map = segmentation_map.squeeze(0)
#         segmentation_map = segmentation_map.astype(np.int64)
#         return segmentation_map

#     def __call__(self, images, segmentation_maps=None, **kwargs):
#         return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

#     @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="4.41.0")
#     @filter_out_non_signature_kwargs()
#     def preprocess(
#         self,
#         images: ImageInput,
#         segmentation_maps: Optional[ImageInput] = None,
#         do_resize: Optional[bool] = None,
#         size: Optional[Dict[str, int]] = None,
#         resample: PILImageResampling = None,
#         do_rescale: Optional[bool] = None,
#         rescale_factor: Optional[float] = None,
#         do_normalize: Optional[bool] = None,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         do_reduce_labels: Optional[bool] = None,
#         return_tensors: Optional[Union[str, TensorType]] = None,
#         data_format: ChannelDimension = ChannelDimension.FIRST,
#         input_data_format: Optional[Union[str, ChannelDimension]] = None,
#     ) -> PIL.Image.Image:
#         do_resize = do_resize if do_resize is not None else self.do_resize
#         do_rescale = do_rescale if do_rescale is not None else self.do_rescale
#         do_normalize = do_normalize if do_normalize is not None else self.do_normalize
#         do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
#         resample = resample if resample is not None else self.resample
#         size = size if size is not None else self.size
#         rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
#         image_mean = image_mean if image_mean is not None else self.image_mean
#         image_std = image_std if image_std is not None else self.image_std

#         images = make_list_of_images(images)

#         if segmentation_maps is not None:
#             segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

#         if not valid_images(images):
#             raise ValueError(
#                 "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
#                 "torch.Tensor, tf.Tensor or jax.ndarray."
#             )
#         validate_preprocess_arguments(
#             do_rescale=do_rescale,
#             rescale_factor=rescale_factor,
#             do_normalize=do_normalize,
#             image_mean=image_mean,
#             image_std=image_std,
#             do_resize=do_resize,
#             size=size,
#             resample=resample,
#         )

#         images = [
#             self._preprocess_image(
#                 image=img,
#                 do_resize=do_resize,
#                 resample=resample,
#                 size=size,
#                 do_rescale=do_rescale,
#                 rescale_factor=rescale_factor,
#                 do_normalize=do_normalize,
#                 image_mean=image_mean,
#                 image_std=image_std,
#                 data_format=data_format,
#                 input_data_format=input_data_format,
#             )
#             for img in images
#         ]

#         data = {"pixel_values": images}

#         if segmentation_maps is not None:
#             segmentation_maps = [
#                 self._preprocess_mask(
#                     segmentation_map=segmentation_map,
#                     do_reduce_labels=do_reduce_labels,
#                     do_resize=do_resize,
#                     size=size,
#                     input_data_format=input_data_format,
#                 )
#                 for segmentation_map in segmentation_maps
#             ]
#             data["labels"] = segmentation_maps

#         return BatchFeature(data=data, tensor_type=return_tensors)

#     # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation with Beit->Segformer
#     def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
#         # TODO: add support for other frameworks
#         logits = outputs.logits

#         # Resize logits and compute semantic segmentation maps
#         if target_sizes is not None:
#             if len(logits) != len(target_sizes):
#                 raise ValueError(
#                     "Make sure that you pass in as many target sizes as the batch dimension of the logits"
#                 )

#             if is_torch_tensor(target_sizes):
#                 target_sizes = target_sizes.numpy()

#             semantic_segmentation = []

#             for idx in range(len(logits)):
#                 resized_logits = torch.nn.functional.interpolate(
#                     logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
#                 )
#                 semantic_map = resized_logits[0].argmax(dim=0)
#                 semantic_segmentation.append(semantic_map)
#         else:
#             semantic_segmentation = logits.argmax(dim=1)
#             semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

#         return semantic_segmentation
# class SegformerFeatureExtractor(SegformerImageProcessor):
#     def __init__(self, *args, **kwargs) -> None:
#         warnings.warn(
#             "The class SegformerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
#             " Please use SegformerImageProcessor instead.",
#             FutureWarning,
#         )
#         super().__init__(*args, **kwargs)