import os
import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def mlp(x, hidden_units, dropout_rate): # 线性转换,也叫多层感知机
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size # 图片被分成的小单元大小
    def call(self, images): # 前向传播
        input_shape = ops.shape(images) #（1,72,72,3）
        batch_size = input_shape[0] # 1
        # height = input_shape[1] # 72
        # width = input_shape[2]
        channels = input_shape[-1] # 通道
        # num_patches_h = height // self.patch_size # 列上的小单元个数
        # num_patches_w = width // self.patch_size # 行上的小单元个数
        # 不能这样用,必须用extract_patches,不然顺序是乱的
        # patches=tf.reshape(images,(batch_size,num_patches_h*num_patches_w,-1))
        # 根据图片和小单元大小提取小单元
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        # print(patches.shape) # (1, 15, 15, 300)
        patches = ops.reshape(
            patches,
            ( batch_size,# 批次大小,
             -1,
            # num_patches_h * num_patches_w, # 图片被分成的小单元个数
            self.patch_size * self.patch_size * channels,# 小单元内的数字元素个数
            ),
        )
        return patches
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size}) # 更新变量到配置
        return config

# 图像块编码层
class PatchEncoder(layers.Layer): # 单元编码
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches # 144
        self.projection = layers.Dense(units=projection_dim) # 投影到64维，(1,144,64)
        # 把单元按顺序编号,为每个位置嵌入64维向量表示这个位置，形状（1,144,64）
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch) # (n,np,d) 投影
        # 这样对应的图像块就都添加了位置信息，位置和图像块是对应相加的
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})#更新参数,单元数
        return config

# 构建 ViT 模型
# def create_vit_classifier(data_augmentation):
#     inputs = keras.Input(shape=input_shape) # (72,72,3)
#     # Augment data.
#     augmented = data_augmentation(inputs) # 增强
#     # Create patches.
#     patches = Patches(patch_size)(augmented) # 划分成小单元,(1,144,108)
#     # Encode patches.
#     encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) # 编码单元(1,144,64)
#     # 创建多层transformer块
#     for _ in range(transformer_layers):
#         # 层标准化
#         x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         # Create a multi-head attention layer.
#         attention_output = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=projection_dim, dropout=0.1
#         )(x1, x1)
#         # 将自注意力前后的图片数据做残差连接
#         x2 = layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2.
#         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#         # MLP.
#         x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
#         # 将感知机前后的图片数据做残差,有利于模型学习恒等映射,减少梯度问题
#         encoded_patches = layers.Add()([x3, x2]) # 这里重置了encoded_patches,便于循环编码
#     # Create a [batch_size, projection_dim] tensor.
#     representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#     representation = layers.Flatten()(representation)#扁平化
#     representation = layers.Dropout(0.5)(representation)
#     # Add MLP.
#     features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
#     # Classify outputs.
#     logits = layers.Dense(num_classes)(features)
#     # Create the Keras model.
#     model = keras.Model(inputs=inputs, outputs=logits)
#     return model