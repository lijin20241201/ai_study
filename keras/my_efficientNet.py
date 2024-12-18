import os
# os.environ["KERAS_BACKEND"] = "tensorflow" 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def silu_activation_block(x):# 激活块
    x = layers.BatchNormalization()(x) # 批次标准化
    return layers.Activation(keras.activations.silu)(x)

def conv_block(inputs,fiters,kernel_size=3,strides=1,dropout=None):
    x=layers.Conv2D(fiters,kernel_size,strides=strides,padding='same',use_bias=False)(inputs) 
    x=silu_activation_block(x)
    if dropout:
        x=layers.Dropout(dropout)(x)
    return x

def point_wise_conv_block(inputs,fiters,strides=1,dropout=None,use_act=False):
    x=layers.Conv2D(fiters,1,strides=strides,padding='same',use_bias=False)(inputs) 
    x=layers.BatchNormalization()(x)
    if use_act:
        x=layers.Activation(keras.activations.silu)(x)
    if dropout:
        x=layers.Dropout(dropout)(x)
    return x

def depthwiseConv_block(inputs,filters,kernel_size=3,strides=1):
    x=inputs
    # 逐点卷积
    x=layers.Conv2D(filters,1,padding='same',use_bias=False)(x) 
    x=silu_activation_block(x)
    # 深度卷积
    x=layers.DepthwiseConv2D(kernel_size,strides=strides,padding='same',use_bias=False)(x) 
    x=silu_activation_block(x)
    return x

def se(inputs,in_c,out_c):
    multiply=inputs
    # (n,c) 全局平均池化,获取样本的全局分类信息
    x=layers.GlobalAveragePooling2D()(multiply)
    # 变形
    x=layers.Reshape([1,1,-1])(x)
    # 这里用了截距,这里减少通道数是为了紧凑特征,同时可以减少模型过拟合
    x=layers.Conv2D(in_c,1,padding='same',activation=keras.activations.silu)(x) 
    # 之后放大到原通道数,这里卷积核大小是1,是点卷积,sigmoid会给每个通道打分
    # 分数表示每个通道的重要程度
    x=layers.Conv2D(out_c,1,padding='same',activation=keras.activations.sigmoid)(x) 
    # 对通道重定位
    x=layers.Multiply()([x,multiply])
    return x
mlp_head_units = [
    512,
    128
] 
def mlp(x, hidden_units, dropout_rate): # 线性转换,也叫多层感知机
    for units in hidden_units:
        x = layers.Dense(units,activation=keras.activations.silu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def get_efficientNetV2B3_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Normalization()(x)
    x=conv_block(x,24,3,2) # (112,112),下采样
    x=conv_block(x,16,3)
    x0=x
    x=conv_block(x0,16,3,dropout=(1/315)*3)
    x=layers.add([x,x0])
    x=conv_block(x,64,3,2) # (56,56),下采样
    x=point_wise_conv_block(x,40)
    for i in range(2):
        x0=x
        x=conv_block(x0,160,3)
        x=point_wise_conv_block(x,40,dropout=(1/315)*9+i*(1/315))
        x=layers.add([x,x0])
    x=conv_block(x,160,3,2) # (28,28),下采样
    x=point_wise_conv_block(x,56)
    for i in range(2):
        x0=x
        x=x=conv_block(x0,224,3)
        x=point_wise_conv_block(x,56,dropout=(1/315)*18+i*(1/315))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,224,strides=2) # (14,14),下采样
    x=se(x,14,224)
    x=point_wise_conv_block(x,112)
    for i in range(4):
        x0=x
        x=depthwiseConv_block(x0,448,strides=1)
        x=se(x,28,448)
        x=point_wise_conv_block(x,112,dropout=(1/315)*27+i*(1/315))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,672,strides=1)
    x=se(x,28,672)
    x=point_wise_conv_block(x,136)
    for i in range(6): # (14,14)的特征图一共做了10次残差
        x0=x
        x=depthwiseConv_block(x0,816,strides=1)
        x=se(x,34,816)
        x=point_wise_conv_block(x,136,dropout=(1/315)*42+i*(1/315))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,816,strides=2)  # (7,7)
    x=se(x,34,816)
    x=point_wise_conv_block(x,232)
    for i in range(11): # (7,7)的特征图一共做了11次残差
        x0=x
        x=depthwiseConv_block(x0,1392,strides=1)
        x=se(x,58,1392)
        x=point_wise_conv_block(x,232,dropout=(1/315)*63+i*(1/315))
        x=layers.add([x,x0])
    x=point_wise_conv_block(x,1536,use_act=True)
    return keras.Model(inputs,x)

def get_my_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=conv_block(x,64,3,2) # (16,16)
    x=depthwiseConv_block(x,196) 
    x=se(x,14,196)
    x=point_wise_conv_block(x,112)
    for i in range(3):
        x0=x
        x=depthwiseConv_block(x0,384,strides=1)
        x=se(x,20,384)
        x=point_wise_conv_block(x,112,dropout=(1/315)*27+i*(1/315))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,384)
    x=se(x,20,384)
    x=point_wise_conv_block(x,136)
    for i in range(3): # (14,14)的特征图一共做了10次残差
        x0=x
        x=depthwiseConv_block(x0,512,strides=1)
        x=se(x,32,512)
        x=point_wise_conv_block(x,136,dropout=(1/315)*42+i*(1/315))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,512,strides=2)  # (8,8)
    x=se(x,32,512)
    x=point_wise_conv_block(x,196)
    for i in range(3): # (7,7)的特征图一共做了11次残差
        x0=x
        x=depthwiseConv_block(x0,768,strides=1)
        x=se(x,48,768)
        x=point_wise_conv_block(x,196,dropout=(1/315)*63+i*(1/315))
        x=layers.add([x,x0])
    x=point_wise_conv_block(x,1024,use_act=True)
    x=layers.GlobalAveragePooling2D()(x)
    x=mlp(x,mlp_head_units,0.25)
    outputs=layers.Dense(num_classes,activation='softmax')(x)
    return keras.Model(inputs,outputs)