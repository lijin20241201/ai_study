import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mlp_head_units = [
    512,
    128
] 

def activation_block(x): # 一般xception是 先批次标准化再激活函数
    x = layers.BatchNormalization()(x) # 批次标准化
    return layers.Activation("relu")(x)

def mlp(x, hidden_units, dropout_rate): # 线性转换,也叫多层感知机
    for units in hidden_units:
        x = layers.Dense(units,activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
# 用深度卷积提取图片空间特征和下采样,用两种逐点卷积(扩张和压缩)来扩张通道以进行
# 深度卷积提取特征,用压缩通道的逐点卷积来残差连接,让网络学习扩张--深度卷积--
# 压缩前后的变化,关于特征图,对28x28的特征图和7x7的特征图,用两个残差连接
# 对56x56的特征图用一个残差连接,对14x14的特征图网络用了5个残差连接,
# 压缩点卷积通道变化 16--24--32--64--96--160--320
# 扩张点卷积通道变化 96--144--192--384--576--960--1280
# 网络遵循扩张点卷积--深度卷积--压缩点卷积的模式,利用深度卷积来提取特征,缩小
# 特征图大小,利用两个连续的通道相同的压缩点卷积来残差连接
def make_mobilenetv2_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    # x=layers.Rescaling(1./127.5, offset=-1)(inputs)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Conv2D(32,3,strides=2,padding='same',use_bias=False)(x) # (112,112)
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # 收缩和扩张点卷积通道都在不断增加,目的为了提取更加抽象的特征
    # 注意一点扩张点卷积后跟批次标准化和激活函数,收缩点卷积后只有批次标准化
    # 深度卷积都是为了提取特征图空间信息,点卷积是为了混合通道间信息
    # 而且点卷积有点类似线性投影层,做通道切换
    x=layers.Conv2D(16,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(96,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (56,56)
    x=activation_block(x)
    x=layers.Conv2D(24,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x0=x # 用一个残差提取56x56特征图的特征 24--144--24
    x=layers.Conv2D(144,1,padding='same',use_bias=False)(x0)
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.Conv2D(24,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x=layers.add([x,x0])
    x=layers.Conv2D(144,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (28,28)
    x=activation_block(x)
    x=layers.Conv2D(32,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    for i in range(2): # 用两个残差连接提取28x28的特征 32--192--32
        x0=x
        x=layers.Conv2D(192,1,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.Conv2D(32,1,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
    x=layers.Conv2D(192,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # x=layers.ZeroPadding2D(padding=(1,1))(x)
    # 用DepthwiseConv2D去缩小特征图大小,(14,14)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.Conv2D(64,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    for i in range(3): # 用三个残差连接提取14x14特征图的特征 64--384--64
        x0=x
        x=layers.Conv2D(384,1,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.Conv2D(64,1,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
    # 这里一般是缩小特征图或者加大扩张和压缩通道的代码
    x=layers.Conv2D(384,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.Conv2D(96,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    for i in range(2): # 再次用两个残差连接提取14x14特征图的特征 96--576--96
        x0=x
        x=layers.Conv2D(576,1,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.Conv2D(96,1,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
    x=layers.Conv2D(576,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # x=layers.ZeroPadding2D(padding=(1,1))(x)
    # 用DepthwiseConv2D来缩小特征图大小 (7,7)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.Conv2D(160,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    for i in range(2): # 用两个残差连接提取7x7特征图的特征 160--960--160
        x0=x
        # 之所以传人x0,是因为有助于理解,在这里模型出现分叉,残差连接后合并分叉
        x=layers.Conv2D(960,1,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.Conv2D(160,1,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
    x=layers.Conv2D(960,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.Conv2D(320,1,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(1280,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    # x = layers.GlobalAveragePooling2D()(x) 
    # outputs = layers.Dense(num_classes)(x)
    return keras.Model(inputs,x)

def make_residual_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # "Entry block" 通常指的是网络开始时的第一个处理模块
    x = layers.Rescaling(1.0 / 255)(inputs) # 标准化数据到0-1之间
    # 卷积层将输入数据的特征深度（也称为通道数）从卷积前的数量增加到了128。这意味着每个位置（在高度和宽度上）现在都有128个特征值，
    # 而不是卷积前的数量。这些特征值是通过128个不同的3x3卷积核计算得到的，每个卷积核都提取了输入数据的一种特定特征
    # 通过卷积操作，网络能够学习到输入数据的局部特征。这些特征是通过卷积核在输入数据上滑动并计算点积来获得的。由于使用了
    # 个卷积核，网络能够学习到多种不同的特征表示
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x) # (90,90)
    x = activation_block(x)
    for size in [128,256,384]:
        residual = x # 残差前段,每次都被设置为新的x
        x = layers.SeparableConv2D(size, 3, padding="same")(residual)
        x = activation_block(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x) # (45,45)
        # 用1x1的逐点卷积,参数是:1*1*64*128,有利于改变通道大小,混合通道信息,步长2
        # 有利于缩小特征图大小
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(residual) # (45,45)
        # 残差后段批次标准化后最大池化,这里也批次标准化,能数据一致
        residual = layers.BatchNormalization()(residual)
        # 残差连接,有利于网络学习恒等映射,学习残差前后差异
        x = layers.add([x, residual]) 
        # x和residual不断变化,最后的x就是经过一系列可分离卷积(包括深度卷积和逐点卷积)和下采样后的x
        # 深度卷积捕捉图片尺寸大小内的空间信息,逐点卷积混合各个通道信息,它们分工合作,而普通卷积
        # 既要捕捉空间信息,同时又要混合通道信息
        x=layers.Activation("relu")(x)
    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = activation_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    
    x = layers.Dropout(0.3)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

# Xception小型版
def make_xception_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # "Entry block" 通常指的是网络开始时的第一个处理模块
    x = layers.Rescaling(1.0 / 255)(inputs) # 标准化数据到0-1之间
    # 卷积层将输入数据的特征深度（也称为通道数）从卷积前的数量增加到了128。这意味着每个位置（在高度和宽度上）现在都有128个特征值，
    # 而不是卷积前的数量。这些特征值是通过128个不同的3x3卷积核计算得到的，每个卷积核都提取了输入数据的一种特定特征
    # 通过卷积操作，网络能够学习到输入数据的局部特征。这些特征是通过卷积核在输入数据上滑动并计算点积来获得的。由于使用了
    # 个卷积核，网络能够学习到多种不同的特征表示
    x = layers.Conv2D(32, 3, strides=2, padding="same",use_bias=True)(x) 
    x = activation_block(x)
    x = layers.Conv2D(64, 3,use_bias=True)(x) 
    x = activation_block(x)
    for size in [128,256,512]:
        residual = x  
        x = layers.SeparableConv2D(size, 3, padding="same",use_bias=True)(residual)
        x =activation_block(x)
        x = layers.SeparableConv2D(size, 3, padding="same",use_bias=True)(x)
        # 经过第二个sepconv后,只bn,不用激活函数,因为后面还要最大池化
        x =layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(padding="same")(x) # (38,38)
        residual = layers.Conv2D(size, 1, strides=2, padding="same",use_bias=True)(
            residual
        )
        # 上面的是为了让残差块尺寸相同,这个是要做批次标准化,不做激活函数处理是因为maxpooling,
        # 还有残差两部分处理方式要一样,因为如果做relu激活,maxpool取2x2中的最大就不准了
        residual =layers.BatchNormalization()(residual)
        # x 被更新为 x 和 residual 的元素级相加的结果。此时，x 指向一个新的张量，
        # 该张量是原始 x 和 residual 相加的结果
        x = layers.add([x,residual])
        x = layers.Activation("relu")(x)
    for _ in range(3):
        # residual 和 x 最初指向内存中的同一个对象，但随后对 x 的修改（通过层操作）不会改变 residual
        # 指向的原始对象的内容，因为 x 被赋予了新的、经过层处理的张量的引用。
        residual = x 
        x = layers.SeparableConv2D(512, 3, padding="same",use_bias=True)(residual)
        x =activation_block(x)
        x = layers.SeparableConv2D(512, 3, padding="same",use_bias=True)(x)
        x =activation_block(x)
        x = layers.SeparableConv2D(512, 3, padding="same",use_bias=True)(x)
        x =layers.BatchNormalization()(x)
        x = layers.add([x,residual])
    x = layers.Activation("relu")(x)
    residual = x 
    x = layers.SeparableConv2D(512, 3, padding="same",use_bias=True)(residual)
    x =activation_block(x)
    x = layers.SeparableConv2D(768, 3, padding="same",use_bias=True)(x)
    # 经过第二个sepconv后,只bn,不用激活函数,因为后面还要最大池化
    x =layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(padding="same")(x) #(45,45)
    residual = layers.Conv2D(768, 1, strides=2, padding="same",use_bias=True)(
        residual
    )
    residual =layers.BatchNormalization()(residual)
    x = layers.add([x,residual]) 
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(1024, 3, padding="same",use_bias=True)(x)
    x = activation_block(x)
    x = layers.SeparableConv2D(1024, 3, padding="same",use_bias=True)(x)
    x = activation_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    # x=mlp(x,mlp_head_units,0.2)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

def make_vgg19_model(input_shape,num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    # for size in [64,128,256,512,512]:
    #     if size in [64,128]:
    #         for i in range(2):
    #             x=layers.Conv2D(size,3,padding='same',activation='relu')(x)
    #         x=layers.MaxPooling2D()(x) 
    #     else:
    #         for i in range(4):
    #             x=layers.Conv2D(size,3,padding='same',activation='relu')(x)
    #         x=layers.MaxPooling2D()(x) 
    conv_blocks = [(64, 2),(128, 2), (256, 4), (512, 4), (512, 4)] 
    for size, num_conv_layers in conv_blocks:  
        for _ in range(num_conv_layers):  
            x = layers.Conv2D(size, 3, padding='same', activation='relu')(x)  
        x = layers.MaxPooling2D()(x)  
    # x=layers.GlobalAveragePooling2D()(x) 
    # x = layers.Dropout(0.3)(x)
    # outputs = layers.Dense(num_classes,activation="softmax")(x)
    return keras.Model(inputs,x)

def make_resnet_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x=layers.Conv2D(64,7,strides=2,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.MaxPooling2D(padding='same')(x)
    for i in range(3):
        x0=x
        x=layers.SeparableConv2D(64,3,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.SeparableConv2D(64,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
        x=layers.Activation('relu')(x)
    x0=x
    x=layers.SeparableConv2D(128,3,strides=2,padding='same',use_bias=False)(x0)
    x=activation_block(x)
    x=layers.SeparableConv2D(128,3,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x0=layers.Conv2D(128,1,strides=2,padding='same',use_bias=False)(x0)
    x0=layers.BatchNormalization()(x0)
    x=layers.add([x,x0]) 
    x=layers.Activation('relu')(x)
    for i in range(3):
        x0=x
        x=layers.SeparableConv2D(128,3,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.SeparableConv2D(128,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
        x=layers.Activation('relu')(x) 
    x0=x
    x=layers.SeparableConv2D(256,3,strides=2,padding='same',use_bias=False)(x0)
    x=activation_block(x)
    x=layers.SeparableConv2D(256,3,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x0=layers.Conv2D(256,1,strides=2,padding='same',use_bias=False)(x0)
    x0=layers.BatchNormalization()(x0)
    x=layers.add([x,x0]) 
    x=layers.Activation('relu')(x)
    for i in range(5):
        x0=x
        x=layers.SeparableConv2D(256,3,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.SeparableConv2D(256,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
        x=layers.Activation('relu')(x) 
    x0=x
    x=layers.SeparableConv2D(512,3,strides=2,padding='same',use_bias=False)(x0)
    x=activation_block(x)
    x=layers.SeparableConv2D(512,3,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x0=layers.Conv2D(512,1,strides=2,padding='same',use_bias=False)(x0)
    x0=layers.BatchNormalization()(x0)
    x=layers.add([x,x0]) 
    x=layers.Activation('relu')(x)
    for i in range(2):
        x0=x
        x=layers.SeparableConv2D(512,3,padding='same',use_bias=False)(x0)
        x=activation_block(x)
        x=layers.SeparableConv2D(512,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0]) 
        x=layers.Activation('relu')(x)
    x = layers.SeparableConv2D(512, 3, padding="same",use_bias=False)(x)
    x = activation_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    
    x = layers.Dropout(0.3)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)