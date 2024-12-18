import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras import layers

# 一定要在编译模型前加载模型权重,否则会报优化器参数不一致,肯定不一致,都没保存优化器参数
# 优化器参数无关紧要,因为下次肯定调整新的学习率
# MobileNetV2 的一个关键特性是使用了“扩张-压缩”模式，即首先通过 1x1 卷积（点卷积）增加通道数（扩张），
# 然后通过深度可分离卷积处理特征，最后再通过 1x1 卷积减少通道数（压缩）,越往下,扩张和压缩通道越大
# 并且在特征图到60x60以下逐渐加大提取特征力度,用残差网络
# MobileNetV2 中的瓶颈块会根据网络的深度重复不同的次数。您的代码中为不同的瓶颈块设置了不同的重复次数（
# 如 2 次、3 次），这通常是正确的
def relu_activation_block(x): # 大多预训练模型都是先批次标准化,再激活函数
    # 如果把激活函数放前面,在模型摘要里会先显示激活函数,但是这不是大多数模型的摘要信息
    # 说明是先批次标准化,之后激活函数
    # relu6:如果输入 x 是正数，则输出 x，但不超过6；如果 x 是负数，则输出0；如果 x 大于6，
    # 则输出6.而relu无限制
    x = layers.BatchNormalization()(x) 
    return layers.Activation('relu6')(x)
def activation_block(x): # 大多预训练模型都是先批次标准化,再激活函数
    # 如果把激活函数放前面,在模型摘要里会先显示激活函数,但是这不是大多数模型的摘要信息
    # 说明是先批次标准化,之后激活函数
    x = layers.BatchNormalization()(x) 
    return layers.Activation(keras.activations.hard_swish)(x)

def se(inputs,in_c,out_c):
    multiply=inputs
    # (n,c) 全局平均池化,获取样本的全局分类信息
    x=layers.GlobalAveragePooling2D()(multiply)
    # 变形
    x=layers.Reshape([1,1,-1])(x)
    # 这里用了截距,这里减少通道数是为了紧凑特征,同时可以减少模型过拟合
    x=layers.Conv2D(in_c,1,padding='same')(x) 
    # 之后放大到原通道数,这里卷积核大小是1,是点卷积
    x=layers.Conv2D(out_c,1,padding='same')(x) 
    x=layers.Multiply()([x,multiply])
    return x

def depthwiseConv_block(inputs,filters,kernel_size=3,strides=1,is_relu=True):
    x=inputs
    x=layers.Conv2D(filters,1,padding='same',use_bias=False)(x) 
    if is_relu:
        x=relu_activation_block(x)
    else:
        x=activation_block(x)
    x=layers.DepthwiseConv2D(kernel_size,strides=strides,padding='same',use_bias=False)(x) 
    if is_relu:
        x=relu_activation_block(x)
    else:
        x=activation_block(x)
    return x

def conv1_block(inputs,filters,dropout=None,use_noise=True):
    x=inputs
    x=layers.Conv2D(filters,1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # Dropout被设置为0的单元的比例由rate参数控制。未被设置为0的单元会被缩放以
    # 保持整体输入的期望值不变。
    # noise_shape: 1维整数张量，代表将与输入相乘的二进制dropout掩码的形状。这允许你对dropout
    # 的应用进行更细粒度的控制。例如，如果你的输入形状为(batch_size, timesteps, features)，
    # 而你希望对于所有时间步长使用相同的dropout掩码，你可以设置
    # noise_shape=(batch_size, 1, features)
    # seed: 一个Python整数，用作随机数生成的种子。这有助于实验的可重复性。
    # 调用Dropout:inputs: 输入张量（可以是任意阶的）。
    # training: 一个Python布尔值，指示层是否应以训练模式（添加dropout）或推理模式（不执行任何操作）
    # 运行。当使用model.fit时，training参数会自动设置为True。在其他情况下，你可能需要在调用层时显式
    # 设置此参数。
    if dropout:
        if use_noise:
            x=layers.Dropout(dropout,noise_shape=(None,1,1,1))(x)
        else:
           x=layers.Dropout(dropout)(x) 
    return x

def get_efficientNetV2M_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/127.5,offset=-1)(inputs)
    x=layers.Conv2D(24,3,strides=2,padding='same',use_bias=False)(x) # (112,112)
    x=relu_activation_block(x)
    # 在(112,112)特征图上残差连接3次,用的普通卷积
    for i in range(3): 
        x0=x
        x=layers.Conv2D(24,3,padding='same',use_bias=False)(x0)
        x=relu_activation_block(x)
        if i != 0:
            x=layers.Dropout(0.003508*i,noise_shape=(None,1,1,1))(x)
        x=layers.add([x,x0])
    x=layers.Conv2D(96,3,strides=2,padding='same',use_bias=False)(x) # (56,56)
    x=relu_activation_block(x)
    x=conv1_block(x,48)
    # 在(56,56)特征图上残差连接4次,用的普通卷积和逐点卷积 
    for i in range(4): 
        x0=x
        x=layers.Conv2D(192,3,padding='same',use_bias=False)(x0)
        x=relu_activation_block(x)
        x=conv1_block(x,48,0.003508*(4+i))
        x=layers.add([x,x0])
    x=layers.Conv2D(192,3,strides=2,padding='same',use_bias=False)(x) # (28,28)
    x=relu_activation_block(x)
    x=conv1_block(x,80)
    # 在(28,28) 特征图上残差连接4次,用的普通卷积和逐点卷积,点卷积用来切换通道 
    for i in range(4):
        x0=x
        x=layers.Conv2D(320,3,padding='same',use_bias=False)(x0)
        x=relu_activation_block(x)
        x=conv1_block(x,80,0.003508*(9+i))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,320,strides=2,kernel_size=3) # (14,14)
    x=se(x,20,320)
    x=conv1_block(x,160)
    # 用se全局注意力,深度卷积和逐点卷积在(14,14)的特征图上狠提特征,一共残差19次
    # dropout会随层数增加递增
    for i in range(6):
        x0=x
        x=depthwiseConv_block(x0,640)
        x=se(x,40,640)
        x=conv1_block(x,160,0.003508*(14+i))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,960,is_relu=False)
    x=se(x,40,960)
    x=conv1_block(x,176)
    for i in range(13): 
        x0=x
        x=depthwiseConv_block(x0,1056,is_relu=False)
        x=se(x,44,1056)
        x=conv1_block(x,176,0.003508*(21+i))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,1056,strides=2,kernel_size=3,is_relu=False) # (7,7)
    x=se(x,44,1056)
    x=conv1_block(x,304)
    # 在7x7的特征图上一共残差21次
    for i in range(17):
        x0=x
        x=depthwiseConv_block(x0,1824,is_relu=False)
        x=se(x,76,1824)
        x=conv1_block(x,304,0.003508*(35+i))
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,1824,is_relu=False)
    x=se(x,76,1824)
    x=conv1_block(x,512)
    for i in range(4):
        x0=x
        x=depthwiseConv_block(x0,3072,is_relu=False)
        x=se(x,128,3072)
        x=conv1_block(x,512,0.003508*(53+i))
        x=layers.add([x,x0])
    x=layers.Conv2D(1280,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    return keras.Model(inputs,x)

# 收缩逐点卷积:48--80--160--224--384--640
# 扩张逐点卷积：288--480--960--1344--2304--3840
# global注意力模块收缩逐点卷积:12--20--40--56--96--160
def get_efficientnetb7_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Normalization()(x)
    x=layers.Rescaling(scale=[2.0896918976428642, 2.1128856368212916, 2.1081851067789197], offset=0.0)(x) 
    x=layers.Conv2D(64,3,strides=2,padding='same',use_bias=False)(x) # (112,112)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=se(x,16,64)
    x=conv1_block(x,32)
    # 其实想想都知道它加深网络干啥事,之前200多层的网络主要提取14x14,7x7这些特征图的特征
    # 现在它有足够的层数,就加大了112x112,56x56特征图的提取力度
    for i in range(3): # 在 112x112的特征图上用了3次残差
        x0=x
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x0)
        x=relu_activation_block(x)
        x=se(x,8,32)
        x=conv1_block(x,32,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,192,strides=2) # (56,56)
    x=se(x,8,192)
    x=conv1_block(x,48)
    for i in range(6): # 在56x56的特征图上用了6次残差
        x0=x
        x=depthwiseConv_block(x0,288)
        x=se(x,12,288)
        x=conv1_block(x,48,0.025)
        x=layers.add([x,x0])
    # 用5x5的核,可以加强感受野
    x=depthwiseConv_block(x,288,strides=2,kernel_size=5) # (28,28)
    x=se(x,12,288)
    x=conv1_block(x,80)
    for i in range(6): # 在28x28的特征图上用了6次残差
        x0=x
        x=depthwiseConv_block(x0,480,kernel_size=5)
        x=se(x,20,480)
        x=conv1_block(x,80,0.025)
        x=layers.add([x,x0])
    # 用3x3的核
    x=depthwiseConv_block(x,480,kernel_size=3,strides=2) # (14,14)
    x=se(x,20,480)
    x=conv1_block(x,160)
    for i in range(9): 
        x0=x
        x=depthwiseConv_block(x0,960)
        x=se(x,40,960)
        x=conv1_block(x,160,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,960,kernel_size=5)
    x=se(x,40,960)
    x=conv1_block(x,224)
    for i in range(9): # 在14x14的特征图上一共用了18次残差
        x0=x
        x=depthwiseConv_block(x0,1344,kernel_size=5)
        x=se(x,56,1344)
        x=conv1_block(x,224,0.025)
        x=layers.add([x,x0])
    # 深度卷积核大小:5x5
    x=depthwiseConv_block(x,1344,kernel_size=5,strides=2) # (7,7)
    x=se(x,56,1344)
    x=conv1_block(x,384)
    for i in range(12):
        x0=x
        x=depthwiseConv_block(x0,2304,kernel_size=5)  # 核大小:5x5
        x=se(x,96,2304)
        x=conv1_block(x,384,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,2304,kernel_size=3)
    x=se(x,96,2304)
    x=conv1_block(x,640)
    for i in range(3): # 在7x7的特征图上一共用了15次残差
        x0=x
        x=depthwiseConv_block(x0,3840)
        x=se(x,160,3840)
        x=conv1_block(x,640,0.025)
        x=layers.add([x,x0])
    x=layers.Conv2D(2560,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    return keras.Model(inputs,x)

# 收缩点卷积: 24-40--64--128--176--304--512
# 扩张点卷积: 144--240--384--768--1056--1824--3072
# global注意力模块: 6--10--16--32--44--76--128
def get_efficientnetb5_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Normalization()(x)
    x=layers.Rescaling(scale=[2.0896918976428642, 2.1128856368212916, 2.1081851067789197], offset=0.0)(x) 
    x=layers.Conv2D(48,3,strides=2,padding='same',use_bias=False)(x) # (112,112)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=se(x,12,48)
    x=conv1_block(x,24)
    # 其实想想都知道它加深网络干啥事,之前200多层的网络主要提取14x14,7x7这些特征图的特征
    # 现在它有足够的层数,就加大了112x112,56x56特征图的提取力度
    for i in range(2): # 在112x112的特征图上用了两次残差
        x0=x
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x0)
        x=relu_activation_block(x)
        x=se(x,6,24)
        x=conv1_block(x,24,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,144,strides=2) # (56,56)
    x=se(x,6,144)
    x=conv1_block(x,40)
    for i in range(4): # 在56x56的特征图上用了四次残差
        x0=x
        x=depthwiseConv_block(x0,240)
        x=se(x,10,240)
        x=conv1_block(x,40,0.025)
        x=layers.add([x,x0])
    # 用5x5的核,可以加强感受野
    x=depthwiseConv_block(x,240,strides=2,kernel_size=5) # (28,28)
    x=se(x,10,240)
    x=conv1_block(x,64)
    for i in range(4): # 在28x28的特征图上用了四次残差
        x0=x
        x=depthwiseConv_block(x0,384,kernel_size=5)
        x=se(x,16,384)
        x=conv1_block(x,64,0.025)
        x=layers.add([x,x0])
    # 用3x3的核
    x=depthwiseConv_block(x,384,kernel_size=3,strides=2) # (14,14)
    x=se(x,16,384)
    x=conv1_block(x,128)
    for i in range(6): 
        x0=x
        x=depthwiseConv_block(x0,768)
        x=se(x,32,768)
        x=conv1_block(x,128,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,768,kernel_size=5)
    x=se(x,32,768)
    x=conv1_block(x,176)
    for i in range(6): # 在14x14的特征图上一共用了12次残差
        x0=x
        x=depthwiseConv_block(x0,1056,kernel_size=5)
        x=se(x,44,1056)
        x=conv1_block(x,176,0.025)
        x=layers.add([x,x0])
    # 深度卷积核大小:5x5
    x=depthwiseConv_block(x,1056,kernel_size=5,strides=2) # (7,7)
    x=se(x,44,1056)
    x=conv1_block(x,304)
    for i in range(8):
        x0=x
        x=depthwiseConv_block(x0,1824,kernel_size=5)  # 核大小:5x5
        x=se(x,76,1824)
        x=conv1_block(x,304,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,1824,kernel_size=3)
    x=se(x,76,1824)
    x=conv1_block(x,512)
    for i in range(2): # 在 7x7的特征图上一共用了10次残差
        x0=x
        x=depthwiseConv_block(x0,3072)
        x=se(x,128,3072)
        x=conv1_block(x,512,0.025)
        x=layers.add([x,x0])
    x=layers.Conv2D(2048,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    return keras.Model(inputs,x)

# 收缩点卷积:16--24--40--80--112--192
# 扩张点卷积:96--144--240--480--672--1152
# global注意力模块:4--6--10--20--28--48
def get_efficientnetb0_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Normalization()(x)
    x=layers.Rescaling(scale=[2.0896918976428642, 2.1128856368212916, 2.1081851067789197], offset=0.0)(x) 
    x=layers.Conv2D(32,3,strides=2,padding='same',use_bias=False)(x) # (112,112)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=se(x,8,32) # 有利于通道信息的重修订,减少过拟合,只关注重要的特征
    x=conv1_block(x,16)
    # 深度卷积模块,用来提取空间信息
    x=depthwiseConv_block(x,96,strides=2)  # (56,56)
    x=se(x,4,96)
    x=conv1_block(x,24) 
    x0=x
    x=depthwiseConv_block(x0,144) 
    x=se(x,6,144)
    x=conv1_block(x,24,0.025)
    x=layers.add([x,x0])
    # 注意:kernel_size=5,核大,视野就大
    x=depthwiseConv_block(x,144,kernel_size=5,strides=2) # (28,28)
    x=se(x,6,144)
    x=conv1_block(x,40)
    x0=x
    x=depthwiseConv_block(x0,240,kernel_size=5)
    x=se(x,10,240)
    x=conv1_block(x,40,0.025)
    x=layers.add([x,x0])
    # 这个位置用的kernel_size=3
    x=depthwiseConv_block(x,240,strides=2) # (14,14)
    x=se(x,10,240)
    x=conv1_block(x,80)
    for i in range(2):
        x0=x
        x=depthwiseConv_block(x0,480)
        x=se(x,20,480)
        x=conv1_block(x,80,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,480,kernel_size=5)
    x=se(x,20,480)
    x=conv1_block(x,112)
    for i in range(2):
        x0=x
        x=depthwiseConv_block(x0,672,kernel_size=5)
        x=se(x,28,672)
        x=conv1_block(x,112,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,672,kernel_size=5,strides=2) # (7,7)
    x=se(x,28,672)
    x=conv1_block(x,192)
    for i in range(3):
        x0=x
        x=depthwiseConv_block(x0,1152,kernel_size=5)
        x=se(x,48,1152)
        x=conv1_block(x,192,0.025)
        x=layers.add([x,x0])
    x=depthwiseConv_block(x,1152,kernel_size=3)
    x=se(x,48,1152)
    x=conv1_block(x,320)
    x=layers.Conv2D(1280,1,padding='same',use_bias=False)(x)
    x=activation_block(x)
    return keras.Model(inputs,x)

# 64--128--256--512--1024
# 没有残差,点卷积用来切换通道,便于深度卷积提取信息
def get_mobilenet_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Conv2D(32,3,strides=2,padding='same',use_bias=False)(x)  # (80,80,32)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.Conv2D(64,1,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (40,40,64)
    x=relu_activation_block(x)
    for i in range(2): # 在40x40的特征图上深度卷积两次
        x=layers.Conv2D(128,1,padding='same',use_bias=False)(x)
        x=relu_activation_block(x)
        if i==0:
            x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) 
        else:
           x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (20,20,128) 
        x=relu_activation_block(x)
    for i in range(2): # 在20x20的特征图上深度卷积两次
        x=layers.Conv2D(256,1,padding='same',use_bias=False)(x)
        x=relu_activation_block(x)
        if i ==0:
            x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) 
        else:
            x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (10,10,256)
        x=relu_activation_block(x)
    for i in range(6): # 在10x10的特征图上狠提特征
        x=layers.Conv2D(512,1,padding='same',use_bias=False)(x)
        x=relu_activation_block(x)
        if i !=5:
            x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) 
        else:
            x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x)  # (5,5,512)
        x=relu_activation_block(x)
    x=layers.Conv2D(1024,1,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.Conv2D(1024,1,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    return keras.Model(inputs,x)

# xception,在14x14的特征图上狠提取特征
def get_xception_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Conv2D(32,3,strides=2,use_bias=False)(x) # (112,112)
    x=activation_block(x)
    x=layers.Conv2D(64,3,use_bias=False)(x) # (109,109)
    x=activation_block(x)
    # filters 参数（也称为卷积核的数量或输出空间的维度）
    # 缩小特征图大小,用残差连接
    for filters in (128,256,728): # (55,55),(28,28),(14,14)
        x0=x
        if filters !=128:
            x=layers.Activation('relu')(x0)
        x=layers.SeparableConv2D(filters,3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.SeparableConv2D(filters,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
        x0=layers.Conv2D(filters,1,strides=2,padding='same',use_bias=False)(x0)
        x0=layers.BatchNormalization()(x0)
        x=layers.add([x,x0]) # 残差连接
    for i in range(8): # 用残差连接在(14,14)的特征图上提特征
        x0=x
        x=layers.Activation('relu')(x0)
        x=layers.SeparableConv2D(728,3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.SeparableConv2D(728,3,padding='same',use_bias=False)(x)
        x=activation_block(x)
        x=layers.SeparableConv2D(728,3,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0])
    x0=x # 残差连接缩小特征图到(7,7)
    x=layers.Activation('relu')(x0)
    x=layers.SeparableConv2D(728,3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.SeparableConv2D(1024,3,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D(pool_size=2,strides=2,padding='same')(x)
    x0=layers.Conv2D(1024,1,strides=2,padding='same',use_bias=False)(x0)
    x0=layers.BatchNormalization()(x0)
    x=layers.add([x,x0])
    x=layers.SeparableConv2D(1536,3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    x=layers.SeparableConv2D(2048,3,padding='same',use_bias=False)(x)
    x=activation_block(x)
    return keras.Model(inputs,x)

# 提高模型泛化能力：通过在网络的不同部分使用不同的激活函数（如前半段使用ReLU，后半段使用h-swish），
# 可以增加模型的灵活性，使其能够更好地适应不同的数据分布和任务需求，从而提高模型的泛化能力。
# 在网络的前半部分，ReLU因其简单性和计算效率而被广泛使用。然而，随着网络深度的增加，ReLU可能会遇到梯度消失的问题
# （即“死亡ReLU”问题）。因此，在更深的层中，可能会看到其他类型的激活函数，如LeakyReLU、PReLU（Parametric 
# ReLU）或ELU（Exponential Linear Unit）等，这些激活函数旨在缓解梯度消失的问题。
# 压缩点卷积:16--24--40--80--112--160
# 扩张点卷积:64--72--120--240--480--672--960
# 全局平均池化注意力加权
def get_mobileNetV3_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x=layers.Rescaling(1.0/255)(inputs)
    x=layers.Conv2D(16,3,2,padding='same',use_bias=False)(x) # (112,112,3)
    x=relu_activation_block(x)
    x0=x
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x0)
    x = relu_activation_block(x)
    x=layers.Conv2D(16,1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x=layers.add([x,x0])
    x=layers.Conv2D(64,1,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (56,56,3)
    x=relu_activation_block(x)
    x=layers.Conv2D(24,1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x0=x # 在56x56特征图上残差
    x=layers.Conv2D(72,1,padding='same',use_bias=False)(x0)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.Conv2D(24,1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x=layers.add([x,x0])
    x=layers.Conv2D(72,1,padding='same',use_bias=False)(x)
    x=relu_activation_block(x)
    x=layers.DepthwiseConv2D(5,strides=2,padding='same',use_bias=False)(x) # (28,28,3)
    x=relu_activation_block(x)
    multipy=x
    x=layers.GlobalAveragePooling2D(keepdims=True)(x)
    x0=x
    x=layers.Conv2D(24,1,padding='same')(x0) # 这里用了截距
    x=layers.ReLU()(x)
    x=layers.Conv2D(72,1,padding='same')(x) # 这里用了截距
    x=layers.add([x,x0])
    x=layers.ReLU()(x)
    x=layers.Multiply()([multipy,x])
    x=layers.Conv2D(40,1,padding='same',use_bias=False)(x) 
    x = layers.BatchNormalization()(x)
    for i in range(2):
        x1=x
        x=layers.Conv2D(120,1,padding='same',use_bias=False)(x1) 
        x=relu_activation_block(x)
        x=layers.DepthwiseConv2D(5,padding='same',use_bias=False)(x)
        x=relu_activation_block(x)
        multipy=x
        x=layers.GlobalAveragePooling2D(keepdims=True)(x)
        x0=x
        # 这里的两个Relu是普通的relu函数
        x=layers.Conv2D(32,1,padding='same')(x0) # 这里用了截距
        x=layers.ReLU()(x)
        x=layers.Conv2D(120,1,padding='same')(x) # 这里用了截距
        x=layers.add([x,x0])
        x=layers.ReLU()(x)
        x=layers.Multiply()([multipy,x])
        x=layers.Conv2D(40,1,padding='same',use_bias=False)(x) 
        x = layers.BatchNormalization()(x)
        x=layers.add([x,x1])
    x=layers.Conv2D(240,1,padding='same',use_bias=False)(x) 
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x) # (14,14)
    x=activation_block(x)
    x=layers.Conv2D(80,1,padding='same',use_bias=False)(x) 
    x=layers.BatchNormalization()(x)
    for filters in [200,184,184]: # 在(14,14)的特征图上狠提特征
        x0=x
        x=layers.Conv2D(filters,1,padding='same',use_bias=False)(x0) 
        x=activation_block(x)
        x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) # (14,14)
        x=activation_block(x)
        x=layers.Conv2D(80,1,padding='same',use_bias=False)(x) 
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x0])
    x=layers.Conv2D(480,1,padding='same',use_bias=False)(x) 
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) # (14,14)
    x=activation_block(x)
    multipy=x
    x=layers.GlobalAveragePooling2D(keepdims=True)(x)
    x0=x
    x=layers.Conv2D(120,1,padding='same')(x0) # 这里用了截距
    x=layers.ReLU()(x)
    x=layers.Conv2D(480,1,padding='same')(x) # 这里用了截距
    x=layers.add([x,x0])
    x=layers.ReLU()(x)
    x=layers.Multiply()([multipy,x])
    x=layers.Conv2D(112,1,padding='same',use_bias=False)(x) 
    x=layers.BatchNormalization()(x)
    x1=x
    x=layers.Conv2D(672,1,padding='same',use_bias=False)(x1) 
    x=activation_block(x)
    x=layers.DepthwiseConv2D(3,padding='same',use_bias=False)(x) 
    x=activation_block(x)
    multipy=x
    x=layers.GlobalAveragePooling2D(keepdims=True)(x)
    x0=x
    x=layers.Conv2D(168,1,padding='same')(x0) # 这里用了截距
    x=layers.ReLU()(x)
    x=layers.Conv2D(672,1,padding='same')(x) # 这里用了截距
    x=layers.add([x,x0])
    x=layers.ReLU()(x)
    x=layers.Multiply()([multipy,x])
    x=layers.Conv2D(112,1,padding='same',use_bias=False)(x) 
    x=layers.BatchNormalization()(x)
    x=layers.add([x,x1])
    x=layers.Conv2D(672,1,padding='same',use_bias=False)(x) 
    x=activation_block(x)
    x=layers.DepthwiseConv2D(5,strides=2,padding='same',use_bias=False)(x) # (7,7)
    x=activation_block(x)
    multipy=x
    x=layers.GlobalAveragePooling2D(keepdims=True)(x)
    x0=x
    x=layers.Conv2D(168,1,padding='same')(x0) # 这里用了截距
    x=layers.ReLU()(x)
    x=layers.Conv2D(672,1,padding='same')(x) # 这里用了截距
    x=layers.add([x,x0])
    x=layers.ReLU()(x)
    x=layers.Multiply()([multipy,x])
    x=layers.Conv2D(160,1,padding='same',use_bias=False)(x) 
    x=layers.BatchNormalization()(x)
    for i in range(2):
        x1=x
        x=layers.Conv2D(960,1,padding='same',use_bias=False)(x1) 
        x=activation_block(x)
        x=layers.DepthwiseConv2D(5,padding='same',use_bias=False)(x) 
        x=activation_block(x)
        multipy=x
        x=layers.GlobalAveragePooling2D(keepdims=True)(x)
        x0=x
        x=layers.Conv2D(240,1,padding='same')(x0) # 这里用了截距
        x=layers.ReLU()(x)
        x=layers.Conv2D(960,1,padding='same')(x) # 这里用了截距
        x=layers.add([x,x0])
        x=layers.ReLU()(x)
        x=layers.Multiply()([multipy,x])
        x=layers.Conv2D(160,1,padding='same',use_bias=False)(x) 
        x=layers.BatchNormalization()(x)
        x=layers.add([x,x1])
    x=layers.Conv2D(960,1,padding='same',use_bias=False)(x) 
    x=activation_block(x)
    return keras.Model(inputs,x)

# 用深度卷积提取图片空间特征和下采样,用两种逐点卷积(扩张和压缩)来扩张通道以进行
# 深度卷积提取特征,用压缩通道的逐点卷积来残差连接,让网络学习扩张--深度卷积--
# 压缩前后的变化,关于特征图,对28x28的特征图和7x7的特征图,用两个残差连接
# 对56x56的特征图用一个残差连接,对14x14的特征图网络用了5个残差连接,
# 压缩点卷积通道变化 16--24--32--64--96--160--320
# 扩张点卷积通道变化 96--144--192--384--576--960--1280
# 网络遵循扩张点卷积--深度卷积--压缩点卷积的模式,利用深度卷积来提取特征,缩小
# 特征图大小,利用两个连续的通道相同的压缩点卷积来残差连接
def get_mobilenetv2_model(input_shape,num_classes):
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