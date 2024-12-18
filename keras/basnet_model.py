def relu_activation_block(x): 
    x = layers.BatchNormalization()(x) 
    return layers.Activation('relu6')(x)
def activation_block(x): 
    x = layers.BatchNormalization()(x) 
    return layers.Activation(keras.activations.hard_swish)(x)

# 获取下采样的特征图(112x112,...,7x7)
def get_down_stack_maps(inputs,input_shape):
    efficientNet=keras.applications.EfficientNetV2B3(
        include_top=False,input_tensor=inputs,input_shape=input_shape)
    # efficientNet=keras.applications.EfficientNetV2S(
    #     include_top=False,input_tensor=inputs,input_shape=input_shape)
    # efficientNet=keras.applications.EfficientNetB5(
    #     include_top=False,input_tensor=inputs,input_shape=input_shape)
    # 下面的提取层是EfficientNetV2B3的
    layer_names = [
    'stem_activation',   # 112x112
    'block2c_expand_activation',   # 56x56
    'block4a_expand_activation',   # 28x28
    'block6a_expand_activation',  # 14x14
    'top_activation',      # 7x7
    ]
    # 下面的提取层是EfficientNetB5的
    # layer_names = [
    # 'block2a_expand_activation',   # 112x112
    # 'block3a_expand_activation',   # 56x56
    # 'block4a_expand_activation',   # 28x28
    # 'block6a_expand_activation',  # 14x14
    # 'top_activation',      # 7x7
    # ]
    # 下面的提取层是EfficientNetV2S的
    # layer_names = [
    # 'block1b_project_activation',   # 112x112
    # 'block2d_expand_activation',   # 56x56
    # 'block4a_expand_activation',   # 28x28
    # 'block5i_activation',  # 14x14
    # 'top_activation',      # 7x7
    # ]
    base_model_outputs = [efficientNet.get_layer(name).output for name in layer_names]
    down_stack = keras.Model(inputs=efficientNet.input, outputs=base_model_outputs)
    down_stack.trainable=False
    return down_stack

# 定义上采样器（解码器）：
def upsample(inputs,filters, size,dropout=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    x=layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(inputs)    
    x=layers.BatchNormalization()(x)
    if dropout:
        x=layers.Dropout(dropout)(x)
    x=layers.Activation(keras.activations.hard_swish)(x)
    return x

# basnet的核心就是分粗提取和细提取,这个相当有用
def basnet_predict(input_shape, out_classes): # (224,224,3)
    inputs = layers.Input(shape=input_shape)
    down_stack=get_down_stack_maps(inputs,input_shape)
    skips = down_stack.outputs # 下采样特征图,里面4个特征图
    x = skips[-1] # 7x7特征图,先从这个特征图开始上采样
    skips = list(reversed(skips[:-1])) # 14x14,...,112x112
    for i,filters in enumerate([512,256,128,64]):
        dropout=0.5-i*0.15
        x=upsample(x,filters,3,dropout) # 14x14
        # 在特征轴合并上采样后的特征图和对应尺寸的特征图
        x=layers.Concatenate()([x,skips[i]])
        # print(x.shape)
    outputs =layers.Conv2DTranspose(out_classes,3,strides=2,padding='same')(x) # (224,224,1) 
    # print(outputs.shape)
    return keras.Model(inputs=inputs, outputs=outputs)

def convolution_block(x_input, filters,strides=1): # 卷积块
    x = layers.Conv2D(filters,3, padding="same",strides=strides,use_bias=False)(x_input)
    x=activation_block(x) # 批次标准化,hard_swish激活函数
    return x

def basnet_rrm(base_model, out_classes):
    num_stages = 4
    filters = 64
    x_input = base_model.output  # (224,224,1)
    x=x_input
    encoder_blocks = [] # (112,112,64),(56,56,64),(28,28,64),(14,14,64)
    for i in range(num_stages): # 0,1,2,3
        x = convolution_block(x,filters=filters,strides=2) # 特征提取
        encoder_blocks.append(x)
    x = convolution_block(x,filters=filters,strides=2) # (7,7,64)
    for i in reversed(range(num_stages)): # 3,2,1,0
        dropout=None if i==0 else 0.3-0.1*(num_stages-1-i)
        x=upsample(x,filters,3,dropout) # 上采样 (14,14)
        # 7x7上采样后是14x14,之后和对应尺寸的下采样特征图合并特征
        x = layers.Concatenate()([encoder_blocks[i], x]) 
    x = layers.Conv2DTranspose(out_classes,3,strides=2,padding='same')(x) 
    outputs = layers.Add()([x_input, x]) # 残差连接,有利于模型学习差异
    return keras.Model(inputs=base_model.input, outputs=outputs)

def basnet(input_shape, out_classes): # (224,224,1)
    predict_model=basnet_predict(input_shape,out_classes)
    refine_model = basnet_rrm(predict_model, out_classes) 
    x=refine_model.output # logits
    output=layers.Activation('sigmoid')(x) # 处理成概率
    return keras.models.Model(inputs=predict_model.input, outputs=output)