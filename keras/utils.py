import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def h_swish(x):  
    return x * tf.nn.relu6(x + 3.0) / 6.0  

@tf.function()
def random_invert_img(x, p=0.5):
    rand= tf.random.uniform([])
    if rand < p:
        x = 255-x
    return x
class RandomInvert(layers.Layer):
  def __init__(self, factor=0.5, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor
  def call(self, x):
    return random_invert_img(x)
# 随机反转图片颜色,必须用带@tf.function()的函数
random_invert=layers.Lambda(lambda x:random_invert_img(x))

# 可视化数据
def display(display_list):
    # plt.figure(figsize=(8,8))
    title = ["Input Image", "True Mask", "Pre Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def display2(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
    plt.show()

def load_train_data(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image=tf.image.resize(image, (180,180))
    image=tf.image.random_crop(image, (image_size, image_size, 3)) 
    label = tf.one_hot(label, depth=NUM_CLASSES)
    image=tf.cast(image,dtype=tf.float32)
    label=tf.cast(label,dtype=tf.int32)
    return image,label
def load_dev_data(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image=tf.image.resize(image, (image_size,image_size))
    label = tf.one_hot(label, depth=NUM_CLASSES)
    image=tf.cast(image,dtype=tf.float32)
    label=tf.cast(label,dtype=tf.int32)
    return image,label

auto = tf.data.AUTOTUNE
def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset=dataset.map(load_train_data,num_parallel_calls=auto)
    else:
        dataset=dataset.map(load_dev_data,num_parallel_calls=auto)
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)# 用缓冲区刷新
    dataset = dataset.batch(batch_size) #形成批次
    if is_train: #如果是训练集，增强数据
        dataset = dataset.map(
            lambda x, y: (augment_images(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)
