import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import keras_cv
import tensorflow as tf
import keras
from keras import layers

class BasnetLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="basnet_loss", **kwargs)
        self.smooth = 1.0e-9 # 很小的正数
        # 二元交叉熵,因为模型返回的是sigmoid处理后的pred,所以这里from_logits=False
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy(reduction=None) 
        self.ssim_value = tf.image.ssim # 结构相似
    def calculate_iou(self,y_true,y_pred):
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, [1, 2, 3]) + tf.reduce_sum(y_pred, [1, 2, 3])
        union = union - intersection
        return tf.reduce_mean(
            (intersection + self.smooth) / (union + self.smooth), axis=0
        )
    def call(self, y_true, y_pred):
        y_true_ez=tf.round(y_true)
        # 0:背景,1:前景
        class_weights = tf.constant([1.0,4.0])
        # 没加权重前,两处位置:2,乘于2是因为不乘的话,损失明显比之前损失少了很多
        class_weights = (class_weights/tf.reduce_sum(class_weights))*2
        weights=tf.gather(class_weights, indices=tf.cast(y_true_ez,tf.int32))
        cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred) # 像素级损失
        weighted_entropy_loss=cross_entropy_loss * tf.squeeze(weights,axis=-1)
        weighted_loss=tf.reduce_mean(weighted_entropy_loss)
        ssim_value = self.ssim_value(y_true, y_pred, max_val=1) # 使用 SSIM 函数计算真实值和预测值之间的结构相似性。
        #计算结构相似性损失。这里使用了 1 - ssim_value 来反转相似度得分（因为 SSIM 越高，损失应该越低）
        ssim_loss = tf.reduce_mean(1 - ssim_value + self.smooth, axis=0)
        # 个人理解iou不能作为损失,cross_entropy能做为损失是因为它比较模型预测的具体像素值和真实值
        # iou只是计算结果,之后拿1-iou,这个去指导模型训练的话,就反过来了
        # 所以损失函数是模型性能提升的源,作为损失函数必须能知道模型训练,iou不能指导训练,只是见到的计算交并比
        # 最多作为正则化项了,但是正则化项也有坏处,有可能拖后腿
        iou_value = self.calculate_iou(y_true, y_pred) # 调用之前定义的 calculate_iou 方法来计算交并比。
        iou_loss = 1 - iou_value# 计算交并比损失，也是使用 1 - iou_value 来反转得分。
        return weighted_loss + ssim_loss + iou_loss # 最后，将这三种损失相加，得到最终的损失值。
        # return weighted_loss+ssim_loss

class DiceMetric(keras.metrics.Metric):
    def __init__(self, name='dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = 1.0e-9
        self.total_dice = self.add_variable(  
            shape=(),  
            initializer='zeros',  
            name='total_dice'  
        )  
        self.batch_count = self.add_variable(  
            shape=(),  
            initializer='zeros',  
            dtype=tf.int64,  
            name='batch_count'  
        )  
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred) # 0.5(包括)以下为0,0.5以上为1
        y_true = tf.round(y_true) 
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred),axis=[1, 2, 3])
        union=tf.reduce_sum(y_pred,axis=[1, 2, 3]) + tf.reduce_sum(y_true,axis=[1, 2, 3])
        dice = tf.reduce_mean(2.*(intersection + self.smooth)/(union+ self.smooth),axis=0)
        self.total_dice.assign(self.total_dice+dice)  
        self.batch_count.assign(self.batch_count+1)  
    def result(self):  
        return self.total_dice / self.batch_count  
    def reset_states(self): 
        self.total_dice.assign(0)  
        self.batch_count.assign(0)

class IouMetric(keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        self.smooth = 1.0e-9
        super().__init__(name=name, **kwargs)
        self.total_iou = self.add_variable(  
            shape=(),  
            initializer='zeros',  
            name='total_iou'  
        )  
        self.batch_count = self.add_variable(  
            shape=(),  
            initializer='zeros',  
            dtype=tf.int64,  
            name='batch_count'  
        ) 
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred) # 0.5(包括)以下为0,0.5以上为1
        y_true = tf.round(y_true) 
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred),axis=[1,2,3])
        union = tf.reduce_sum(y_true,axis=[1,2,3]) + tf.reduce_sum(y_pred,axis=[1,2,3])
        iou_score = tf.reduce_mean(
            (intersection + self.smooth)/(union-intersection + self.smooth),axis=0) 
        self.total_iou.assign(self.total_iou+iou_score)  
        self.batch_count.assign(self.batch_count+1)
    def result(self):
        return self.total_iou / self.batch_count 
    def reset_states(self): 
        self.total_iou.assign(0)  
        self.batch_count.assign(0)

class TotalMetric(keras.metrics.Metric):  
    def __init__(self, name='avg_score', **kwargs):  
        super().__init__(name=name, **kwargs)  
        self.iou_metric = IouMetric(name='iou')  
        self.dice_metric = DiceMetric(name='dice')
       
    def update_state(self, y_true, y_pred, sample_weight=None):  
        # 更新 IoU 和 Dice 指标的状态  
        self.iou_metric.update_state(y_true, y_pred, sample_weight)  
        self.dice_metric.update_state(y_true, y_pred, sample_weight)
    def result(self):  
        # 获取 IoU 和 Dice 的当前结果  
        iou = self.iou_metric.result()  
        dice = self.dice_metric.result()
        harmonic_mean = 2.0 * (iou * dice) / (iou + dice)
        return harmonic_mean
    def reset_states(self):  
        # 重置 IoU 和 Dice 指标的状态  
        self.iou_metric.reset_states()  
        self.dice_metric.reset_states()