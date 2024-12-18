import os
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

def write_result3(file_name,test_ds,model_list):
    with open(file_name,mode='w',encoding='utf-8') as f:
        format_str=''
        for img,img_path in test_ds:
            top_3_pred_lst=[]
            pred=None
            for model in model_list:
                pred_i=model.predict(img[tf.newaxis,...],verbose=0)
                prob_i = tf.nn.softmax(pred_i, axis=-1).numpy()[0]
                prob_tupule=zip(range(len(prob_i)),prob_i)
                top_3=sorted(prob_tupule,key=lambda x:x[1] ,reverse=True)[:3]
                top_3_pred_lst.append(top_3)
            class_weights={}
            for i in range(len(top_3_pred_lst)):
                for j in range(3):
                    current_label=top_3_pred_lst[i][j][0]
                    current_prob=top_3_pred_lst[i][j][1]
                    # print('第'+str(j)+'个',current_label,current_prob)
                    if current_label not in class_weights.keys():
                        class_weights[current_label]=current_prob
                    else:
                        class_weights[current_label] += current_prob
            sort_weights=sorted(class_weights.items(),key=lambda x:x[1],reverse=True)
            pred=sort_weights[0][0]
            print('根据top3权重选择:',sort_weights,pred)
            img_name=img_path.numpy().decode('utf-8').split('/')[-1] # 图片名
            format_str += '{},{}\n'.format(img_name,pred)
        f.write(format_str)

def top_5_result(file_name,test_ds,model_list):
    with open(file_name,mode='w',encoding='utf-8') as f:
        format_str=''
        for img,img_path in test_ds:
            pred_lst=[]
            prob_lst=[]
            top_5_pred_lst=[]
            # top_5_prob_lst=[]
            dict_=defaultdict(list) # 每次新样本预测都重置
            # print(dict_)
            pred=None
            for model in model_list:
                pred_i=model.predict(img[tf.newaxis,...],verbose=0)
                prob_i = tf.nn.softmax(pred_i, axis=-1).numpy()[0]
                prob_tupule=zip(range(len(prob_i)),prob_i)
                top_5=sorted(prob_tupule,key=lambda x:x[1] ,reverse=True)[:5]
                # print(prob_i.max(),prob_i.min(),prob_i.std(),prob_i.mean())
                # max_value = np.max(prob_i)
                # min_value = np.min(prob_i)
                # 这里不应该对概率归一化,因为这个代表置信度,低置信度的归一化后会变很大
                # 因为比如一个模型预测的最高概率0.5,最小概率0,如果归一化,本来较低的置信度
                # 会变成1
                # prob_i=(prob_i - min_value) / (max_value - min_value)
                prob_i=np.max(prob_i) # 当前模型的最高置信度
                pred_i=tf.argmax(pred_i,axis=-1).numpy()[0] # 当前模型置信的标签
                pred_lst.append(pred_i)
                prob_lst.append(prob_i)
                top_5_pred_lst.append(top_5)
            for pred,prob in zip(pred_lst,prob_lst):
                if pred not in dict_.keys():
                    dict_[pred]=[prob]
                else:
                    dict_[pred].append(prob)
            dict_=sorted(dict_.items(),key=lambda x: len(x[1]),reverse=True)
            if len(dict_)==1: # 第一个预测一致
                pred=dict_[0][0]
                print('所有模型预测一致',pred,dict_[0][1])
            else: # 其他情况
                class_weights={}
                for i in range(len(top_5_pred_lst)):
                    for j in range(5):
                        current_label=top_5_pred_lst[i][j][0]
                        current_prob=top_5_pred_lst[i][j][1]
                        # print('第'+str(j)+'个',current_label,current_prob)
                        if current_label not in class_weights.keys():
                            class_weights[current_label]=current_prob
                        else:
                            class_weights[current_label] += current_prob
                           
                sort_weights=sorted(class_weights.items(),key=lambda x:x[1],reverse=True)
                pred=sort_weights[0][0]
                print('根据top5权重选择:',sort_weights,pred)
            img_name=img_path.numpy().decode('utf-8').split('/')[-1] # 图片名
            format_str += '{},{}\n'.format(img_name,pred)
        f.write(format_str)



