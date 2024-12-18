import paddle
import numpy as np
import time
import os 
import sys
import paddle.nn.functional as F
import paddle.nn as nn
from paddlenlp.utils.log import logger
# AUC（Area Under the Curve）之所以需要排序，是因为AUC的计算本质上是在评估模型对所有样本的排序能力，
# 即模型能否将正样本排在负样本前面的能力。为了计算整个数据集的AUC，你需要将所有样本的预测概率从高到低进
# 行排序，并根据它们的真实标签来计算TPR和FPR。这样，你就可以绘制出ROC曲线，并计算曲线下的面积（AUC）。
# AUC（Area Under the Curve）并不仅仅是一个排序专用指标，但它确实在评估模型的排序能力方面非常重要。
# AUC的主要用途是评估分类模型（尤其是二分类模型）的性能，它反映了模型区分正负样本的能力。然而，由于AUC的
# 计算方式不依赖于具体的分类阈值，而是考虑所有可能的分类阈值下的模型表现，因此它特别适合于评估那些关注样本
# 间相对排序顺序的场景，如推荐系统、搜索引擎的排序算法等。
# 具体来说，AUC的计算是通过考虑所有正负样本对之间的预测概率来实现的。对于每一对正负样本，如果正样本的预测概率
# 大于负样本的预测概率，则认为这一对样本被正确排序。AUC的值就是所有正负样本对中正确排序的比例。因此，AUC的值越
# 高，说明模型将正样本排在负样本前面的能力越强，即模型的排序性能越好。
# 虽然AUC在评估排序性能方面非常有用，但它并不是唯一的排序专用指标。在实际应用中，还会根据具体场景和需求选择其他指标
# 来综合评估模型的性能，如精确度（Precision）、召回率（Recall）、F1分数（F1 Score）等。这些指标可以从不同的角度
# 反映模型的性能特点
from ai_utils.paddle.eval_utils import evaluate_seq_classification

def train_seq_classification(epochs,train_data_loader,dev_data_loader,
                             model,loss_fn,metric,optimizer,scheduler,
                            save_dir,tokenizer):
    early_stop =True
    early_stop_nums=3
    global_step,best_loss = 0,1e9
    early_stop_count = 0
    tic_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        if early_stop and early_stop_count >= early_stop_nums: # 早停
            logger.info("Early stop!")
            break
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids,labels=batch
            logits = model(input_ids,token_type_ids) # logits:置信度分布
            loss = loss_fn(logits, labels) # 计算损失
            correct = metric.compute(logits, labels) # 计算正确数
            metric.update(correct)
            loss.backward()
            optimizer.step() # 更新参数
            scheduler.step()
            optimizer.clear_grad() # 清理梯度
            global_step += 1
            if global_step % 10 == 0 :
                acc = metric.accumulate() # 计算累积平均正确率
                logger.info(
                    "global step %d,batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step,step, loss.item(), acc, 10 /
                       (time.time() - tic_train)))
                tic_train = time.time() # 重置日志开始时间
                metric.reset()
        avg_val_loss,acc = evaluate_seq_classification(model, loss_fn, metric, dev_data_loader)
        if avg_val_loss < best_loss:
            early_stop_count = 0 # 当损失下降时,这个重置计数
            best_loss = avg_val_loss
            paddle.save(model.state_dict(),os.path.join(save_dir,'model_state.pdparams'))
            # model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            # 当损失不再下降,这个计数器才加1,累加超过3次,就退出训练
            early_stop_count += 1 
        logger.info("best_loss: %.5f,avg_val_loss:%.6f,avg_val_acc:%.6f,lr:%.6f" 
                    % (best_loss,avg_val_loss,acc,optimizer.get_lr()))

@paddle.no_grad()
def evaluate_pairwise(model, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        # 模型预测当前文本对属于正样本的概率
        pos_probs = model.predict(input_ids=input_ids,
                                  token_type_ids=token_type_ids)
        # 得到当前样本对属于负样本的概率
        neg_probs = 1.0 - pos_probs 
        # 合并正负样本对概率,因为指标需要
        preds = np.concatenate((neg_probs, pos_probs), axis=1)
        metric.update(preds=preds, labels=labels)
    score=metric.accumulate()
    print("eval_{} auc:{:.2}".format(phase,score))
    metric.reset()
    model.train()
    return score

def train_pairwise(epochs,model,nums,lr_scheduler,optimizer,
          metric,train_data_loader,dev_data_loader,num_training_steps,save_dir,tokenizer):
    global_step,best_score = 0,0.0
    losses=[]
    tic_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids = batch
            loss = model(pos_input_ids=pos_input_ids,
                         neg_input_ids=neg_input_ids,
                         pos_token_type_ids=pos_token_type_ids,
                         neg_token_type_ids=neg_token_type_ids)
            if nums>1:
                loss/=nums
            loss.backward()
            losses.append(loss.item())
            if step % nums==0:
                global_step += 1 
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 10 == 0: 
                    print("global step %d,batch: %d, loss: %.5f,speed: %.2f step/s"
                        % (global_step,step,np.mean(losses),10 /(time.time() - tic_train)))
                    tic_train = time.time() 
                    losses=[]
                if global_step % 300 ==0 or global_step==num_training_steps:
                    score=evaluate_pairwise(model,metric,dev_data_loader)
                    print('epoch:%d,best_score:%.6f,score:%.6f,lr:%.6f'
                         %(epoch,best_score,score,optimizer.get_lr()))
                    if best_score<score:
                        best_score=score
                        paddle.save(model.state_dict(),os.path.join(save_dir,'model_state.pdparams'))
                        tokenizer.save_pretrained(save_dir)

@paddle.no_grad()
def evaluate_pointwise_auc(model, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        # 返回logits,置信度
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        # 概率,两个值,前一个是负样本概率,后一个是正样本概率
        probs = F.softmax(logits) 
        metric.update(preds=probs.numpy(),labels=labels)
    auc=metric.accumulate()
    print("eval_{} auc:{:.5}".format(phase, auc))
    metric.reset()
    model.train()
    return auc

@paddle.no_grad()
def evaluate_pointwise(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        correct = metric.compute(logits,labels)
        metric.update(correct)
    acc = metric.accumulate()
    print("eval {} loss: {:.5}, acc: {:.5}".format(phase, np.mean(losses),
                                                    acc))
    model.train()
    metric.reset()
    return acc

def train_pointwise(epochs,model,criterion,nums,lr_scheduler,optimizer,
          metric,train_data_loader,dev_data_loader,num_training_steps,save_dir,tokenizer):
    global_step,best_score = 0,0.0
    losses,acces=[],[]
    tic_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            if nums>1:
                loss/=nums
            loss.backward()
            losses.append(loss.item())
            preds=paddle.argmax(logits,axis=-1)
            acces.append((preds==labels.reshape([-1])).sum()/len(labels))
            if step % nums==0:
                global_step += 1 
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 10 == 0: 
                    print("global step %d,batch: %d, loss: %.5f,acc: %.5f, speed: %.2f step/s"
                        % (global_step,step,np.mean(losses),np.mean(acces),10 /(time.time() - tic_train)))
                    tic_train = time.time() 
                    losses,acces=[],[]
                if global_step % 300 ==0 or global_step==num_training_steps:
                    val_acc=evaluate_pointwise(model,criterion,metric,dev_data_loader)
                    print('epoch:%d,best_score:%.6f,val_acc:%.6f,lr:%.6f'
                         %(epoch,best_score,val_acc,optimizer.get_lr()))
                    if best_score<val_acc:
                        best_score=val_acc
                        paddle.save(model.state_dict(),os.path.join(save_dir,'model_state.pdparams'))
                        tokenizer.save_pretrained(save_dir)


@paddle.no_grad()
def evaluate_dt(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    total_num = 0
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _ = model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          istrain=False)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        correct = metric.compute(logits,labels)
        metric.update(correct)
    acc = metric.accumulate()
    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(
        np.mean(losses), acc, total_num))
    model.train()
    metric.reset()
    return acc

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids, labels = batch
        logits = model(query_input_ids=query_input_ids,
                       title_input_ids=title_input_ids,
                       query_token_type_ids=query_token_type_ids,
                       title_token_type_ids=title_token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        correct = metric.compute(logits,labels) # 正确数
        metric.update(correct) # 更新指标
    acc = metric.accumulate()
    print("eval loss: %.5f, acc: %.5f" %(np.mean(losses), acc))
    model.train()
    return acc

def train_dt(epochs,model,criterion,nums,lr_scheduler,optimizer,
          metric,train_data_loader,dev_data_loader,num_training_steps,save_dir,tokenizer,rdrop_coef):
    global_step,best_score = 0,0.0
    losses=[]
    tic_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits1, kl_loss = model(input_ids=input_ids,
                                     token_type_ids=token_type_ids)
            
            ce_loss = criterion(logits1, labels)
            if kl_loss>0.0:
                # 这里还是paddle张量形式,只有张量能进行梯度运算
                loss = ce_loss + kl_loss * rdrop_coef
                # 这里变成标量,因为后边不用item了,而且后边也没用到
                kl_loss=kl_loss.item()
            else:
                loss = ce_loss
            if nums>1:
                loss/=nums
            loss.backward()
            losses.append(loss.item())
            if step % nums==0:
                global_step += 1 
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 10 == 0: 
                    print("global step %d,batch: %d, loss: %.5f,kl_loss: %.5f, speed: %.2f step/s"
                        % (global_step,step,np.mean(losses),kl_loss,10 /(time.time() - tic_train)))
                    tic_train = time.time() 
                    losses=[]
                if global_step % 300 ==0 or global_step==num_training_steps:
                    val_acc=evaluate_dt(model,criterion,metric,dev_data_loader)
                    print('epoch:%d,best_score:%.6f,val_acc:%.6f,lr:%.6f'
                         %(epoch,best_score,val_acc,optimizer.get_lr()))
                    if best_score<val_acc:
                        best_score=val_acc
                        paddle.save(model.state_dict(),os.path.join(save_dir,'model_state.pdparams'))
                        tokenizer.save_pretrained(save_dir)

def train(epochs,model,criterion,nums,lr_scheduler,optimizer,
          metric,train_data_loader,dev_data_loader,num_training_steps,save_dir,tokenizer):
    global_step,best_score = 0,0.0
    losses=[]
    tic_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids, labels = batch
            logits = model(query_input_ids=query_input_ids,
                           title_input_ids=title_input_ids,
                           query_token_type_ids=query_token_type_ids,
                           title_token_type_ids=title_token_type_ids)   
            loss = criterion(logits, labels)
            if nums>1:
                loss/=nums
            loss.backward()
            losses.append(loss.item())
            if step % nums==0:
                global_step += 1 
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 10 == 0: 
                    print("global step %d,batch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step,step,np.mean(losses),10 /(time.time() - tic_train)))
                    tic_train = time.time() 
                    losses=[]
                if global_step % 300 ==0 or global_step==num_training_steps:
                    val_acc=evaluate(model,criterion,metric,dev_data_loader)
                    print('epoch:%d,best_score:%.6f,val_acc:%.6f,lr:%.6f'
                         %(epoch,best_score,val_acc,optimizer.get_lr()))
                    if best_score<val_acc:
                        best_score=val_acc
                        paddle.save(model.state_dict(),os.path.join(save_dir,'model_state.pdparams'))
                        tokenizer.save_pretrained(save_dir)