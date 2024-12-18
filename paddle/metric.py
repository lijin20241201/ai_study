from typing import List, Tuple
import paddle
import numpy as np
import paddle.nn as nn
from paddle.metric import Metric

class F1Score(Metric):
    """
    F1分数是精确率和召回率的调和平均数。微平均（Micro-averaging）是首先为所有示例创建一个全局的混淆矩阵
    ，然后基于这个混淆矩阵计算F1分数。这个类用于评估对话槽位填充（Dialogue Slot Filling）的性能。
    """

    def __init__(self, name='F1Score', *args, **kwargs):
        super(F1Score, self).__init__(*args, **kwargs)
        self._name = name
        self.reset()

    def reset(self):
        self.tp = {}
        self.fn = {}
        self.fp = {}
    # 计算对话槽位填充任务中的F1分数
    def update(self, logits, labels):
        # 将logits（预测的概率分布）转换为预测的标签，通过选择每个槽位上概率最高的类别索引
        preds = paddle.argmax(logits, axis=-1)
        preds = preds.numpy()
        labels = labels.numpy()
        # 这两个断言确保预测结果和标签的维度相匹配，即批次大小和槽位数量相同。
        assert preds.shape[0] == labels.shape[0]
        assert preds.shape[1] == labels.shape[1]
        # 遍历批次中的每个样本（对话轮次）。
        for i in range(preds.shape[0]):
            start, end = 1, preds.shape[1]
            while end > start: # 找到结束位置
                if labels[i][end - 1] != 0:
                    break
                end -= 1
            # 根据前面找到的start和end索引，从预测结果和标签中截取有效部分。
            pred, label = preds[i][start:end], labels[i][start:end]
            # 遍历截取后的预测结果和标签中的每个槽位。
            for y_pred, y in zip(pred, label):
                # 如果预测值和真实标签相同，则将该类别的真正例（TP）计数加一。
                if y_pred == y: # True Positive:真正例
                    self.tp[y] = self.tp.get(y, 0) + 1
                else:
                    # 如果预测值和真实标签不同，则将该预测值的假正例（FP）计数加一，并将真实标签的假负例（
                    # FN）计数加一。False Positive, False Negative
                    self.fp[y_pred] = self.fp.get(y_pred, 0) + 1
                    self.fn[y] = self.fn.get(y, 0) + 1

    def accumulate(self):
        tp_total = sum(self.tp.values())
        fn_total = sum(self.fn.values())
        fp_total = sum(self.fp.values())
        p_total = float(tp_total) / (tp_total + fp_total)
        r_total = float(tp_total) / (tp_total + fn_total)
        if p_total + r_total == 0:
            return 0
        f1_micro = 2 * p_total * r_total / (p_total + r_total)
        return f1_micro

    def name(self):
       
        return self._name

class JointAccuracy(Metric):
    """
    联合准确率（Joint Accuracy Rate）用于评估多轮对话状态追踪（Dialogue State Tracking）的性能。在每一轮对话
    中，只有当且仅当state_list中的所有状态都被正确预测时，该轮对话的状态预测才被认为是正确的。如果所有状态都被正确
    预测，则联合准确率为1；否则，联合准确率为0。
    """
    def __init__(self, name='JointAccuracy', *args, **kwargs):
        super(JointAccuracy, self).__init__(*args, **kwargs)
        self._name = name
        self.sigmoid = nn.Sigmoid()
        self.reset()
    def reset(self): # 重置指标状态
        self.num_samples = 0
        self.correct_joint = 0.0
    def update(self, logits, labels):
        # n 表示批次中对话的轮次数
        # s 表示在该轮对话中需要预测的不同状态的数量（例如，槽位-值对的数量）。
        # probs 矩阵的每一行代表一个对话轮次的状态预测概率，每一列代表一个特定状态的预测概率。
        # 每个状态都被视为一个二分类问题（即，该状态是否存在或是否被激活）。
        probs = self.sigmoid(logits)  # 计算概率
        probs = probs.numpy()
        labels = labels.numpy()
        assert probs.shape[0] == labels.shape[0] # 确保形状匹配
        assert probs.shape[1] == labels.shape[1]
        # pred 列表收集了所有预测为激活（即概率 >= 0.5）的状态索引，而 refer 列表则收集了所有真实标签中
        # 激活的状态索引
        for i in range(probs.shape[0]): # 遍历批次中对话的轮次数
            pred, refer = [], []
            for j in range(probs.shape[1]): # 每轮对话中需要预测的状态数量。
                
                if probs[i][j] >= 0.5:
                    pred.append(j)
                if labels[i][j] == 1:
                    refer.append(j)
            if not pred:
                pred = [np.argmax(probs[i])]
            # 联合准确率（Joint Accuracy Rate）的定义，即只有当且仅当所有状态都被正确预测时，
            # 该轮对话的状态预测才被认为是正确的。
            # correct_joint 记录了所有轮次中正确预测的轮次数
            if pred == refer:
                self.correct_joint += 1
        # 记录了处理的对话轮次数
        self.num_samples += probs.shape[0]

    def accumulate(self):
        joint_acc = self.correct_joint / self.num_samples
        return joint_acc
    def name(self):
        return self._name

class RecallAtK(Metric):
    """
    Recall@K 是指在前K个检索结果中，相关结果所占的比例，用于评估对话响应选择（Dialogue Response Selection）的性能。
    需要注意的是，这个类仅针对二分类任务来管理Recall@K分数。在对话系统中，对话响应选择通常被视为一个二分类问题，即判断给
    定的响应是否是对用户查询的合适回答。Recall@K指标特别适用于评估系统从大量候选响应中选出前K个最相关响应的能力。高
    Recall@K值意味着系统能够更准确地从众多候选中识别出与用户查询最相关的响应。
    """
    def __init__(self, name='Recall@K', *args, **kwargs):
        super(RecallAtK, self).__init__(*args, **kwargs)
        self._name = name
        self.softmax = nn.Softmax()
        self.reset()
    def reset(self): # 重置指标状态
        self.num_sampls = 0
        self.p_at_1_in_10 = 0.0
        self.p_at_2_in_10 = 0.0
        self.p_at_5_in_10 = 0.0
    def get_p_at_n_in_m(self, data, n, m, idx):
        # 一组中,只有第一个的标签为1,为正样本,这个记录它的模型分数
        pos_score = data[idx][0] 
        # 10个text_a一致的样本,只有第一个表示话题相关
        curr = data[idx:idx + m] 
        # 对这一组中的模型预测分数进行从高到低的排序
        curr = sorted(curr, key=lambda x: x[0], reverse=True)
        # n==1,2,5,因为分数是倒序排序,如果curr[n - 1][0] <= pos_score
        # 说明成功召回,成功召回:1,未召回:0
        # 因为返回时,他可以按这个排序截取召回n前面的部分
        if curr[n - 1][0] <= pos_score:
            return 1
        return 0
    def update(self, logits, labels):
        """
        根据当前的小批次预测结果更新状态
        Args:
            logits (Tensor): The predicted value is a Tensor with 
                shape [batch_size, 2] and type float32 or float64.
            labels (Tensor): The ground truth value is a 2D Tensor, 
                its shape is [batch_size, 1] and type is int64.
        """
        # print(logits.shape,labels) # (10,2),(10,1)
        probs = self.softmax(logits) # 求概率
        # print(probs)
        probs = probs.numpy()
        labels = labels.numpy()
        # print(probs.shape,labels.shape)
        assert probs.shape[0] == labels.shape[0]
        data = [] # 保存模型预测属于正样本的概率,真实的label
        for prob, label in zip(probs, labels):
            data.append((prob[1], label))
        # 10个样本是对一个话题的response
        assert len(data) % 10 == 0
        length = int(len(data) / 10)
        # 样本数(因为10个为一组,是对同一个话题的回应)
        self.num_sampls += length
        for i in range(length):
            idx = i * 10 # i=0的话,这个idx就是0,这是第一个sampls
            # 原始数据集特点是10个为一组相同的话题和回应,只有第一个为正确的回应
            # 标签为1,其他为0
            assert data[idx][1] == 1 # (text_a,label)
            # recall@1,recall@2,recall@5
            self.p_at_1_in_10 += self.get_p_at_n_in_m(data, 1, 10, idx)
            self.p_at_2_in_10 += self.get_p_at_n_in_m(data, 2, 10, idx)
            self.p_at_5_in_10 += self.get_p_at_n_in_m(data, 5, 10, idx)

    def accumulate(self):
        # self.p_at_1_in_10这些事累加的召回标记
        # self.num_sampls是组数
        metrics_out = [
            self.p_at_1_in_10 / self.num_sampls,
            self.p_at_2_in_10 / self.num_sampls,
            self.p_at_5_in_10 / self.num_sampls
        ]
        # 返回的列表是recall@1,recall@2,recall@5
        return metrics_out

    def name(self):
        return self._name

class SequenceAccuracy(paddle.metric.Metric):
    
    def __init__(self):
        super(SequenceAccuracy, self).__init__()
        self.correct_k = 0
        self.total = 0

    def compute(self, pred, label, ignore_index):
        pred = paddle.argmax(pred, 1)
        active_acc = label.reshape([-1]) != ignore_index
        active_pred = pred.masked_select(active_acc)
        active_labels = label.masked_select(active_acc)
        correct = active_pred.equal(active_labels)
        return correct

    def update(self, correct):
        self.correct_k += correct.cast('float32').sum(0)
        self.total += correct.shape[0]

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def accumulate(self):
        return float(self.correct_k) / self.total

    def name(self):
        return "Masked Language Model Accuracy"

def wordseg_hard_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-seg

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            acc += 1.0
            p += 1
            q += 1
            a_l = a_r + 1
            b_l = b_r + 1
            continue
        p += 1
    return acc

def wordtag_hard_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-tag

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            if list_a[p][-1] == list_b[q][-1]:
                acc += 1.0
            p += 1
            q += 1
            a_l, b_l = a_r + 1, b_r + 1
            continue
        p += 1
    return acc

def wordtag_soft_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-tag

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            if list_a[p][-1] == list_b[q][-1]:
                acc += 1.0
            elif list_b[q][-1].startswith(list_a[p][-1]):
                acc += 1.0
            elif list_b[q] == "词汇用语":
                acc += 1.0
            p += 1
            q += 1
            a_l, b_l = a_r + 1, b_r + 1
            continue
        p += 1
    return acc

def wordseg_soft_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-seg

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    i, j = 0, 0
    acc = 0.0
    a_l, b_l = 0, 0
    while i < len(list_a) and j < len(list_b):
        a_r = a_l + len(list_a[i][0]) - 1
        b_r = b_l + len(list_b[j][0]) - 1
        if a_r < b_l:
            i += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            j += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            acc += 1.0
            a_l, b_l = a_r + 1, b_r + 1
            i, j = i + 1, j + 1
            continue
        if a_l == b_l and a_r < b_r:
            cnt = 0.0
            tmp_a_r = a_r
            for k in range(i + 1, len(list_a)):
                tmp_a_r += len(list_a[k])
                cnt += 1.0
                if tmp_a_r == b_r:
                    acc += cnt
                    i, j = k + 1, j + 1
                    a_l, b_l = tmp_a_r + 1, b_r + 1
                    break
            i += 1
            continue
        if a_l == b_l and a_r > b_r:
            tmp_b_r = b_r
            for k in range(j + 1, len(list_b)):
                tmp_b_r += len(list_b[k])
                if tmp_b_r == a_r:
                    acc += 1.0
                    i, j = i + 1, k + 1
                    a_l, b_l = a_r + 1, tmp_b_r + 1
                break
            j += 1
            continue
        i += 1
    return acc