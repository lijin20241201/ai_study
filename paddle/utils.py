import os
from paddlenlp.data import DataCollatorWithPadding
import random
import numpy as np
import paddle
import json
from paddlenlp.utils.log import logger
from paddlenlp.transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from paddlenlp.datasets import MapDataset

# 更新模型的dropout
def update_model_dropout(model, p=0.0):
    model.base_model.embeddings.dropout.p = p
    for i in range(len(model.base_model.encoder.layers)):
        model.base_model.encoder.layers[i].dropout.p = p
        model.base_model.encoder.layers[i].dropout1.p = p
        model.base_model.encoder.layers[i].dropout2.p = p

# 负样本标题neg_title有一定概率和正样本标题title一样
# 所以对于新生成的样本还要过滤,把一样的过滤掉
def gen_pair(dataset, pool_size=100):
    if len(dataset) < pool_size:
        pool_size = len(dataset)
    new_examples = [] # 新样本
    pool = [] # 临时容器,存放样本对应的title
    tmp_exmaples = [] # 临时容器,存放样本
    for example in dataset:
        label = example["label"]
        # 这里是生成neg_title,label==0的本来就是neg_title
        # 要生成也是正标题,但是正标题无法生成
        if label == 0:
            continue
        tmp_exmaples.append(example)
        pool.append(example["title"]) #池子存放的是样本对应的标题
        if len(pool) >= pool_size: # 如果够批次了
            np.random.shuffle(pool) # 随机刷新title顺序
            #遍历临时容器中的每个样本
            for idx, example in enumerate(tmp_exmaples): 
                # 设置neg_title为别人的title
                example["neg_title"] = pool[idx] 
                # 把修改后的样本加入新样本集
                new_examples.append(example)
            tmp_exmaples = [] # 清空,以存放下个批次数据
            pool = [] 
    if len(pool)>0:
        np.random.shuffle(pool) 
        for idx, example in enumerate(tmp_exmaples): 
            # 设置neg_title为别人的title
            example["neg_title"] = pool[idx] 
            # 把修改后的样本加入新样本集
            new_examples.append(example)
    return MapDataset(new_examples)

def cal_md5(str):
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()

def create_dataloader(dataset,
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None,
                      return_list=True,
                      mode='train'):
    if trans_fn:
        dataset = dataset.map(trans_fn)
    shuffle = True if mode == 'train' else False
    batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    if not return_list:
        return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn)
    
    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn, 
                                return_list=return_list)

# 重复词策略
def word_repetition(input_ids, token_type_ids, dup_rate=0.32):
    """Word Repetition strategy."""
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()
    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = [] # 用来装重复词后的批次ids
    repetitied_token_type_ids = [] # 用来装重复词后的批次sids
    rep_seq_len = seq_len # 用来设定重复词策略后的批次最大序列长度
    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id) # 非填充token
        dup_word_index = []
        # If sequence length is less than 5, skip it
        if actual_len > 5:
            # 重复长度是0--int(dup_rate * actual_len)之间的值
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            # 刨除[CLS]和[SEP],随机采样dup_len个
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):
            # 插入重复单词,如果idx在dup_word_index中,idx从0开始,
            # 在里面,说明被采样为要重复的token
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            # 正常的token只会被添加一次,选中的重复词token会被添加两次
            # 实现了重复词策略
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id) # 重复词后的批次内单样本长度
        repetitied_input_ids.append(r_input_id) 
        repetitied_token_type_ids.append(r_token_type_id)
        # 更新rep_seq_len
        if after_dup_len > rep_seq_len:
            rep_seq_len = after_dup_len
    # 填充批次数据到同一序列长度
    for batch_id in range(batch_size):
        after_dup_len = len(repetitied_input_ids[batch_id]) # 这个批次内第i个样本的序列长度
        pad_len = rep_seq_len - after_dup_len # 要填充的长度
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len
    # 返回重复词策略后的数据
    return paddle.to_tensor(repetitied_input_ids, dtype="int64"), paddle.to_tensor(
        repetitied_token_type_ids, dtype="int64"
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def read_by_lines(path):
    result = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            result.append(line.strip())
    return result

def write_by_lines(path, data):
    with open(path, "w", encoding="utf8") as f:
        [f.write(d + "\n") for d in data]

def write_text(path,data):
    with open(path,mode='w',encoding='utf-8') as fout:
        for i in data:
            fout.write('{}\n'.format(i.strip()))