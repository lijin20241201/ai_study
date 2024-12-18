import numpy as np
from rouge import Rouge
import paddle
from paddle.io import DataLoader,BatchSampler
from paddlenlp.data import Pad
from paddlenlp.metrics import BLEU
from functools import partial
from paddlenlp.utils.log import logger

def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
    rouge = Rouge()
    # n_size=4指的是在计算BLEU分数时考虑的n-gram的最大长度为4。nram用于比较
    # 机器翻译的输出（候选译文）与一组参考译文（通常是人工翻译）之间的相似度
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            # 根据pred和target算出rouge分
            score = rouge.get_scores(' '.join(pred), ' '.join(target))
            # 以元组的形式添加进列表
            scores.append((score[0]['rouge-1']['f'], 
                           score[0]['rouge-2']['f'],
                            score[0]['rouge-l']['f']
            ))
        except ValueError:
            scores.append((0, 0, 0))
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores]) # i[0]是rouge-1
    rouge2 = np.mean([i[1] for i in scores]) # i[1]是rouge-2
    rougel = np.mean([i[2] for i in scores])
    blue_score=bleu4.score()
    print('The auto evaluation result is:')
    print('rouge-1:%.4f,rouge-2:%.4f,rouge-l:%.4f,BLEU-4:%.4f'
         %(rouge1,rouge2,rougel,blue_score))
    return blue_score,rouge1

def convert_example(example,
                    tokenizer,
                    max_seq_len=512,
                    max_target_len=128,
                    mode='train'):
    source = example['content']
    if mode != 'test':
        # source:生成任务的源文本，应为字符串类型。
        # target:生成任务的目标文本。在训练模型时应设置，而在进行推理时应为None。默认为None
        # title (str, 可选): 某些生成任务（如摘要）的附加信息。默认为None。
        # max_seq_len (int, 可选): 编码后的序列最大长度
        # max_target_len (int, 可选): 输入target的编码后序列最大长度。默认为128。
        # pad_to_max_seq_len (bool, 可选): 是否将返回的序列填充至max_seq_len长度。注意，
        # 该方法中返回的序列将在左侧进行填充。默认为False。
        tokenized_example = tokenizer.gen_encode(source,
                                                 target=example['summary'],
                                                 max_seq_len=max_seq_len,
                                                 max_title_len=1,
                                                 max_target_len=max_target_len,
                                                 return_position_ids=True,
                                                 return_length=True)
        # 从索引1开始查找元素cls_id,就会返回目标开始符位置
        target_start = tokenized_example['input_ids'].index(
            tokenizer.cls_token_id, 1) 
        # 包含内容和摘要的整个序列长度
        target_end = tokenized_example['seq_len'] 
        # 掩码位置target :-1
        tokenized_example['masked_positions'] = list(
            range(target_start, target_end - 1))
        # target_out: 1:,下一个token任务
        tokenized_example['labels'] = tokenized_example['input_ids'][
            target_start + 1:target_end]
        return tokenized_example
    else:
        # add_start_token_for_decoding (bool, 可选): 在进行推理时，是否在序列末尾添加特殊令牌"
        # [CLS]"作为目标序列的开头，以强制模型开始生成目标序列。默认为False。
        tokenized_example = tokenizer.gen_encode(
            source,
            max_seq_len=max_seq_len,
            max_title_len=1,
            max_target_len=max_target_len,
            add_start_token_for_decoding=True,
            return_position_ids=True)

        if 'summary' in example and example['summary']:
            tokenized_example['summary'] = example['summary']
        return tokenized_example

def batchify_fn(batch_examples, pad_val, mode):
    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask) # 批次大小
        max_len = max(map(len, batch_attention_mask)) # 一个批次中的最长者
        # 初始化掩码，很大的负值表示被遮盖
        attention_mask = np.ones(
            (batch_size, max_len, max_len), dtype='float32') * -1e9
        # 遍历批次中的每一个样本
        for i, mask_data in enumerate(attention_mask):
            # 获取批次中单个样本掩码的长度
            seq_len = len(batch_attention_mask[i]) 
            # 因为填充在左侧,这里设置右下掩码数据为原掩码数据
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i],
                                                       dtype='float32')
        # 为了保证正确的广播机制,在1轴增加一个维度，可以与多头注意力的head广播(b,1,s,s)
        attention_mask = np.expand_dims(attention_mask, axis=1)
        # 返回填充后的掩码
        return attention_mask
    # 填充左边
    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype='int64')
    # 填充input_ids,填充值为pad_val
    input_ids = pad_func([example['input_ids'] for example in batch_examples])
    # 填充token_type_ids
    token_type_ids = pad_func(
        [example['token_type_ids'] for example in batch_examples])
    # 填充position_ids
    position_ids = pad_func(
        [example['position_ids'] for example in batch_examples])
    # 填充attention_mask
    attention_mask = pad_mask(
        [example['attention_mask'] for example in batch_examples])
    if mode != 'test':
        # 获取批次中样本seq_len的最长者作为整个批次要被填充到的长度
        max_len = max([example['seq_len'] for example in batch_examples])
        # 这个是移位操作,因为填充在左边,所以要为填充移动一定位置
        masked_positions = np.concatenate([
            np.array(example['masked_positions']) +
            (max_len - example['seq_len']) + i * max_len
            for i, example in enumerate(batch_examples)
        ])
        # 1：的目标序列tokens
        labels = np.concatenate([
            np.array(example['labels'], dtype='int64')
            for example in batch_examples
        ])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else: 
        return input_ids, token_type_ids, position_ids, attention_mask
   
def create_data_loader(dataset, tokenizer,trans_fun,batch_size,mode='train'):
    dataset = dataset.map(trans_fun)
    shuffle=True if mode=='train' else False
    batch_sampler = BatchSampler(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
    # 填充pad_token_id,就是0
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=collate_fn,
                             return_list=True)
    return dataset, data_loader

def post_process_sum(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos] # 预测序列
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    special_tokens = ['[UNK]']
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens

def select_sum(ids,
               scores,
               tokenizer,
               max_dec_len=None,
               num_return_sequences=1):
    results = []# 用于存储最终处理后的结果（通常是文本）
    group = [] # 用于临时存储一定数量的预测结果（及其分数，如果有的话），以便之后进行排序和选择
    tmp = [] # 用于临时存储当前批次的预测结果（及其分数，如果有的话）
    # 一种是带有分数（scores）的预测结果，另一种是不带分数的预测结果。
    if scores is not None:
        ids = ids.numpy()
        scores = scores.numpy()
        # 检查ids（预测标识符）和scores（对应的分数）的长度是否相等
        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError(
                "the length of `ids` is {}, but the `num_return_sequences` is {}"
                .format(len(ids), num_return_sequences))
        #　遍历预测结果：对于ids和scores中的每一对预测和分数
        for pred, score in zip(ids, scores):
            # 将预测标识符pred转换为文本标记pred_tokens和对应的标识符pred_token_ids
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids) # 计算生成的文本长度num_token。
            # 生成文本
            target = "".join(pred_tokens)
            
            # 如果设置了最大解码长度max_dec_len且生成的文本长度超过了这个限制，则对分数进行
            # 惩罚,减去一个较大的值
            if max_dec_len is not None and num_token >= max_dec_len:
                score -= 1e3
            # 将生成的文本和分数（或仅文本，如果未提供分数）添加到tmp列表中
            tmp.append([target, score])
            # 当tmp列表中的元素数量达到num_return_sequences时，将其添加到group列表中，并清空tmp以
            # 存储下一批结果。之后，对于group中的每一组预测结果，根据分数进行降序排序，并选择分数最高的结
            # 果（文本）添加到results列表中。
            # print(len(tmp)),tmp的长度和num_return_sequences相同
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []
        # group和批次大小相同
        for preds in group:
            # print('preds:',preds)
            # 预测长度和传人的num_return_sequences相同,按分数的倒序排序
            # preds[0][0]就是分数最高的那个文本内容
            preds = sorted(preds, key=lambda x:x[1],reverse=True)
            results.append(preds[0][0])
    else:
        ids = ids.numpy()

        for pred in ids:
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)
            response = "".join(pred_tokens)

            # TODO: Support return scores in FT.
            tmp.append([response])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            # print('1',preds)
            results.append(preds[0][0])
    return results