# 获取支持数据:分析结果,支持数据条数,阈值
def get_support_data(analysis_result, support_num, support_threshold=0.7):
    ret_idxs = [] 
    ret_scores = [] 
    rationale_idx = 0
    try:# 退出条件:没找到足够的support_num
        while len(ret_idxs) < support_num: 
            # 遍历分析结果
            for res in analysis_result:
                # 获取当前rationale_idx对应的分数
                #这个是支持数据集中和当前样本最相似的
                score = res.pos_scores[rationale_idx]
                # 如果分数大于阈值
                if score > support_threshold:
                    # 获取对应的支持数据集中的索引
                    idx = res.pos_indexes[rationale_idx]
                    # ret_idxs不加重复的
                    if idx not in ret_idxs:
                        ret_idxs.append(idx)
                        ret_scores.append(score)
                    # 够数就退出
                    if len(ret_idxs) >= support_num:
                        break
            # 如果遍历一遍analysis_result后不够support_num
            # 就设置rationale_idx += 1
            rationale_idx += 1
    except IndexError:
        logger.error(
            f"The index is out of range, please reduce support_num or increase support_threshold. Got {len(ret_idxs)} now."
        )
    return ret_idxs, ret_scores

def get_dirty_data(weight_matrix, dirty_num, threshold=0):
    scores = []
    # 遍历代表点模型抽取的每个样本的权重,weight_matrix形状:(n,6)
    for idx in range(weight_matrix.shape[0]):
        weight_sum = 0 # 用来累加大于阈值的权重
        count = 0 # weight大于阈值的计数器
        # weight:每个样本的权重
        for weight in weight_matrix[idx].numpy():
            if weight > threshold: # 如果weight比阈值大
                count += 1
                weight_sum += weight
        # 每个样本的分数是大于阈值的权重个数和这些权重的和
        scores.append((count, weight_sum))
    # 倒序，权重大于阈值的个数越多,说明数据越可能是脏数据
    sorted_idx_score_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)  
    sorted_idxs = [idx for idx, _ in sorted_idx_score_pairs]  
    sorted_scores = [score for _, score in sorted_idx_score_pairs]
    # 返回200条脏数据分数
    ret_scores = sorted_scores[:dirty_num]
    # 返回这些分数对应的训练样本的索引
    ret_idxs = sorted_idxs[:dirty_num]
    return ret_idxs, ret_scores

def get_sparse_data(analysis_result, sparse_num): # 获取稀疏数据
    idx_scores = {} # 保存索引-->相似度分数
    preds = [] # 保存预测类别,这个类别不是数据标注类别,而是模型预测
    for i,res in enumerate(analysis_result):
        # 与当前样本相似的3个训练集中的样本
        scores = res.pos_scores  
        # 计算出平均分数
        idx_scores[i] = sum(scores) / len(scores) 
        # 把模型预测类别加进去
        preds.append(res.pred_label)
    # 按分数从低到高排序,分数低的说明在训练集中很难找到相似的语义文本
    idx_socre_list = sorted(idx_scores.items(),
                                 key=lambda x: x[1])[:sparse_num]
    ret_idxs, ret_scores = zip(*idx_socre_list)
    # 返回稀疏数据在评估集中的索引,分数,模型预测
    return ret_idxs, ret_scores, preds

