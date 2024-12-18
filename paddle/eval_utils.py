import os
import hnswlib
from paddlenlp.utils.log import logger
import numpy as np
import paddle
import json
@paddle.no_grad()
def evaluate_glue(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, labels = batch
        logits = model(input_ids)
        loss = loss_fct(logits,labels)
        correct = metric.compute(logits,labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (loss.item(),*res))
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.item(),res[0]))
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.item(),*res),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.item(), res))
    metric.reset()
    model.train()
    return res[0] if isinstance(res,list) else res

@paddle.no_grad()
def evaluate_clue(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        labels = batch.pop("labels") # 弹出栈,labels
        logits = model(**batch) # **自动拆包
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels) # 计算正确数
        metric.update(correct)
    res = metric.accumulate() # 评估结果
    logger.info("eval loss: %f, acc: %s, " % (loss.item(), res))
    metric.reset()
    model.train()
    return res

@paddle.no_grad()
def do_evaluate2(model, tokenizer, data_loader, label_normalize_dict):
    model.eval()
    total_num = 0
    correct_num = 0
    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]
    label_length = len(normed_labels[0]) # 标签长度
    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels = batch
        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(input_ids=src_ids,
                                         token_type_ids=token_type_ids,
                                         masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]
        # prediction_probs: [batch_size, label_length, vocab_size]
        prediction_probs = paddle.reshape(prediction_probs,
                                          shape=[batch_size, -1,
                                                 vocab_size]).numpy()

        # [label_num, label_length]
        label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])
        y_pred = np.ones(shape=[batch_size, len(label_ids)])
        # 计算候选标签的联合分布。
        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]
        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)
        y_true_index = []
        for masked_lm_label in masked_lm_labels.numpy():
            label_text = "".join(
                tokenizer.convert_ids_to_tokens(list(masked_lm_label)))

            label_index = normed_labels.index(label_text)
            y_true_index.append(label_index)
        y_true_index = np.array(y_true_index)
        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()
    model.train()
    return 100 * correct_num / total_num, total_num

@paddle.no_grad()
def do_evaluate_chid(model, tokenizer, data_loader, label_normalize_dict):
    """
        FCLUE `chid` 数据集在评估时具有特殊性：输入槽中包含额外的 `candidate_label_ids`，
        因此需要自定义评估函数。
    """
    model.eval()
    total_num = 0
    correct_num = 0

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]
    label_length = len(normed_labels[0])
    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_label_ids = batch
        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(input_ids=src_ids,
                                         token_type_ids=token_type_ids,
                                         masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]
        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(prediction_probs,
                                          shape=[batch_size, -1,
                                                 vocab_size]).numpy()

        candidate_num = candidate_label_ids.shape[1]
        # [batch_size, candidate_num(7)]
        y_pred = np.ones(shape=[batch_size, candidate_num])
        for label_idx in range(candidate_num):
            # [bathc_size, label_length(4)]
            single_candidate_label_ids = candidate_label_ids[:, label_idx, :]
            # Calculate joint distribution of candidate labels
            for index in range(label_length):
                # [batch_size,]
                slice_word_ids = single_candidate_label_ids[:, index].numpy()

                batch_single_token_prob = []
                for bs_index in range(batch_size):
                    # [1, 1]
                    single_token_prob = prediction_probs[
                        bs_index, index, slice_word_ids[bs_index]]
                    batch_single_token_prob.append(single_token_prob)
                y_pred[:, label_idx] *= np.array(batch_single_token_prob)
        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)
        y_true_index = []
        for index, masked_lm_label in enumerate(masked_lm_labels.numpy()):
            # [cantidate_num, label_length]
            tmp_candidate_label_ids = candidate_label_ids[index, :, :]
            for idx, label_ids in enumerate(tmp_candidate_label_ids.numpy()):
                if np.equal(label_ids, masked_lm_label).all():
                    y_true_index.append(idx)
                    continue
        y_true_index = np.array(y_true_index)
        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()
    model.train()
    return 100 * correct_num / total_num, total_num

@paddle.no_grad()
def do_evaluate(model, tokenizer, data_loader, task_label_description):
    model.eval()
    total_num = 0
    correct_num = 0
    class_num = len(task_label_description) # 15
    # [total_num * class_num, 2]
    all_prediction_probs = []
    # [total_num * class_num]
    all_labels = []
    for batch in data_loader:
        src_ids, token_type_ids, true_labels = batch
        prediction_probs = model(input_ids=src_ids,
                                 token_type_ids=token_type_ids).numpy()

        all_prediction_probs.append(prediction_probs)
        all_labels.append(true_labels.numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))
    # (total_num,class_num,1),1是属于正样本对的分数,0是属于负样本对的分数
    prediction_pos_probs = all_prediction_probs[:, :, 1] 
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    # 获取total_num中每个样本属于各个类别的预测类别,y_pred_index的值会是
    # 0--class_num-1间的值,属于模型预测sentence1和sentence2中哪一个是正样本
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)
    # 每15个是同一个原样本和不同提示的样本对,这里只用每个原样本一个真实标签就行
    # idx % class_num == 0保证15个里只取一个索引,这个索引对应真实标签
    y_true_index = np.array([
        true_label_index for idx, true_label_index in enumerate(all_labels)
        if idx % class_num == 0
    ])
    
    total_num = len(y_true_index) # 总的评估样本数,sentence1
    correct_num = (y_pred_index == y_true_index).sum() #预测对的样本数

    model.train()
    return 100 * correct_num / total_num, total_num


hnsw_max_elements=1000000
hnsw_ef=100
hnsw_m=100

def build_index(data_loader, model,output_emb_size):
    index = hnswlib.Index(
        space='ip',
        dim=output_emb_size if output_emb_size > 0 else 768)
    index.init_index(max_elements=hnsw_max_elements,
                     ef_construction=hnsw_ef,
                     M=hnsw_m)
    index.set_ef(hnsw_ef)
    index.set_num_threads(6)
    logger.info("start build index..........")
    all_embeddings = []
    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings.numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)
    logger.info("Total index number:{}".format(index.get_current_count()))
    return index

def write_recall_file(model,query_data_loader,final_index,text_list,
                      id2corpus,recall_result_file,recall_num=20):
    query_embedding = model.get_semantic_embedding(query_data_loader)
    with open(recall_result_file, 'w', encoding='utf-8') as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(\
                batch_query_embedding.numpy(),recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                text_index = batch_size * batch_index + row_index # 对应query列表中的文本
                #把原query,从索引库召回的50条语义相近索引,前两者的相似度写入召回文件
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write("{}\t{}\t{}\n".format(
                        text_list[text_index]["text"], id2corpus[doc_idx],
                        1.0 - cosine_sims[row_index][idx]))

@paddle.no_grad()
def evaluate(model, corpus_data_loader, query_data_loader, recall_result_file,output_emb_size,
             text_list,text2similar,id2corpus,recall_num=20,final_index=None):
    model.eval()
    def recall(rs, N=10):
        recall_flags = [np.sum(r[:N]) for r in rs]#前N个有一个1就是1
        return np.mean(recall_flags)#返回的是topK召回精确率
    # 构建索引库，这个不能写外面，因为构建索引的模型每次都不一样,如果写外面,传进来,这在模型不训练只评估时还行
    # 但是在训练中评估时就不行,因为不同模型构建的索引库不同,因为由他们提前的向量不同,你用新模型提取的向量
    #去旧的索引库查找,肯定出问题.
    if final_index is None:
        final_index = build_index(corpus_data_loader,model,output_emb_size)
    write_recall_file(model,query_data_loader,final_index,text_list,
                      id2corpus,recall_result_file,recall_num)
    rs = []
    with open(recall_result_file, 'r', encoding='utf-8') as f:
        relevance_labels = []
        for index, line in enumerate(f):
            #用来保存一个文本的召回文本的标记，召回到相似文本标记为1
            if index % recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []# 够一个原query的召回就清空，之后存放下个query的召回
            text, recalled_text, cosine_sim = line.rstrip().split("\t")#原文本，召回文本，距离
            #召回是模拟用户不规则的query,去召回语料库中的title,如果召回的文本和相似文本对中query对应的
            # 文本一样,那说明成功召回,设置当前召回标记1,不管多少召回,就只会有1个标记1的,但是算精度的时候,
            # 只要成功,在召回k中成功召回,都是1召回精度衡量的是N次召回的平均成功几率,text2similar中query
            # 对应的相似文本必须在corpus语料库里存在,否则就不可能成功召回,所以在评估前必须校准一下,query
            # 对应的必须是title,如果不是,就得过滤掉，因为如果在corpus里找不到,就不可能召回,标记里全是0,
            # 评估就失准
            if text2similar[text] == recalled_text:# 成功召回，设置标记1,表示找到
                relevance_labels.append(1)
            else:
                relevance_labels.append(0)# 不相同就设置0,表示没找到
    recall_N = []# 召回精确率,N越小,召回精度越高越好
    recall_nums = [1, 5, 10, 20]
    for topN in recall_nums:# 遍历召回数
        R = round(100 * recall(rs, N=topN), 3)
        recall_N.append(R)
    # evaluate_result_file = os.path.join(recall_result_dir,
    #                                     evaluate_result)
    # result = open(evaluate_result_file, 'a')
    # res = []
    # timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    # res.append(timestamp)
    for key, val in zip(recall_nums, recall_N):
        print('recall@{}={}'.format(key, val))
        # res.append(str(val))
    # result.write('\t'.join(res) + '\n')
    # print(res)
    model.train()
    score=recall_N[0]*0.3+recall_N[1]*0.3+recall_N[2]*0.2+recall_N[3]*0.2
    return score

def predict_rank_predict(model, data_loader):
    all_probs = []
    model.eval()
    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            batch_prob = model.predict(input_ids=input_ids,#批次相似度
                                       token_type_ids=token_type_ids).numpy()
            all_probs.append(batch_prob)
    if (len(all_probs) == 1): # 只预测一个批次的情况
        all_probs = np.array(all_probs)
    else:
        all_probs = np.concatenate(all_probs, axis=0) # 合并
    return all_probs
    
@paddle.no_grad()
def evaluate_rank_auc(model, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        # 获取预测概率
        pos_probs = model.predict(input_ids=input_ids,
                                  token_type_ids=token_type_ids)

        neg_probs = 1.0 - pos_probs # 负概率
        # 预测(neg_probs, pos_probs)
        preds = np.concatenate((neg_probs, pos_probs), axis=1)
        metric.update(preds=preds, labels=labels)
    auc=metric.accumulate()
    print("eval_{} auc:{:.3}".format(phase,auc))
    metric.reset()
    model.train()
    return auc

@paddle.no_grad()
def evaluate_seq_classification(model, criterion, metric, data_loader):
    model.eval() # 评估模式
    metric.reset() # 指标重置
    losses = [] # 用来保存批次损失
    for batch in data_loader:
        input_ids,token_type_ids,labels=batch
        logits = model(input_ids,token_type_ids)
        loss = criterion(logits, labels) # 计算损失
        losses.append(loss.item()) 
        correct = metric.compute(logits, labels) # 计算正确数
        metric.update(correct)  # 更新指标
    acc = metric.accumulate() # 平均准确率
    logger.info("eval loss: %.5f, acc: %.5f" % (np.mean(losses), acc))
    metric.reset()
    model.train()
    return np.mean(losses),acc

def predict_sims(model, data_loader):
    cosine_sims = []
    model.eval()
    with paddle.no_grad():
        for batch_data in data_loader:
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch_data
            batch_cosine_sim = model.cosine_sim( # [n]
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids).numpy()
            cosine_sims.append(batch_cosine_sim)
        cosine_sims = np.concatenate(cosine_sims, axis=0)
        return cosine_sims

