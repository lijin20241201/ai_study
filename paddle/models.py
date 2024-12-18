import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as nlp

class SentenceTransformer(nn.Layer):

    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"] * 3, 2)

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):
        # (n,s,d)
        query_token_embedding, _ = self.ptm(query_input_ids,
                                            query_token_type_ids,
                                            query_position_ids,
                                            query_attention_mask)
        query_token_embedding = self.dropout(query_token_embedding)
        
        query_attention_mask = paddle.unsqueeze( # (n,s,1)
            (query_input_ids != self.ptm.pad_token_id).astype(
                self.ptm.pooler.dense.weight.dtype),
            axis=2)
        # (n,s,d)*(n,s,1) 其中填充处掩码为0,相乘后填充位置的嵌入
        # 变成0
        query_token_embedding = query_token_embedding * query_attention_mask
        #(n,s,d) sum将序列上所有token表示的向量想加
        query_sum_embedding = paddle.sum(query_token_embedding, axis=1)
        #(n,t) 1轴上会聚合,得到样本非填充token数
        query_sum_mask = paddle.sum(query_attention_mask, axis=1)
        #(n,d)/(n,1) query_mean形状(n,d)是批次内样本序列的平均token嵌入
        query_mean = query_sum_embedding / query_sum_mask
        # (n,s,d)
        title_token_embedding, _ = self.ptm(title_input_ids,
                                            title_token_type_ids,
                                            title_position_ids,
                                            title_attention_mask)
        title_token_embedding = self.dropout(title_token_embedding)
        # (n,s,1) 非填充token掩码为1
        title_attention_mask = paddle.unsqueeze(
            (title_input_ids != self.ptm.pad_token_id).astype(
                self.ptm.pooler.dense.weight.dtype),
            axis=2)
        # 把填充token的嵌入置为0
        title_token_embedding = title_token_embedding * title_attention_mask
        # 把序列中所有token的嵌入表示聚合想加
        title_sum_embedding = paddle.sum(title_token_embedding, axis=1)
        # 1的轴得到所有非填充token总数
        title_sum_mask = paddle.sum(title_attention_mask, axis=1)
        # title_mean形状(n,d)是批次内样本序列的平均token嵌入
        title_mean = title_sum_embedding / title_sum_mask
        # query和title向量表示的差值(减法操作)
        sub = paddle.abs(paddle.subtract(query_mean, title_mean))
        # (n,d*3) 在特征轴合并,query,title,两者的差值
        projection = paddle.concat([query_mean, title_mean, sub], axis=-1)
        # 输出层,2分类
        logits = self.classifier(projection)

        return logits

class SimNet(nn.Layer):

    def __init__(self,
                 network,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 pad_token_id=0):
        super().__init__()

        network = network.lower()
        if network == 'bow':
            self.model = BoWModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  padding_idx=pad_token_id)
        elif network == 'cnn':
            self.model = CNNModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  padding_idx=pad_token_id)
        elif network == 'gru':
            self.model = GRUModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  direction='forward',
                                  padding_idx=pad_token_id)
        elif network == 'lstm':
            self.model = LSTMModel(vocab_size,
                                   num_classes,
                                   emb_dim,
                                   direction='forward',
                                   padding_idx=pad_token_id)
        else:
            raise ValueError(
                "Unknown network: %s, it must be one of bow, cnn, lstm or gru."
                % network)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        logits = self.model(query, title, query_seq_len, title_seq_len)
        return logits


class BoWModel(nn.Layer):
    """
    该类实现了词袋分类网络模型，用于对文本进行分类。从高层次来看，该模型首先将单词嵌入到词向量中，
    然后将这些表示通过一个词袋编码器（`BoWEncoder`）进行编码。最后，我们将编码器的输出作为最终表示，
    并将其通过一些全连接层输出（`output_layer`）。
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 fc_hidden_size=128):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        # 词袋编码
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        # 线性投影层
        self.fc = nn.Linear(self.bow_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        # 输出层
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        embedded_query = self.embedder(query) # (n,s,d)
        embedded_title = self.embedder(title) # (n,s,d)
        summed_query = self.bow_encoder(embedded_query)
        summed_title = self.bow_encoder(embedded_title)
        encoded_query = paddle.tanh(summed_query)
        encoded_title = paddle.tanh(summed_title)
        # Shape: (n,d*2)
        contacted = paddle.concat([encoded_query, encoded_title], axis=-1)
        # Shape: (n,d)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (n,c)
        logits = self.output_layer(fc_out) # 模型预测的类别分布
        # probs = F.softmax(logits, axis=-1)
        return logits


class LSTMModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=128,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=128):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        # lstm
        self.lstm_encoder = nlp.seq2vec.LSTMEncoder(emb_dim,
                                                    lstm_hidden_size,
                                                    num_layers=lstm_layers,
                                                    direction=direction,
                                                    dropout=dropout_rate)
        # 线性投影层
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        # 输出层
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len, title_seq_len):
        assert query_seq_len is not None and title_seq_len is not None
        # Shape: (n,s,d)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (n,d)
        query_repr = self.lstm_encoder(embedded_query,
                                       sequence_length=query_seq_len)
        title_repr = self.lstm_encoder(embedded_title,
                                       sequence_length=title_seq_len)
        
        # Shape: (n, 2*d)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (n,d)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (n,c)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)
        return logits


class GRUModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 gru_hidden_size=128,
                 direction='forward',
                 gru_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        
        self.gru_encoder = nlp.seq2vec.GRUEncoder(emb_dim,
                                                  gru_hidden_size,
                                                  num_layers=gru_layers,
                                                  direction=direction,
                                                  dropout=dropout_rate)
        self.fc = nn.Linear(self.gru_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len, title_seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (batch_size, gru_hidden_size)
        query_repr = self.gru_encoder(embedded_query,
                                      sequence_length=query_seq_len)
        title_repr = self.gru_encoder(embedded_title,
                                      sequence_length=title_seq_len)
        # Shape: (batch_size, 2*gru_hidden_size)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)

        return logits


class CNNModel(nn.Layer):
    """
   这个CNN模型通过词嵌入、卷积层、最大池化层以及前馈层等步骤，将输入的文本标记转换为最终的预测结果。这种结构特别
   适用于处理文本数据，特别是当需要捕捉局部特征（如n-gram）时
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=256,
                 ngram_filter_sizes=(3, ),
                 fc_hidden_size=128):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
       
        self.encoder = nlp.seq2vec.CNNEncoder(
            emb_dim=emb_dim,
            num_filter=num_filter,
            ngram_filter_sizes=ngram_filter_sizes)
        
        self.fc = nn.Linear(self.encoder.get_output_dim() * 2, fc_hidden_size)
        
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        # Shape: (n,s,d)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (n,f)
        query_repr = self.encoder(embedded_query)
        title_repr = self.encoder(embedded_title)
        # Shape: (n, 2*f)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (n,f)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)
        return logits

# 单塔 Point-wise 匹配模型适合直接对文本对进行2分类的应用场景,比如判断两个文本是否相似
# 因为输出是[负样本概率和正样本概率],所以评估可以直接用auc,不用合并概率
# 取相似度时,取[:,1]就是预测的属于正样本对的相似度,用于排序模块给文本对打分
class PointwiseMatching(nn.Layer):

    def __init__(self, pretrained_model,num_classes=2, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"],num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding) # (n,num_classes)

        return logits


class PairwiseMatching_Static(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin
        # self.pos_prob=0.0
        # self.neg_prob=0.0
        # hidden_size -> 1, calculate similarity
        self.similarity = nn.Linear(self.ptm.config["hidden_size"], 1)

    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ])
    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding) # (n,d)
        sim_score = self.similarity(cls_embedding) # (n,1)
        sim_score = F.sigmoid(sim_score) # 属于正样本对的概率

        return sim_score

    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids=None,
                neg_token_type_ids=None,
                pos_position_ids=None,
                neg_position_ids=None,
                pos_attention_mask=None,
                neg_attention_mask=None):
        # 正样本对的嵌入(n,d)
        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)
        # 负样本对的嵌入(n,d)
        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)
        
        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)
        # (n,1) 正负样本对的相似度分数
        pos_sim = self.similarity(pos_embedding) # (n,1)
        neg_sim = self.similarity(neg_embedding) # (n,1)
        # 正负样本对的相似度概率
        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)
        # record1 = pos_sim.detach()
        # record2 = neg_sim.detach()
        # self.pos_prob = paddle.mean(record1).item()
        # self.neg_prob = paddle.mean(record2).item()
        
        labels = paddle.full(shape=[pos_cls_embedding.shape[0]],
                             fill_value=1.0,
                             dtype='float32')

        loss = F.margin_ranking_loss(pos_sim,
                                     neg_sim,
                                     labels,
                                     margin=self.margin)

        return loss


# Pair-wise匹配模型适合将文本对作为特征输入到排序模块进行排序的应用场景。
# 是排序模型,不是提取特征向量的模型,用来给qeury,title对打分
class PairwiseMatching(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin
        # hidden_size -> 1, calculate similarity
        self.similarity = nn.Linear(self.ptm.config["hidden_size"], 1)

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding) # (n,d)
        sim_score = self.similarity(cls_embedding) # (n,1)
        sim_score = F.sigmoid(sim_score) # 概率(0--1)

        return sim_score

    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids=None,
                neg_token_type_ids=None,
                pos_position_ids=None,
                neg_position_ids=None,
                pos_attention_mask=None,
                neg_attention_mask=None):
        # 正样本对的嵌入(n,d)
        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)
        # 负样本对的嵌入(n,d)
        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)
        
        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)
        # (n,1) 正负样本对的相似度分数
        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)
        # 正负样本对的相似度概率
        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)
        # 标签全1
        labels = paddle.full(shape=[pos_cls_embedding.shape[0]],
                             fill_value=1.0,
                             dtype='float32')
        
        # F.margin_ranking_loss 是 PyTorch 中用于排名任务的一个损失函数，它特别适用于那些需要比较两个输
        # 入（通常是相似度分数）并根据给定的标签（通常是 1 表示第一个输入应该高于第二个输入，而 -1 表示相反）
        # 来优化模型的任务。该损失函数的基本思想是最大化正样本对之间的相似度与负样本对之间的相似度之间的差异
        # 标签,1表示第一个输入应该高于第二个输入 
        # 对于每一对样本（正样本对或负样本对）的相似度分数 pos_sim 和 neg_sim，以及对应的标签 y（其中 
        # y = 1 表示 pos_sim 应该大于 neg_sim，y = -1 表示 neg_sim 应该大于 pos_sim），损失计算
        # 如下: loss(s1​,s2​,y)=max(0,−y∗(s1​−s2​)+margin)
        # s1​ 是正样本对的相似度分数（pos_sim）,s2​ 是负样本对的相似度分数（neg_sim）。y 是标签，取值
        # 为 1 或 -1。margin 是一个超参数，用于控制边界的宽度
        loss = F.margin_ranking_loss(pos_sim,
                                     neg_sim,
                                     labels,
                                     margin=self.margin)

        return loss
# 加了kl_loss的单塔模型,用于排序模块
class QuestionMatching(nn.Layer): # 问题匹配

    def __init__(self, pretrained_model,rdrop_coef=0.0,dropout=None):
        super().__init__()
        self.ptm = pretrained_model # 预训练模型
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # 之所以加这个变量是因为forward用到,不加这个,每次都会有cls_embedding2 
        # 这是个不小的内存消耗,rdrop_coef设置了在这里效果不好,远没不设置的好
        self.rdrop_coef=rdrop_coef
        # 二分类,相似,不相似
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_loss = nlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                is_train=True):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        # 选取[CLS]的嵌入向量作为整个序列的表示
        cls_embedding = self.dropout(cls_embedding) # (n,d)
        logits = self.classifier(cls_embedding) # (n,2)
        kl_loss=0.0 
        if self.rdrop_coef>0.0 and is_train:
            # 因为默认dropout=0.1,所以输出不同,但是输入相同
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids,
                                         position_ids, attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2) # (n,2)
            kl_loss = self.rdrop_loss(logits, logits2)
        return logits, kl_loss