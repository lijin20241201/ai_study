import abc
import sys
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp

class SemanticIndexBase(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768,
                                                      output_emb_size,
                                                      weight_attr=weight_attr)
            
    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding # 返回单位向量(n,d)

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)
                yield text_embeddings # (n,d)
    
    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask)
        #(n,d)*(n,d)=(n,d) #query中各个token的向量与title中各个token的向量点乘
        #就是返回每个query，title对的余弦相似度，也是相应向量夹角的余弦值
        #向量的点积=两个向量的模*两个向量间夹角的余弦值
        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1) # (n,)
        return cosine_sim

    @abc.abstractmethod
    def forward(self):
        pass

class SemanticIndexBaseStatic(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768,
                                                      output_emb_size,
                                                      weight_attr=weight_attr)

    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ])
    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        # 向量单位化,向量的模变成1
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding # (n,d)

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)
                # 返回输入数据的语义向量
                yield text_embeddings # yield (n,d)

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):
        #(n,d)
        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask)
        # (n,d)
        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask)
        #(n,d)*(n,d)就是批次中对应query和title的余弦相似度
        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding) # (n,d)
        cls_embedding = self.dropout(cls_embedding)
        # 返回文本的语义向量(n,d),是单位向量
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding

class SemanticIndexBatchNeg(SemanticIndexBase):
    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.2,
                 scale=20,
                 output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)
        self.margin = margin
        # 加速收敛模型
        self.sacle = scale
        self.record = None
        self.classifier = nn.Linear(output_emb_size, 2)
        # self.linear1=nn.Linear(output_emb_size*2,output_emb_size)
        self.rdrop_loss = paddlenlp.losses.RDropLoss()
        self.pos_avg = 0.0
        self.neg_avg = 0.0
    # 加个分类没用,这个模型不是用于分类的,是用来提取特征的,分类效果不好
    # def predict(self,
    #                 query_input_ids,
    #                title_input_ids,
    #                query_token_type_ids=None,
    #                query_position_ids=None,
    #                query_attention_mask=None,
    #                title_token_type_ids=None,
    #                title_position_ids=None,
    #                title_attention_mask=None):

    #     query_cls_embedding = self.get_pooled_embedding(query_input_ids,
    #                                                     query_token_type_ids,
    #                                                     query_position_ids,
    #                                                     query_attention_mask)

    #     title_cls_embedding = self.get_pooled_embedding(title_input_ids,
    #                                                     title_token_type_ids,
    #                                                     title_position_ids,
    #                                                     title_attention_mask)
    #     contacted = paddle.concat([query_cls_embedding,title_cls_embedding ], axis=-1) #(n,2*d)
    #     contacted =self.linear1(contacted) # (n,d)
    #     contacted  = self.dropout(contacted) 
    #     logits = self.classifier(contacted) # (n,2)
    #     return logits  
    
    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None,train=True):
        #(n,d)
        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask)
        #(n,d)
        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask)
        logits1 = self.classifier(query_cls_embedding) # (n,2)
        logits2 = self.classifier(title_cls_embedding)
        kl_loss = self.rdrop_loss(logits1, logits2)
        
        # (n,d)@(d,n)-->(n,n) query批次中的每一行表示第i个样本和title批次中的
        # 全部样本的相似度
        cosine_sim = paddle.matmul(query_cls_embedding,
                                   title_cls_embedding,
                                   transpose_y=True)
        
        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]],
                                  fill_value=self.margin,
                                  dtype=paddle.get_default_dtype())
        
        # 从所有正样本中减去margin值，然后计算余弦相似度（cosine_sim()）。
        cosine_sim = cosine_sim - paddle.diag(margin_diag) # (n,n)
        # 当你对一个张量调用detach()时，它会返回一个新的张量，这个新张量与原始张量共享数据，但它们在计算图中是独立的。
        # 换句话说，新张量将不会计算梯度，也不会将梯度传播回原始张量所在的计算图部分。
        # 新张量与原始张量共享数据意味着这两个张量在内存中指向相同的数值数据区域。
        self.record = cosine_sim.detach()
        # 相似度矩阵行和列中的较小值
        size = self.record.shape[0]
        # 相似度矩阵中所有元素
        num_item = size **2
        if size>1: # 如果size==1的话,num_item-size会相等,除0错误
            # 对角线上被视为正样本,这个是正样本的平均分数,这时已经减了一个margin
            self.pos_avg = paddle.diag(self.record).sum().item() / size
            self.neg_avg =(self.record.sum().item() - paddle.diag(
                self.record).sum().item())/(num_item-size)
        
        # 让模型更快收敛
        cosine_sim *= self.sacle
        # 批次索引
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        # labels = paddle.reshape(labels, shape=[-1, 1]) #(n,1)
        # 交叉熵损失
        loss = F.cross_entropy(input=cosine_sim, label=labels)
       
        return loss,kl_loss

class SimCSE(nn.Layer):
    
    def __init__(self, pretrained_model, dropout=None, margin=0.0, scale=20, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model# 预训练模型
        # dropout is not None和dropout是不一样的,dropout＝０．时,dropout是Ｆalse,dropout is not None是Ｔrue
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:# 如果output_emb_size>0,线性转换
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768, output_emb_size, weight_attr=weight_attr)
        self.margin = margin
        self.scale = scale
        self.classifier = nn.Linear(output_emb_size, 2)
        self.rdrop_loss = paddlenlp.losses.RDropLoss()
    
    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ])
    def get_pooled_embedding(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, with_pooler=True
    ):
        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        if with_pooler is False:# 如果ptm不返回池化层,把［CLS］输出作为池化输出
            cls_embedding = sequence_output[:, 0, :]
        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)# 向量单位化（n,d）
        return cls_embedding
        
    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)
                yield text_embeddings # 获取文本语义嵌入(n,d)

    def cosine_sim(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
        with_pooler=True,
    ):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask, with_pooler=with_pooler
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask, with_pooler=with_pooler
        )
        # query和title的余弦相似度,相当于对应向量点乘,因为两个都是单位向量,所以就是两个向量夹角的余弦值
        #(n,d)*(n,d)=(n,d),对应数据相乘之后聚合,相当于向量相乘
        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding, axis=-1)
        return cosine_sim
    
    def forward(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
    ):
        # query语义向量(n,d)
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )
        # title语义向量(n,d)
        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )
        logits1 = self.classifier(query_cls_embedding) # (n,2)
        logits2 = self.classifier(title_cls_embedding)
        kl_loss = self.rdrop_loss(logits1, logits2)
        # 因为query和title语义相似，所以这里是训练两者损失接近0
        # (n,d)@(d,n)=(n,n),得到的是query中的每个语义向量和title的每个语义向量的余弦值．
        # 每一行都是query和当前批次中的title的余弦相似度,按理对角线上更相似,因为对角线上是对应的query和title
        # 而余弦值范围是-1--1,所以模型要做的是让对角线上的值变大，其他位置值变小
        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)
        # 现在只是list形式
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype=paddle.get_default_dtype()
        )
        cosine_sim = cosine_sim - paddle.diag(margin_diag)# 在对角线上减去固定值
        # 缩放余弦相似度，让模型更好的收敛
        cosine_sim *= self.scale
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        # (n,1),下面这个不必要,因为硬标签时,paddle期望的标签形状是[n]或(n,1)
        # labels = paddle.reshape(labels, shape=[-1, 1]) # (n,1)
        # 会让模型区分问题的不同,在PaddlePaddle中，没有直接接受One-Hot编码作为输入的交叉熵函数。通常，交叉熵损失函数
        # （如F.cross_entropy）期望的输入是类别索引（class indices），而不是One-Hot编码的向量
        loss = F.cross_entropy(input=cosine_sim, label=labels)
        return loss,kl_loss
