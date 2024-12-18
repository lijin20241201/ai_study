import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import AutoTokenizer, AutoModel, ErnieForMaskedLM
from ai_utils.paddle.custom_ernie import ErnieModel as CustomErnie

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    #mask中1的位置填充y对应,0的位置填充x对应元素
    return paddle.where(mask, y, x)

def get_special_tokens():
    return ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

def get_special_token_ids(tokenizer):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    return tokenizer.convert_tokens_to_ids(special_tokens)

def get_special_token_dict(tokenizer):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    special_token_dict = dict(
        zip(special_tokens, tokenizer.convert_tokens_to_ids(special_tokens)))
    return special_token_dict

def mask_tokens(batch_inputs, tokenizer, mlm_probability=0.15):
    """
    Description: Mask input_ids for masked language modeling: 80% MASK, 10% random, 10% original
    """
    # 克隆输入：首先，通过.clone()方法克隆batch_inputs来创建mlm_inputs和mlm_labels的副本。这两个副本将用于后续的MLM处理，
    # 而原始batch_inputs保持不变。
    mlm_inputs = batch_inputs.clone()
    mlm_labels = batch_inputs.clone()
    # 创建概率矩阵：使用paddle.full创建一个形状与mlm_inputs相同的张量probability_matrix，并将其所有元素初始化为mlm_probability
    # （一个预设的遮蔽概率值，如0.15）。
    probability_matrix = paddle.full(mlm_inputs.shape, mlm_probability)
    special_tokens_mask = paddle.cast(paddle.zeros(mlm_inputs.shape),
                                      dtype=bool)
    # 特殊标记掩码：初始化一个全为False的布尔型张量special_tokens_mask，其形状与mlm_inputs相同。然后，遍历所有特殊标记ID
    # （如[CLS], [SEP]等），并将mlm_inputs中这些特殊标记的位置在special_tokens_mask中标记为True。
    for special_token_id in get_special_token_ids(tokenizer):
        special_tokens_mask |= (mlm_inputs == special_token_id)
    # 调整概率矩阵：将probability_matrix中特殊标记位置的概率设置为0，因为这些位置不应该被遮蔽。
    probability_matrix = masked_fill(probability_matrix, special_tokens_mask,0.0)
    # 生成遮蔽索引：根据调整后的probability_matrix，使用paddle.bernoulli生成一个布尔型张量masked_indices，
    # 表示哪些位置应该被遮蔽。
    masked_indices = paddle.cast(paddle.bernoulli(probability_matrix),
                                 dtype=bool)
    # 处理标签：将mlm_labels中未被遮蔽的位置（即~masked_indices）设置为-100，这通常用于在模型训练时
    # 忽略这些位置的损失计算。
    # ~masked_indices 使用了按位取反操作符（bitwise NOT operator），它会将 True 转换为 False，
    # 将 False 转换为 True,被填充为-100的是没被遮挡的位置
    mlm_labels = masked_fill(mlm_labels, ~masked_indices, -100)

    # 80%的时间：将 masked_indices中80%的位置替换为tokenizer.mask_token_id（即[MASK]标记）。
    # 当您在布尔张量上使用&时，它会对两个张量中对应位置的元素进行逐元素比较。如果两个元素都是True，
    # 则结果张量中对应位置的元素也是True；否则，结果是False。
    indices_replaced = paddle.cast(paddle.bernoulli(
        paddle.full(mlm_inputs.shape, 0.8)),
                                   dtype=bool) & masked_indices
    
    mlm_inputs = masked_fill(mlm_inputs, indices_replaced,
                             tokenizer.mask_token_id)

    # 10%的时间：在剩余的masked_indices中，随机选择50%的位置（即总体遮蔽位置的10%），并用随机单词ID替换这些位置。
    # 这些随机单词ID通过paddle.randint生成。
    indices_random = paddle.cast(
        paddle.bernoulli(paddle.full(mlm_inputs.shape, 0.5)),
        dtype=bool) & masked_indices & ~indices_replaced
    random_words = paddle.randint(0,
                                  len(tokenizer),
                                  mlm_inputs.shape,
                                  dtype=mlm_inputs.dtype)
    mlm_inputs = paddle.where(indices_random, random_words, mlm_inputs)

    #剩余的10%：保持这些位置不变。返回处理后的输入和标签
    # 处理后的输入,15%的概率被处理,在15%里,80%的概率被遮挡,10%概率被替换,10%保持不变
    return mlm_inputs, mlm_labels
# 全连接层
class ProjectionMLP(nn.Layer): 

    def __init__(self, in_dim):
        super(ProjectionMLP, self).__init__()
        hidden_dim = in_dim * 2
        out_dim = in_dim
        affine = False
        # 一维批量归一化层
        # 在使用nn.BatchNorm1D时，需要注意它通常用于处理形状为(N, C, L)的输入，其中N是批量大小，C是特征数量（通道数），L是特征长度。
        # 然而，在这个上下文中，由于它紧跟在线性层之后，输入实际上是(N, hidden_dim)，即L=1。因此，这里使用nn.BatchNorm1D是合适的，
        # 尽管它主要用于处理序列数据（如时间序列或自然语言处理中的嵌入向量），但在某些情况下也可以用于简单的全连接层之后。
        list_layers = [
            nn.Linear(in_dim, hidden_dim, bias_attr=False),
            nn.BatchNorm1D(hidden_dim),
            nn.ReLU()
        ]
        list_layers += [
            nn.Linear(hidden_dim, out_dim, bias_attr=False),
            nn.BatchNorm1D(out_dim)
        ]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Layer):
    
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(axis=-1) # nn.CosineSimilarity是一个用于计算两个张量之间余弦相似度的函数
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y, one_vs_one=False):
        if one_vs_one: # one2one,直接返回相似度
            sim = self.cos(x, y) 
            return sim

        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        sim = self.cos(x, y)
        # detach() 方法用于从计算图中分离一个张量（Tensor），使其成为一个新的张量，这个新张量不再需要计算梯度。
        # 这通常用于当你想要保存一个张量的值用于后续的计算或记录，但又不想让这个张量参与反向传播（即梯度计算）时
        self.record = sim.detach()
        #相似度矩阵行和列中的较小值
        min_size = min(self.record.shape[0], self.record.shape[1])
        # 相似度矩阵中所有元素
        num_item = self.record.shape[0] * self.record.shape[1] 
        # 对角线上被视为正样本,这个是正样本的平均分数
        self.pos_avg = paddle.diag(self.record).sum().item() / min_size
        # 除了对角线上,其他的被视为负样本,这个是负样本的平均相似度分数
        self.neg_avg = (self.record.sum().item() - paddle.diag(
            self.record).sum().item()) / (num_item - min_size)
        return sim / self.temp
# 编码器的主要任务是提取输入文本（如query_input_ids和key_input_ids）的语义特征，这些特征随后可以用于各种下游任务
# ，如文本分类、聚类或检索。编码器通常会有一个损失函数（如对比损失Contrastive Loss），用于优化编码器提取的特征，使得来
# 自相同语义的文本在特征空间中更加接近，而来自不同语义的文本则更加远离。
class Encoder(nn.Layer):

    def __init__(self, pretrained_model_name, temp=0.05, output_emb_size=None):
        super(Encoder, self).__init__()
        self.ptm = AutoModel.from_pretrained(pretrained_model_name)
        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size
        self.output_emb_size = output_emb_size
        self.mlp = ProjectionMLP(self.ptm.config['hidden_size']) # mlp

        if output_emb_size is not None:
            self.emb_reduce_linear = nn.Linear(self.ptm.config['hidden_size'],
                                               output_emb_size)

        self.temp = temp
        self.sim = Similarity(temp)

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             with_pooler=True):
        
        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)
        if not with_pooler:
            ori_cls_embedding = sequence_output[:, 0, :]
        else:
            ori_cls_embedding = cls_embedding
        
        mlp_cls_embedding = self.mlp(ori_cls_embedding) # 全连接层 (n,d)
        if self.output_emb_size is not None:
            cls_embedding = self.emb_reduce_linear(mlp_cls_embedding) # (n,32)

        return cls_embedding, mlp_cls_embedding

    def cosine_sim(self,
                   query_input_ids,
                   key_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   key_token_type_ids=None,
                   key_position_ids=None,
                   key_attention_mask=None,
                   with_pooler=False):
        query_cls_embedding, _ = self.get_pooled_embedding( # (n,32)
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler)
        key_cls_embedding, _ = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler)
        # 单样本对的余弦相似度,one_vs_one=True
        cosine_sim = self.sim(query_cls_embedding,
                              key_cls_embedding,
                              one_vs_one=True)
        return cosine_sim

    def forward(self,
                query_input_ids,
                key_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                key_token_type_ids=None,
                key_position_ids=None,
                key_attention_mask=None,
                with_pooler=False):
        query_cls_embedding, mlp_query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler)
        key_cls_embedding, mlp_key_cls_embedding = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler)
        # 多样本对的余弦相似度,默认one_vs_one=False,cosine_sim被缩放了
        cosine_sim = self.sim(query_cls_embedding, key_cls_embedding) # (n,n)
        
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        # labels = paddle.reshape(labels, shape=[-1, 1])
        # 交叉熵损失,没有减margin
        loss = F.cross_entropy(input=cosine_sim, label=labels)
        # (n*2,d)
        mlp_cls_embedding = paddle.concat(
            [mlp_query_cls_embedding, mlp_key_cls_embedding], axis=0)
        return loss, mlp_cls_embedding

# 判别器的任务是区分输入文本中的token是原始文本中的一部分（fixed），还是被生成器替换的（replaced）。
# 这要求判别器能够捕捉到文本中的语义变化，并据此做出判断。
class Discriminator(nn.Layer): # 判别器

    def __init__(self, ptm_model_name):
        super(Discriminator, self).__init__()
        self.ptm = CustomErnie.from_pretrained(ptm_model_name)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                labels,
                cls_input,
                token_type_ids=None,
                attention_mask=None):
        # cls_input:这个不常见,是query和title语义表示(n*2,d)
        sequence_output, _ = self.ptm(input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask,
                                      cls_input=cls_input)
        # 判别器的输入包括 e_inputs（生成器生成的tokens）、e_labels（被替换且非填充的位置标记），
        # 以及 cls_input（一个额外的语义表示,是查询和标题的联合表示）。判别器的目标是区分输入中的
        # 哪些部分是真实的，哪些部分是生成的或经过修改的。
        pred_scores = self.classifier(sequence_output) #(n,s,2)
        loss = F.cross_entropy(input=pred_scores, label=labels)
        # rtd_loss 可能是判别器为了区分真实和生成/修改数据而计算的损失，而 prediction是对
        # 输入中每个位置是真实还是生成的预测。
        return loss, pred_scores.argmax(-1) # (n,s)


class DiffCSE(nn.Layer):

    def __init__(self,
                 encoder_name,
                 generator_name,
                 discriminator_name,
                 enc_tokenizer,
                 gen_tokenizer,
                 dis_tokenizer,
                 temp=0.05,
                 output_emb_size=32,
                 mlm_probability=0.15,
                 lambda_weight=0.15):
        super(DiffCSE, self).__init__()
        self.encoder_name = encoder_name
        self.generator_name = generator_name
        self.discriminator_name = discriminator_name
        self.enc_tokenizer = enc_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.dis_tokenizer = dis_tokenizer
        self.temp = temp
        self.output_emb_size = output_emb_size
        self.mlm_probability = mlm_probability
        self.lambda_weight = lambda_weight
        # 提取器
        self.encoder = Encoder(encoder_name,
                               temp=temp,
                               output_emb_size=output_emb_size)
        # 生成器
        # 在这个模型中，生成器不仅用于训练自身的生成能力，还间接地帮助编码器学习更好的语义表示。因为生成器在预
        # 测被掩码token时，需要利用编码器提取的语义特征。与判别器的关系：生成器的输出（pred_tokens）随后被用作判别
        # 器的输入之一，以评估这些生成的token在多大程度上保留了原始输入的语义信息。
        self.generator = ErnieForMaskedLM.from_pretrained(generator_name)
        # 判别器
        self.discriminator = Discriminator(discriminator_name)

        self.rtd_acc = 0.0
        self.rtd_rep_acc = 0.0
        self.rtd_fix_acc = 0.0
    # 这段代码展示了如何在NLP任务中结合使用生成器和判别器，通过掩码语言模型（MLM）的方式进行文本生成，并使用判别器来评估生成文本的质量。
    # 这种架构在诸如文本生成、文本摘要、对话系统等任务中非常有用。
    def train_forward(self,
                      query_input_ids,
                      key_input_ids,
                      query_token_type_ids=None,
                      key_token_type_ids=None,
                      query_attention_mask=None,
                      key_attention_mask=None):
        # 这行代码调用了编码器模型，传入查询（query）和键（key）的输入ID、token类型ID和注意力掩码。
        # 编码器计算了CL损失（可能包括对比损失或其他类型的损失），并返回了一个多层感知机（MLP）的分类嵌入（mlp_cls_embedding）
        # ，这个嵌入可能用于后续的任务，如分类或进一步处理。
        loss, mlp_cls_embedding = self.encoder(
            query_input_ids,
            key_input_ids,
            query_token_type_ids=query_token_type_ids,
            key_token_type_ids=key_token_type_ids,
            query_attention_mask=query_attention_mask,
            key_attention_mask=key_attention_mask)
        # 使用paddle.no_grad()上下文管理器来禁用梯度计算，这是因为在推理或评估模式下不需要梯度。
        with paddle.no_grad():
            # 将查询和键的输入ID合并，然后调用mask_tokens函数随机掩码一些token，为生成器模型准备输入。
            # (n*2,s)
            input_ids = paddle.concat([query_input_ids, key_input_ids], axis=0)
            # 条件性编码：如果生成器（generator）和编码器（encoder）不是同一个模型，则使用encode_by_generator方法对
            # 合并后的输入进行编码。这可能是为了将输入转换为生成器模型可以理解的格式
            if self.encoder_name != self.generator_name:
                input_ids = self.encode_by_generator(input_ids)
            # 处理注意力掩码：类似地，将查询和键的注意力掩码（attention_mask）合并。(n*2,s)
            attention_mask = paddle.concat(
                [query_attention_mask, key_attention_mask], axis=0)
            # 掩码语言模型（MLM）输入：通过mask_tokens函数随机掩码一些输入token，并准备这些掩码后
            # 的输入用于生成器预测被掩码的token。mlm_input_ids带遮挡的输入
            mlm_input_ids, _ = mask_tokens(input_ids,
                                           self.gen_tokenizer,
                                           mlm_probability=self.mlm_probability)
            # 生成预测：使用生成器模型对掩码后的输入进行预测，并通过argmax获取最可能的token索引。
            # 贪婪搜索,通过mask_tokens函数，随机选择input_ids中的一部分token进行掩码（即替换为特殊的掩码token），
            # 然后生成器（generator）基于这些掩码后的输入尝试预测这些被掩码的token。pred_tokens是生成器对这些掩码位置的预测结果。
            pred_tokens = self.generator(
                mlm_input_ids, attention_mask=attention_mask).argmax(axis=-1)
        # ddetach()用于生成pred_tokens的副本，但不计算梯度，这通常在需要将数据传递给不需要梯度回传的部分时使用
        pred_tokens = pred_tokens.detach()
        # 条件性解码：如果生成器（generator）和判别器（discriminator）不是同一个模型，则使用encode_by_discriminator
        # 方法对预测结果进行编码，以及对原始输入进行编码，以便判别器可以处理它们
        if self.generator_name != self.discriminator_name:
            pred_tokens = self.encode_by_discriminator(pred_tokens)
            input_ids = self.encode_by_discriminator(input_ids)
        # 准备判别器输入：为判别器准备输入，包括设置预测token序列的第一个token为判别器tokenizer的CLS
        #（分类）token ID,使用注意力掩码处理输入，以忽略填充部分。
        # 第一个token_ids设置成判别器的cls_id
        pred_tokens[:, 0] = self.dis_tokenizer.cls_token_id
        e_inputs = pred_tokens * attention_mask # 去掉填充
        replaced = pred_tokens != input_ids # 被替换的位置,是布尔张量,1表示被替换位置
        # 被遮挡和替换的位置
        e_labels = paddle.cast(replaced, dtype="int64") * attention_mask # 非填充替换位置
        # 计算判别器损失：调用判别器模型，传入生成器生成的tokens,真实被替换位置,
        # 以及mlp_cls_embedding(n*2,d)query和title联合语义表示,计算并返回损失和预测结果。
        # 判别器的预测结果是哪些是被替换的token,哪些是原来的token
        rtd_loss, prediction = self.discriminator(e_inputs,
                                                  e_labels,
                                                  cls_input=mlp_cls_embedding)
        # 更新总损失：将判别器的损失乘以一个权重（self.lambda_weight）后加到总损失上。
        loss = loss + self.lambda_weight * rtd_loss
        rep = (e_labels == 1) * attention_mask
        fix = (e_labels == 0) * attention_mask
        # 这个变量计算了判别器在被替换位置上的准确率。通过将判别器的预测（prediction）与 rep 掩码相乘，然后求和并除以 rep 
        # 中1的总数（即被替换且有效的位置数），我们得到了这些位置上的平均准确率。
        self.rtd_rep_acc = float((prediction * rep).sum() / rep.sum())
        # 这个变量试图计算判别器在未替换位置上的准确率，但计算方法略有不同。通常，我们会期望直接使用 (prediction * fix).sum() 
        # / fix.sum() 来计算。然而，这里的代码使用了 1.0 - ... 的形式，这实际上是在计算错误率，并假设“正确”的预测是在这些位置上
        # 判别器预测为0
        self.rtd_fix_acc = float(1.0 - (prediction * fix).sum() / fix.sum())
        # 这个变量计算了判别器在所有非填充位置上的整体准确率。通过将 prediction 与 e_labels 的逐元素比较结果
        # （即两者是否相等）与 attention_mask 相乘，然后求和并除以 attention_mask 中1的总数（即所有非填充的
        # 位置数），我们得到了整体的平均准确率。
        self.rtd_acc = float(((prediction == e_labels) * attention_mask).sum() /
                             attention_mask.sum())

        return loss, rtd_loss

    def encode_by_generator(self, batch_tokens):
        new_tokens = []
        for one_tokens in batch_tokens: # (n*2,s)
            # 用编码器tokenizer先转换成token
            one_gen_tokens = self.enc_tokenizer.convert_ids_to_tokens(
                one_tokens.tolist())
            # 再用生成器tokenizer转换token为ids
            new_tokens.append(
                self.gen_tokenizer.convert_tokens_to_ids(one_gen_tokens))
        # 返回input_ids
        return paddle.to_tensor(new_tokens)

    def encode_by_discriminator(self, batch_tokens):
        new_tokens = []
        for one_tokens in batch_tokens:
            # 用生成器tokenizer转换成tokens
            one_gen_tokens = self.gen_tokenizer.convert_ids_to_tokens(
                one_tokens.tolist())
            # 用判别器tokenizer转换上面的tokens
            new_tokens.append(
                self.dis_tokenizer.convert_tokens_to_ids(one_gen_tokens))

        return paddle.to_tensor(new_tokens)

    def test_forward(self,
                     query_input_ids,
                     key_input_ids,
                     query_token_type_ids=None,
                     key_token_type_ids=None,
                     query_attention_mask=None,
                     key_attention_mask=None):

        # compute cosine similarity for query and key text
        cos_sim = self.encoder.cosine_sim(
            query_input_ids,
            key_input_ids,
            query_token_type_ids=query_token_type_ids,
            key_token_type_ids=key_token_type_ids,
            query_attention_mask=query_attention_mask,
            key_attention_mask=key_attention_mask)

        return cos_sim

    def forward(self,
                query_input_ids,
                key_input_ids,
                query_token_type_ids=None,
                key_token_type_ids=None,
                query_attention_mask=None,
                key_attention_mask=None,
                mode="train"):
        if mode == "train":
            return self.train_forward(query_input_ids,
                                      key_input_ids,
                                      query_token_type_ids=query_token_type_ids,
                                      key_token_type_ids=key_token_type_ids,
                                      query_attention_mask=query_attention_mask,
                                      key_attention_mask=key_attention_mask)
        else:
            return self.test_forward(query_input_ids,
                                     key_input_ids,
                                     query_token_type_ids=query_token_type_ids,
                                     key_token_type_ids=key_token_type_ids,
                                     query_attention_mask=query_attention_mask,
                                     key_attention_mask=key_attention_mask)