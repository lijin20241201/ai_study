import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CrossEntropyCriterion(nn.Layer):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()
    def forward(self, predict, label, trg_mask):
        # reduction='none'会让cost形状和predict一致
        cost = F.cross_entropy(input=predict,
                               label=label,
                               reduction='none',
                               soft_label=False)
        cost = paddle.squeeze(cost, axis=[2])
        masked_cost = cost * trg_mask # 计算非填充token损失
        # 平均样本损失,因为在样本轴平均
        batch_mean_cost = paddle.mean(masked_cost, axis=[0]) 
        # 把一个序列中所有token的损失聚合相加
        seq_cost = paddle.sum(batch_mean_cost) 
        return seq_cost

class Seq2SeqEncoder(nn.Layer):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        # lstm编码部分
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size, # lstm中隐层大小
                            num_layers=num_layers,
                            dropout=0.2 if num_layers > 1 else 0.)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence) # (n,s,d)
        # 得到整个序列的输出和encoder_state(隐藏状态和细胞状态)
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)
        return encoder_output, encoder_state
# 这个注意力层主要用于在序列到序列（Seq2Seq）模型中，特别是在解码器部分，来关注编码器输出
# 的相关部分
# 这个注意力层的设计允许解码器在生成每个输出时，能够关注编码器输出的不同部分，从而提高模型处理
# 长序列和捕获重要信息的能力。
class AttentionLayer(nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # 一个线性层（nn.Linear），用于将编码器输出（encoder_output）投影到一个新的空间，
        # 以便与解码器的隐藏状态（hidden）进行匹配。
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        # 另一个线性层，用于将注意力加权后的编码器输出和解码器的隐藏状态拼接后的结果投影回hidden_size维度。这有助于将注意力机制
        # 和解码器的隐藏状态结合起来，生成最终的注意力层输出。
        self.output_proj = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        # hidden：解码器在当前时间步的隐藏状态，其形状通常为[batch_size, hidden_size]。
        # encoder_output：编码器的输出，其形状通常为[batch_size, seq_len, hidden_size]，
        # 其中seq_len是编码器序列的长度。
        # encoder_padding_mask：一个可选的填充掩码，用于指示encoder_output中的哪些位置是填充的
        # 其形状通常为[batch_size, seq_len]，其中填充位置通常为负无穷大
        # 首先，通过self.input_proj对encoder_output进行线性变换，以准备与hidden进行注意力分数的计算
        encoder_output = self.input_proj(encoder_output)
        # 计算注意力分数。通过将hidden增加一个维度（使用paddle.unsqueeze），使其形状变为[batch_size, 1, hidden_size]，
        # 以便与encoder_output进行矩阵乘法。矩阵乘法的结果attn_scores的形状为[batch_size, 1, seq_len]，表示每个解码器
        # 隐藏状态与编码器每个输出的匹配程度。
        # 如果提供了encoder_padding_mask，则将其添加到attn_scores上，以确保填充位置在softmax
        # 之后被忽略
        attn_scores = paddle.matmul(paddle.unsqueeze(hidden, [1]),
                                    encoder_output,
                                    transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)
        # 应用softmax函数到attn_scores上，得到注意力权重。
        attn_scores = F.softmax(attn_scores)
        # 使用注意力权重对encoder_output进行加权求和，得到注意力加权后的输出attn_out。
        attn_out = paddle.squeeze(paddle.matmul(attn_scores, encoder_output),
                                  [1])
        # 将attn_out与hidden拼接起来，以便将解码器的隐藏状态和注意力信息结合起来。
        attn_out = paddle.concat([attn_out, hidden], 1)
        # 最后，通过self.output_proj对拼接后的结果进行线性变换，得到最终的注意力层输出。
        attn_out = self.output_proj(attn_out)
        return attn_out

# 自定义的RNN单元，该单元结合了LSTM和注意力机制。
class Seq2SeqDecoderCell(nn.RNNCellBase):
    def __init__(self, num_layers, input_size, hidden_size):
        super(Seq2SeqDecoderCell, self).__init__()
        self.dropout = nn.Dropout(0.2)
        # 一个nn.LayerList，包含num_layers个nn.LSTMCell实例。对于第一层，输入尺寸是input_size 
        # + hidden_size（如果包含输入馈送），对于其他层，输入尺寸是hidden_size。
        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(input_size=input_size +
                        hidden_size if i == 0 else hidden_size,
                        hidden_size=hidden_size) for i in range(num_layers)
        ])
        # attention_layer：一个自定义的注意力层（AttentionLayer），用于计算解码器输出和编码器输出
        # 之间的注意力权重。
        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        # step_input：当前时间步的输入（通常是词嵌入）
        # states：LSTM的状态，包括隐藏状态和单元状态。另外，这里还隐式地包含了输入馈送
        # （input_feed），这是通过修改states参数来实现的。
        # encoder_output：编码器的输出，用于注意力机制。
        # encoder_padding_mask：编码器输出的填充掩码，用于注意力机制中忽略填充部分
        lstm_states, input_feed = states
        new_lstm_states = []
        # 首先，将当前时间步的输入step_input与输入馈送input_feed（如果存在）拼接起来。
        # 这通常用于帮助解码器记住之前的上下文信息。
        step_input = paddle.concat([step_input, input_feed], 1)
        # 遍历lstm_cells列表，对每个LSTM单元执行前向传播。在每个时间步，LSTM的输出都会通过Dropout
        # 层以减少过拟合。
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            step_input = self.dropout(out)
            new_lstm_states.append(new_lstm_state)
        # 将LSTM的最终输出传递给注意力层，计算与编码器输出的注意力权重，并返回注意力加权后的输出以
        # 及更新后的状态（这里状态被更新为新的LSTM状态和注意力层的输出）
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]

class Seq2SeqDecoder(nn.Layer):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim) # 词嵌入层
        # Seq2SeqDecoderCell是一个自定义的RNN单元，它封装了LSTM和注意力机制的功能，并且能够处理额外
        # 的encoder_output和encoder_padding_mask参数
        self.lstm_attention = nn.RNN(
            Seq2SeqDecoderCell(num_layers, embed_dim, hidden_size))
        #一个线性层，将RNN的输出转换为词汇表大小的向量，用于预测下一个词的索引。
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    # 目标序列（通常是训练时的教师强制输入或推理时的部分已知序列）。
    # 解码器的初始状态，通常包括LSTM的隐藏状态和单元状态（对于多层LSTM，则是一个状态列表）。
    # 编码器的输出，用于注意力机制,编码器输出的填充掩码，用于在注意力机制中忽略填充部分
    def forward(self, trg, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(trg) # 通过词嵌入层将trg转换为词嵌入向量。
        # decoder_output
        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        # 将decoder_output递给输出层，以获得词汇表大小的分数向量
        predict = self.output_layer(decoder_output)
        return predict

# 序列到序列模型，通过编码器（Encoder）和解码器（Decoder）架构，并在解码器中引入了注意力机制
# 来改进模型的性能。
class Seq2SeqAttnModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size, # LSTM或其他RNN单元中隐藏层的维度。
                 num_layers, # RNN堆叠的层数。
                 pad_id=1): # 填充id
        super(Seq2SeqAttnModel, self).__init__()
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
    # 源序列（输入序列）的Tensor。源序列中每个序列的实际长度（用于处理变长序列）
    # 目标序列（输出序列）的Tensor，通常用于训练时的教师强制
    def forward(self, src, src_length, trg):
        # 通过编码器处理源序列，得到编码器的输出encoder_output和最终状态encoder_final_state。
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        # 将编码器的最终状态encoder_final_state转换成适合解码器初始状态的格式。这是因为解码器可能需
        # 要多个层的状态，以及可能还需要额外的输入（如注意力机制的初始状态）。
        # Transfer shape of encoder_final_states to [num_layers, 2, batch_size, hidden_size]
        encoder_final_states = [(encoder_final_state[0][i],
                                 encoder_final_state[1][i])
                                for i in range(self.num_layers)]
        # Construct decoder initial states: use input_feed and the shape is
        # [[h,c] * num_layers, input_feed], consistent with Seq2SeqDecoderCell.states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # 构建了一个注意力掩码encoder_padding_mask，用于在注意力机制中忽略源序列中的填充部分。
        src_mask = (src != self.pad_id).astype(paddle.get_default_dtype())
        # 通过将源序列中的非结束符位置设置为0（或接近0的值），而结束符和填充位置设置为负无穷（或非常大的负值），
        # 从而在注意力分数的softmax操作中降低这些位置的权重。
        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
        # 最后，使用解码器、解码器的初始状态、编码器的输出和注意力掩码，通过解码器生成预测序列predict。
        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)

        return predict

class Seq2SeqAttnInferModel(Seq2SeqAttnModel):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers
        super(Seq2SeqAttnInferModel,
              self).__init__(vocab_size, embed_dim, hidden_size, num_layers,
                             eos_id)

        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        encoder_final_state = [(encoder_final_state[0][i],
                                encoder_final_state[1][i])
                               for i in range(self.num_layers)]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.pad_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return seq_output