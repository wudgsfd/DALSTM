import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Feed_Forward_block(nn.Module):
    """前馈神经网络模块"""

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ffn_in):
        return self.layer2(self.dropout(F.relu(self.layer1(ffn_in))))


class Encoder_block(nn.Module):
    """SAINT模型编码器模块"""

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)  # 习题嵌入
        self.embd_cat = nn.Embedding(total_cat, embedding_dim=dim_model)  # 类别嵌入
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # 位置嵌入

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, in_ex, in_cat, first_block=True):
        # 处理嵌入
        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = self.embd_cat(in_cat)
            out = in_ex + in_cat  # 合并嵌入
        else:
            out = in_ex

        # 添加位置嵌入
        in_pos = get_pos(self.seq_len).to(out.device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)  # (n, b, d)

        # 多头注意力
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n).to(out.device))
        out = self.dropout(out) + skip_out

        # 前馈网络
        out = out.permute(1, 0, 2)  # (b, n, d)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout(out) + skip_out

        return out


class Decoder_block(nn.Module):
    """SAINT模型解码器模块"""

    def __init__(self, dim_model, total_in, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embd_in = nn.Embedding(total_in, embedding_dim=dim_model)  # 交互嵌入
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # 位置嵌入

        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de)
        self.ffn_en = Feed_Forward_block(dim_model)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, in_in, en_out, first_block=True):
        # 处理嵌入
        if first_block:
            in_in = self.embd_in(in_in)
            out = in_in
        else:
            out = in_in

        # 添加位置嵌入
        in_pos = get_pos(self.seq_len).to(out.device)
        in_pos = self.embd_pos(in_pos)
        out = out + in_pos

        out = out.permute(1, 0, 2)  # (n, b, d)
        n, _, _ = out.shape

        # 第一个多头注意力
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n).to(out.device))
        out = self.dropout(out) + skip_out

        # 第二个多头注意力（与编码器交互）
        en_out = en_out.permute(1, 0, 2)  # (b, n, d) -> (n, b, d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n).to(out.device))
        out = self.dropout(out) + skip_out

        # 前馈网络
        out = out.permute(1, 0, 2)  # (b, n, d)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout(out) + skip_out

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    """生成上三角掩码，防止未来信息泄露"""
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))


def get_pos(seq_len):
    """生成位置索引"""
    return torch.arange(seq_len).unsqueeze(0)


class SAINT(nn.Module):
    """SAINT模型主类"""

    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat,
                 total_in, heads_de, seq_len, output_size):
        super().__init__()
        self.num_en = num_en
        self.num_de = num_de
        self.seq_len = seq_len

        # 创建编码器和解码器层
        self.encoder = get_clones(Encoder_block(dim_model, heads_en, total_ex,
                                                total_cat, seq_len), num_en)
        self.decoder = get_clones(Decoder_block(dim_model, total_in, heads_de,
                                                seq_len), num_de)

        # 输出层
        self.out = nn.Linear(in_features=dim_model, out_features=output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # 从输入中提取SAINT模型需要的组件
        # x的结构: [id_assessment, id_student, date_submitted, score]
        # 我们将使用id_assessment作为习题特征，id_student作为类别特征，score二值化作为交互特征

        # 提取习题ID (已归一化，需要转换回整数索引)
        ex_ids = (x[:, :, 0] * (self.total_ex - 1)).long()  # 反归一化
        # 提取学生ID作为类别特征
        cat_ids = (x[:, :, 1] * (self.total_cat - 1)).long()
        # 将分数二值化作为交互特征 (0或1)
        in_ids = (x[:, :, 3] > 0.5).long()  # 大于0.5视为正确

        # 编码器前向传播
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            ex_ids = self.encoder[i](ex_ids, cat_ids, first_block=first_block)
            cat_ids = ex_ids  # 传递给下一个编码器块

        # 解码器前向传播
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_ids = self.decoder[i](in_ids, en_out=ex_ids, first_block=first_block)

        # 取最后一个时间步的输出并预测
        out = self.out(in_ids[:, -1, :])
        return self.logsoftmax(out)

    def set_vocab_sizes(self, total_ex, total_cat, total_in):
        """设置词汇表大小（在初始化后使用）"""
        self.total_ex = total_ex
        self.total_cat = total_cat
        self.total_in = total_in


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.logsoftmax(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.logsoftmax(out)
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        out = self.network(x_permuted)
        out = out.transpose(1, 2)
        out = self.linear(out[:, -1, :])
        out = self.logsoftmax(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.logsoftmax(out)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.logsoftmax(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._generate_pe(max_len, d_model))

    def _generate_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self.max_len = seq_len
            self.pe = self._generate_pe(seq_len, self.d_model).to(x.device)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.5):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        output = self.out_linear(output)
        return output, attention


class CNN_LSTM_MultiHeadAttention(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5, num_heads=4):
        super(CNN_LSTM_MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_input, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=3, padding=1)
        self.res_conv = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=5, padding=2)

        self.lstm = nn.LSTM(conv_input, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.multi_head_attention = MultiHeadAttentionLayer(hidden_size * 2, num_heads, dropout_prob)
        self.positional_encoding = PositionalEncoding(hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        conv_out = self.conv1(x_permuted)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.res_conv(conv_out) + conv_out
        conv_out = self.conv3(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(conv_out, (h0, c0))

        lstm_out = self.positional_encoding(lstm_out)
        lstm_out = self.dropout(lstm_out)

        context_vector, _ = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        deep_out = context_vector[:, -1, :]
        out = self.fc(deep_out)
        out = self.logsoftmax(out)
        return out


class GraphKT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_concepts,
                 concept_emb_size=None, dropout_prob=0.5):
        super(GraphKT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_concepts = num_concepts
        self.input_size = input_size

        self.concept_emb_size = concept_emb_size if concept_emb_size is not None else input_size
        self.concept_embedding = nn.Embedding(num_concepts, self.concept_emb_size)

        self.gcn = nn.Sequential(
            nn.Linear(self.concept_emb_size, self.concept_emb_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.concept_emb_size, self.concept_emb_size)
        )

        if self.input_size != self.concept_emb_size:
            self.feature_projection = nn.Linear(self.input_size, self.concept_emb_size)
        else:
            self.feature_projection = None

        self.lstm = nn.LSTM(self.concept_emb_size * 2, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, concept_ids=None):
        batch_size, seq_len, input_feat_dim = x.size()

        if self.feature_projection is not None:
            x = self.feature_projection(x)

        if concept_ids is None:
            concept_ids = torch.randint(0, self.num_concepts, (x.size(0), x.size(1)), device=x.device)
        else:
            if concept_ids.size(1) != seq_len:
                if concept_ids.size(1) > seq_len:
                    concept_ids = concept_ids[:, :seq_len]
                else:
                    pad_len = seq_len - concept_ids.size(1)
                    pad = torch.zeros(x.size(0), pad_len, dtype=concept_ids.dtype, device=concept_ids.device)
                    concept_ids = torch.cat([concept_ids, pad], dim=1)

        concept_emb = self.concept_embedding(concept_ids)
        concept_emb = self.gcn(concept_emb)

        if x.size()[:2] != concept_emb.size()[:2]:
            raise RuntimeError(f"维度不匹配: x={x.size()}, concept_emb={concept_emb.size()}")

        x_combined = torch.cat([x, concept_emb], dim=2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x_combined, (h0, c0))

        attn_weights = F.softmax(self.attention(out), dim=1)
        out = torch.sum(out * attn_weights, dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


class BERT4KT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, max_seq_len, num_heads=4, dropout_prob=0.5):
        super(BERT4KT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.logsoftmax(out)
        return out


class SAKT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, dropout_prob=0.5):
        super(SAKT, self).__init__()
        self.hidden_size = hidden_size

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # 问题嵌入层
        self.question_embedding = nn.Linear(input_size, hidden_size)

        # 多头注意力层
        self.multi_head_attention = MultiHeadAttentionLayer(hidden_size, num_heads, dropout_prob)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # 输出层
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        q = x
        x_emb = self.input_embedding(x)
        q_emb = self.question_embedding(q)

        attn_output, _ = self.multi_head_attention(q_emb, x_emb, x_emb)
        attn_output = self.dropout(attn_output)
        out = self.layer_norm1(attn_output + q_emb)

        ff_output = self.feed_forward(out)
        ff_output = self.dropout(ff_output)
        out = self.layer_norm2(ff_output + out)

        out = out[:, -1, :]
        out = self.fc(out)
        out = self.logsoftmax(out)
        return out