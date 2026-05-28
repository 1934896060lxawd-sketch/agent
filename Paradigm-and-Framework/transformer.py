import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    使用正弦和余弦函数生成固定的位置编码，不参与训练。
    参数:
        d_model: 模型维度（词嵌入向量的长度）
        dropout: dropout 概率
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        # 计算 div_term: 用于调整不同维度的频率，形状为 (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe 的大小为 (max_len, d_model)，初始全 0
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer：不会被视为模型参数，但会随模型移动到 GPU 等
        # 增加 batch 维度，形状变为 (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (batch_size, seq_length, d_model)
        取出与输入序列长度相同的位置编码，直接加到词嵌入上
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    参数:
        d_model  : 模型的维度（每个词向量的长度）
        num_heads: 注意力头的个数
    要求: d_model 能被 num_heads 整除，这样才能将向量均匀切分给每个头
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model          # 总维度，例如 512
        self.num_heads = num_heads      # 头数，例如 8
        self.d_k = d_model // num_heads # 每个头分到的维度，例如 64

        # 定义四个线性变换矩阵（可学习参数）
        # W_q: 将输入映射成 Query（查询）
        # W_k: 将输入映射成 Key（键）
        # W_v: 将输入映射成 Value（值）
        # W_o: 对合并后的多头输出做最后一次线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入形状: (batch_size, num_heads, seq_length, d_k)
        mask 中值为 0 的位置会被忽略（权重变成极小值）
        返回形状: (batch_size, num_heads, seq_length, d_k)
        """
        # 1. 计算 Q 与 K 的点积相似度，并除以 sqrt(d_k) 进行缩放
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 如果提供了掩码，将掩码为 0 的位置设成一个非常大的负数
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 在最后一个维度上做 softmax，得到注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 用权重对 V 进行加权求和，得到上下文表示
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入切分成多个头
        输入: (batch_size, seq_length, d_model)
        输出: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        # view 将 d_model 拆成 num_heads * d_k，然后交换维度，使 heads 成为第二维
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多头输出合并回原始维度
        输入: (batch_size, num_heads, seq_length, d_k)
        输出: (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        # 先交换维度 1 和 2，然后使用 contiguous 保证内存连续，最后合并最后两个维度
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Q, K, V 形状: (batch_size, seq_length, d_model)
        输出形状: (batch_size, seq_length, d_model)
        """
        # 1. 线性变换 + 分头
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头 + 最终线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块 (Position-wise Feed-Forward Network)
    对每个位置独立地应用两层全连接网络。
    参数:
        d_model: 输入输出维度
        d_ff   : 中间隐藏层维度（通常是 d_model 的 4 倍）
        dropout: dropout 概率
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层线性变换，将维度升到 d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层线性变换，降回 d_model
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x 形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer 编码器中的一层
    包含：
        1. 多头自注意力子层（带残差连接和层归一化）
        2. 前馈网络子层（带残差连接和层归一化）
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)       # 自注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout) # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一个子层后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个子层后的层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x   : 输入序列 (batch_size, seq_len, d_model)
        mask: 注意力掩码，编码器通常使用 padding mask 避免关注填充符
        """
        # 1. 多头自注意力 + 残差 + 归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Transformer 解码器中的一层
    包含：
        1. 掩码多头自注意力子层（防止关注未来信息）
        2. 交叉注意力子层（关注编码器输出）
        3. 前馈网络子层
    每个子层都带有残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)       # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads)      # 交叉注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout) # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x              : 解码器输入 (batch_size, tgt_len, d_model)
        encoder_output : 编码器输出 (batch_size, src_len, d_model)
        src_mask       : 源序列的掩码（如 padding mask）
        tgt_mask       : 目标序列的掩码（如下三角矩阵，防止关注未来位置）
        """
        # 1. 掩码自注意力 + 残差 + 归一化
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力（Q来自解码器，K,V来自编码器）+ 残差 + 归一化
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x