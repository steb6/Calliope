import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import copy
from config import config
from torch.autograd import Variable


def ifn(x, i):
    return x[i] if x is not None else None


class CompressiveEncoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 ff_mul=config["model"]["ff_mul"],
                 ff_dropout=config["model"]["ff_dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 ):
        super(CompressiveEncoder, self).__init__()

        c = copy.deepcopy

        if config["train"]["use_rel_pos"]:
            self_attn = RelMultiHeadedAttention(heads, d_model, dropout=0.1)
        else:
            self_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)

        ff = FeedForward(d_model, ff_mul, dropout=ff_dropout)

        encoder = Encoder(EncoderLayer(c(self_attn), c(ff)), layers, vocab_size, d_model)
        self.drums_encoder = c(encoder)
        self.bass_encoder = c(encoder)
        self.guitar_encoder = c(encoder)
        self.strings_encoder = c(encoder)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, mask):
        d_z = self.drums_encoder(seq[0], ifn(mask, 0))
        b_z = self.bass_encoder(seq[1], ifn(mask, 1))
        g_z = self.guitar_encoder(seq[2], ifn(mask, 2))
        s_z = self.strings_encoder(seq[3], ifn(mask, 3))
        latents = torch.stack([d_z, b_z, g_z, s_z], dim=1)
        return latents


class CompressiveDecoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 ff_mul=config["model"]["ff_mul"],
                 ff_dropout=config["model"]["ff_dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 ):
        super(CompressiveDecoder, self).__init__()

        c = copy.deepcopy

        if config["train"]["use_rel_pos"]:
            self_attn = RelMultiHeadedAttention(heads, d_model, dropout=0.1)
        else:
            self_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)

        src_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)
        ff = FeedForward(d_model, ff_mul, dropout=ff_dropout)

        decoder = Decoder(DecoderLayer(c(self_attn), c(src_attn), c(ff)), layers, vocab_size, d_model)

        self.drums_decoder = c(decoder)
        self.bass_decoder = c(decoder)
        self.guitar_decoder = c(decoder)
        self.strings_decoder = c(decoder)
        self.generator = Generator(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, trg, src_mask, trg_mask, latent):
        d_out = self.drums_decoder(trg[0], ifn(src_mask, 0), ifn(trg_mask, 0), latent[0])
        b_out = self.bass_decoder(trg[1], ifn(src_mask, 1), ifn(trg_mask, 1), latent[1])
        g_out = self.guitar_decoder(trg[2], ifn(src_mask, 2), ifn(trg_mask, 2), latent[2])
        s_out = self.strings_decoder(trg[3], ifn(src_mask, 3), ifn(trg_mask, 3), latent[3])
        output = torch.stack([d_out, b_out, g_out, s_out], dim=0)
        output = self.generator(output)
        return output


class Encoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N

        max_klen = config["model"]["seq_len"]  # TODO increase it when adding memory
        max_klen += config["model"]["mem_len"] + config["model"]["cmem_len"]
        n_head = config["model"]["heads"]
        d_head = config["model"]["d_model"] // config["model"]["heads"]
        self.r_emb = nn.Parameter(torch.Tensor(N, n_head, max_klen, d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(N, n_head, d_head))
        self.r_bias = nn.Parameter(torch.Tensor(N, n_head, max_klen))

        self.pos = PositionalEncoding(d_model)

    def forward(self, seq, mask):
        seq = self.embed(seq)

        if not config["train"]["use_rel_pos"]:
            seq = self.pos(seq)

        i = 0
        for layer in self.layers:
            seq = layer(seq, mask, self.r_emb[i], self.r_w_bias[i], self.r_bias[i])
            i += 1
        return seq


class Decoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N

        max_klen = config["model"]["seq_len"]  # TODO increase it when adding memory
        max_klen += config["model"]["mem_len"] + config["model"]["cmem_len"]
        n_head = config["model"]["heads"]
        d_head = config["model"]["d_model"] // config["model"]["heads"]
        self.r_emb = nn.Parameter(torch.Tensor(N, n_head, max_klen, d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(N, n_head, d_head))
        self.r_bias = nn.Parameter(torch.Tensor(N, n_head, max_klen))

        self.pos = PositionalEncoding(d_model)

    def forward(self, trg, src_mask, trg_mask, latent):
        trg = self.embed(trg)

        if not config["train"]["use_rel_pos"]:
            trg = self.pos(trg)

        i = 0
        for layer in self.layers:
            trg = layer(trg, src_mask, trg_mask, latent, self.r_emb[i], self.r_w_bias[i], self.r_bias[i])
            i += 1
        return trg


class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(config["model"]["d_model"])
        self.norm2 = nn.LayerNorm(config["model"]["d_model"])

    def forward(self, x, input_mask, pos_emb, r_w_bias, r_bias):
        out = self.self_attn(x, key=x, value=x, mask=input_mask, r_emb=pos_emb, r_w_bias=r_w_bias, r_bias=r_bias)
        x = self.norm1(x + out)
        out = self.feed_forward(x)
        x = self.norm2(x + out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(config["model"]["d_model"])
        self.norm2 = nn.LayerNorm(config["model"]["d_model"])
        self.norm3 = nn.LayerNorm(config["model"]["d_model"])

    def forward(self, x, src_mask, trg_mask, latent, pos_emb, r_w_bias, r_bias):
        out = self.self_attn(x, key=x, value=x, mask=trg_mask, r_emb=pos_emb, r_w_bias=r_w_bias, r_bias=r_bias)
        x = self.norm1(x + out)
        out = self.src_attn(x, key=latent, value=latent, mask=src_mask)
        x = self.norm2(x + out)
        out = self.feed_forward(x)
        x = self.norm3(x + out)
        return x


class RelMultiHeadedAttention(nn.Module):
    """
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L293
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(RelMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_out = d_model // h
        self.h = h
        self.linears = (clones(nn.Linear(d_model, d_model, bias=False), 4))  # TODO bias or not?
        self.dropout = nn.Dropout(p=dropout)
        self.scale = 1 / (h ** 0.5)
        self.seq_len = config["model"]["seq_len"]
        self.mem_len = config["model"]["mem_len"]
        self.cmem_len = config["model"]["cmem_len"]
        self.reconstruction_attn_dropout = nn.Dropout(config["model"]["reconstruction_attn_dropout"])
        self.attn = None

    def forward(self, query, key=None, value=None, mask=None, r_emb=None, r_w_bias=None, r_bias=None):
        """
        :param memories:
        :param query: batch_size x q_len x d_model
        :param key: batch_size x k_len x d_model
        :param value: batch_size x v_len x d_model
        :param mask: batch_size x q_len ( x v_len)
        :param r_emb: n_head x max_klen x d_head
        :param r_w_bias: n_head x d_head
        :param r_bias: n_head x max_klen
        :return: batch_size x q_len x d_model
        """
        n_batches = query.size(0)

        k_len = key.size(1)

        # batch_size x n_head x seq_len x d_head
        q, k, v = [l(x).view(n_batches, -1, self.h, self.d_out).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        r_emb = r_emb[:, -k_len:, :]  # TODO try this (first or last?)
        r_bias = r_bias[:, -k_len:]  # TODO try this (first or last?)

        r_query = q + r_w_bias[None, :, None, :]

        AC = torch.einsum('bnid,bnjd->bnij', r_query, k)
        B_ = torch.einsum('bnid,njd->bnij', q, r_emb)  # r_emb_pe must be 3 4 200
        D_ = r_bias[None, :, None]
        BD = shift(B_ + D_)  # TODO place original shift

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        if mask is not None:  # TODO expand mask
            if mask.dim() == 2:  # transform linear mask to square mask
                mask = mask[:, :, None] * mask[:, None, :]
            mask = mask.unsqueeze(1)  # apply same mask to all heads
            attn_score = attn_score.masked_fill(~mask, -1e9)  # TODO empty row becomes 0.005 is it good?

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        self.attn = attn_prob.detach()

        attn_vec = torch.einsum('bnij,bnjd->bnid', attn_prob, v)
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_out)

        attn_out = self.linears[-1](attn_vec)

        return attn_out


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_out = d_model // h
        self.h = h
        self.linears = (clones(nn.Linear(d_model, d_model, bias=False), 4))  # TODO bias or not?
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key=None, value=None, mask=None, pos_emb=None, r_emb=None, r_w_bias=None, r_bias=None):
        if mask is not None:
            if mask.dim() == 2:  # transform linear mask to square mask
                mask = mask[:, :, None] * mask[:, None, :]
            mask = mask.unsqueeze(1)  # apply same mask to all heads
        n_batches = query.size(0)
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_out).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, weights = full_attn(query, key, value, mask=mask, dropout=self.dropout, pos_emb=pos_emb)
        self.attn = weights
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_out)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mul, dropout=0.):
        super().__init__()
        activation = nn.GELU
        self.w1 = nn.Linear(dim, dim * ff_mul)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * ff_mul, dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return self.norm(x[0]), *x[1:]


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj_drums = nn.Linear(d_model, vocab)
        self.proj_bass = nn.Linear(d_model, vocab)
        self.proj_guitar = nn.Linear(d_model, vocab)
        self.proj_strings = nn.Linear(d_model, vocab)

    def forward(self, x):
        out_drums = F.log_softmax(self.proj_drums(x[0]), dim=-1)
        out_bass = F.log_softmax(self.proj_bass(x[1]), dim=-1)
        out_guitar = F.log_softmax(self.proj_guitar(x[2]), dim=-1)
        out_strings = F.log_softmax(self.proj_strings(x[3]), dim=-1)
        out = torch.stack([out_drums, out_bass, out_guitar, out_strings], dim=0)
        return out


class ConvCompress(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride=ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        aux = self.lut(x)
        return aux * math.sqrt(self.d_model)


def full_attn(q, k, v, mask=None, dropout=None, pos_emb=None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)  # Q K^T

    if pos_emb is not None:
        # TODO remember to modify this when adding memory !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pos_emb = pos_emb[:, :q.shape[2], :]  # get pos_emb of first elements for greedy decoding
        pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * (dim ** 0.5)
        pos_dots = shift(pos_dots)  # left upper triangular has positional embedding of illegal token
        # pos_dots = pos_dots[..., :dots.shape[-1]]  # TODO select useful embedding, confirm or remove
        dots = dots + pos_dots

    if mask is not None:
        mask = mask[:, :, :dots.shape[2], :]  # during evaluation, must adapt mask to dots size
        dots = dots.masked_fill(mask == 0, -1e9)  # same mask for all heads
    attn = dots.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v), attn  # (Q K^T) V


def clones(module, N):
    """ Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def shift(x):
    """
    It skews the matrix x, as done in Relative Local Attention of Music Transformer
    0, 0, a     a, 0, 0
    0, b, c =>  b, c, 0
    d, e, f     d, e, f
    """
    *_, i, j = x.shape  # i=4, j=12
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)  # i=4, j=26
    l = i + j - 1  # 20
    x = x.view(*_, -1)  #
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]


def to(t):
    return {'dtype': t.dtype, 'device': t.device}


# TODO *****************************************************************************************************************
# REMOVE USELESS
# TODO *****************************************************************************************************************

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    left = (*pre_slices, slice(None, index))
    right = (*pre_slices, slice(index, None))
    return t[left], t[right]


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class MyMemoryAttention(nn.Module):
    def __init__(self, h, dim, seq_len, mem_len, cmem_len, cmem_ratio, attn_dropout=0.1,
                 reconstruction_attn_dropout=0.1):
        super(MyMemoryAttention, self).__init__()
        assert dim % h == 0
        self.dim_head = dim // h
        self.h = h
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)  # 1/root(dim_head)
        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)
        self.multi_head_attention = MultiHeadedAttention(h, dim, attn_dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, h, memories=None, input_mask=None, pos_emb=None):
        # Prepare mask
        if input_mask is not None:
            if input_mask.dim() == 2:  # encoder mask, cover just pad
                input_mask = input_mask[:, :, None] * input_mask[:, None, :]
            input_mask = F.pad(input_mask, (self.cmem_len + self.mem_len, 0), value=True)
        # Algorithm from paper
        m, cm = memories
        mem = torch.cat((cm, m, h), dim=1)  # TODO x too?
        a, weights = self.multi_head_attention(h, key=mem, value=mem, mask=input_mask, pos_emb=pos_emb)
        a = self.norm1(a + h)
        old_mem = m[:, :self.seq_len, :]
        new_cm = self.compress_mem_fn(old_mem)
        m = torch.cat((m, h), dim=1)[:, -self.mem_len:, :]
        cm = torch.cat((cm, new_cm), dim=1)[:, -self.cmem_len:, :]
        h = a
        # Attention reconstruction
        h_copy = h.detach().clone()
        old_mem = torch.detach(old_mem)
        Q = torch.detach(self.multi_head_attention.linears[0].weight.data)
        K = torch.detach(self.multi_head_attention.linears[1].weight.data)
        V = torch.detach(self.multi_head_attention.linears[2].weight.data)

        def attn(hh, mm):
            n_batches = hh.shape[0]
            hQ = torch.matmul(hh, Q).view(n_batches, -1, self.h, self.dim_head).transpose(1, 2)
            mK = torch.matmul(mm, K).view(n_batches, -1, self.h, self.dim_head).transpose(1, 2)
            mV = torch.matmul(mm, V).view(n_batches, -1, self.h, self.dim_head).transpose(1, 2)
            attention, _ = full_attn(hQ, mK, mV, dropout=self.reconstruction_attn_dropout)
            return attention

        new_cm = self.compress_mem_fn(old_mem)
        l_attn = F.mse_loss(attn(h_copy, old_mem), attn(h_copy, new_cm))

        return h, m, cm, l_attn, weights
