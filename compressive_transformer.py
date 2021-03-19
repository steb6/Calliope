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
                 attn_layer_dropout=config["model"]["attn_layer_dropout"],
                 recon_attn_dropout=config["model"]["reconstruction_attn_dropout"],
                 ff_dropout=config["model"]["ff_dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 seq_len=config["model"]["seq_len"],
                 mem_len=config["model"]["mem_len"],
                 cmem_len=config["model"]["cmem_len"],
                 cmem_ratio=config["model"]["cmem_ratio"],
                 device=config["train"]["device"]
                 ):
        super(CompressiveEncoder, self).__init__()
        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert cmem_len >= (mem_len // cmem_ratio), f'len of cmem should be at least ' f'{int(mem_len // cmem_ratio)}' \
                                                    f' but it is ' f'{int(cmem_len)}'

        self.pos_emb = nn.Parameter(
            torch.zeros(4, heads, seq_len + mem_len + cmem_len, d_model // heads, device=device,
                        requires_grad=True))
        c = copy.deepcopy

        # self_mem_attn = Residual(PreNorm(d_model, MyMemoryAttention(heads, d_model, seq_len,
        #                                                             mem_len, cmem_len, cmem_ratio,
        #                                                             attn_dropout=attn_layer_dropout,
        #                                                             reconstruction_attn_dropout=recon_attn_dropout)))
        self_mem_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)

        ff = FeedForward(d_model, ff_mul, dropout=ff_dropout)

        encoder = Encoder(EncoderLayer(c(self_mem_attn), c(ff)), layers, vocab_size, d_model)
        self.drums_encoder = c(encoder)
        self.bass_encoder = c(encoder)
        self.guitar_encoder = c(encoder)
        self.strings_encoder = c(encoder)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, mask, mems, cmems):
        d_z, d_mem, d_cmem, d_l, daw = self.drums_encoder(seq[0], ifn(mask, 0), mems[0], cmems[0], self.pos_emb[0])
        b_z, b_mem, b_cmem, b_l, baw = self.bass_encoder(seq[1], ifn(mask, 1), mems[1], cmems[1], self.pos_emb[1])
        g_z, g_mem, g_cmem, g_l, gaw = self.guitar_encoder(seq[2], ifn(mask, 2), mems[2], cmems[2], self.pos_emb[2])
        s_z, s_mem, s_cmem, s_l, saw = self.strings_encoder(seq[3], ifn(mask, 3), mems[3], cmems[3], self.pos_emb[3])
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        latents = torch.stack([d_z, b_z, g_z, s_z], dim=1)
        aux_loss = torch.stack((d_l, b_l, g_l, s_l)).mean()
        aws = torch.stack([daw, baw, gaw, saw], dim=0)
        return latents, mems, cmems, aux_loss, aws


class CompressiveDecoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 ff_mul=config["model"]["ff_mul"],
                 ff_dropout=config["model"]["ff_dropout"],
                 attn_layer_dropout=config["model"]["attn_layer_dropout"],
                 recon_attn_dropout=config["model"]["reconstruction_attn_dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 seq_len=config["model"]["seq_len"],
                 mem_len=config["model"]["mem_len"],
                 cmem_len=config["model"]["cmem_len"],
                 cmem_ratio=config["model"]["cmem_ratio"],
                 device=config["train"]["device"]
                 ):
        super(CompressiveDecoder, self).__init__()
        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert cmem_len >= (mem_len // cmem_ratio), f'len of cmem should be at least ' f'{int(mem_len // cmem_ratio)}' \
                                                    f' but it is ' f'{int(cmem_len)}'
        self.pos_emb = nn.Parameter(torch.zeros(4, heads, seq_len + mem_len + cmem_len, d_model // heads, device=device,
                                                requires_grad=True))
        c = copy.deepcopy
        # self_mem_attn = Residual(PreNorm(d_model, MyMemoryAttention(heads, d_model, seq_len,
        #                                                             mem_len, cmem_len, cmem_ratio,
        #                                                             attn_dropout=attn_layer_dropout,
        #                                                             reconstruction_attn_dropout=recon_attn_dropout)))
        self_mem_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)

        src_attn = MultiHeadedAttention(heads, d_model, dropout=0.1)
        ff = FeedForward(d_model, ff_mul, dropout=ff_dropout)

        decoder = Decoder(DecoderLayer(c(self_mem_attn), c(src_attn), c(ff)), layers, vocab_size, d_model)

        self.drums_decoder = c(decoder)
        self.bass_decoder = c(decoder)
        self.guitar_decoder = c(decoder)
        self.strings_decoder = c(decoder)
        self.generator = Generator(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, trg, trg_mask, src_mask, latent, mems, cmems):
        d_out, d_mem, d_cmem, d_l, d_self_w, d_src_w = self.drums_decoder(trg[0], ifn(trg_mask, 0), ifn(src_mask, 0),
                                                                           latent[0], mems[0], cmems[0], self.pos_emb[0])
        b_out, b_mem, b_cmem, b_l, b_self_w, b_src_w = self.bass_decoder(trg[1], ifn(trg_mask, 1), ifn(src_mask, 1),
                                                                          latent[1], mems[1], cmems[1], self.pos_emb[1])
        g_out, g_mem, g_cmem, g_l, g_self_w, g_src_w = self.guitar_decoder(trg[2], ifn(trg_mask, 2), ifn(src_mask, 2),
                                                                            latent[2], mems[2], cmems[2], self.pos_emb[2])
        s_out, s_mem, s_cmem, s_l, s_self_w, s_src_w = self.strings_decoder(trg[3], ifn(trg_mask, 3), ifn(src_mask, 3),
                                                                             latent[3], mems[3], cmems[3], self.pos_emb[3])
        output = torch.stack([d_out, b_out, g_out, s_out], dim=0)
        output = self.generator(output)
        self_weights = torch.stack([d_self_w, b_self_w, g_self_w, s_self_w], dim=0)
        src_weights = torch.stack([d_src_w, b_src_w, g_src_w, s_src_w])
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        aux_loss = torch.stack((d_l, b_l, g_l, s_l)).mean()
        return output, mems, cmems, aux_loss, self_weights, src_weights


class Encoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N
        self.pos = PositionalEncoding(d_model)

    def forward(self, seq, mask, mems, cmems, pos_emb):
        attn_losses = torch.tensor(0., requires_grad=True, device=seq.device, dtype=torch.float32)
        seq = self.embed(seq)
        seq = self.pos(seq)  # TODO REMOVE
        pos_emb = None  # TODO REMOVE
        new_mems = []
        new_cmems = []
        self_weights = []
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            seq, new_mem, new_cmem, attn_loss, attn = layer(seq, (mem, cmem), mask, pos_emb)  # pos_emb
            self_weights.append(attn)
            new_mems.append(new_mem)
            new_cmems.append(new_cmem)
            attn_losses = attn_losses + attn_loss
        self_weights = torch.stack(self_weights, dim=0)
        new_mems = torch.stack(new_mems)
        new_cmems = torch.stack(new_cmems)
        attn_loss = attn_losses / self.N  # normalize w.r.t number of layers
        return seq, new_mems, new_cmems, attn_loss, self_weights


class Decoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N
        self.pos = PositionalEncoding(d_model)

    def forward(self, trg, trg_mask, src_mask, latent, mems, cmems, pos_emb):
        attn_losses = torch.tensor(0., requires_grad=True, device=trg.device, dtype=torch.float32)
        trg = self.embed(trg)
        trg = self.pos(trg)  # TODO REMOVE
        pos_emb = None  # TODO REMOVE
        self_weights = []
        src_weights = []
        new_mems = []
        new_cmems = []
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            trg, new_mem, new_cmem, attn_loss, self_weight, src_weight = layer(trg, trg_mask, src_mask,
                                                                               latent, (mem, cmem), pos_emb)
            self_weights.append(self_weight)
            src_weights.append(src_weight)
            new_mems.append(new_mem)
            new_cmems.append(new_cmem)
            attn_losses = attn_losses + attn_loss
        src_weights = torch.stack(src_weights, dim=0)
        self_weights = torch.stack(self_weights, dim=0)
        new_mems = torch.stack(new_mems)
        new_cmems = torch.stack(new_cmems)
        attn_losses = attn_losses / self.N
        return trg, new_mems, new_cmems, attn_losses, self_weights, src_weights


class EncoderLayer(nn.Module):
    def __init__(self, mem_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(config["model"]["d_model"])
        self.norm2 = nn.LayerNorm(config["model"]["d_model"])

    def forward(self, x, memories, input_mask, pos_emb):
        # x, m, cm, attn_loss, self_weights = self.mem_attn(x, memories=memories, input_mask=input_mask, pos_emb=pos_emb)
        out, self_weights = self.mem_attn(x, key=x, value=x, mask=input_mask, pos_emb=pos_emb)
        x = self.norm1(x + out)
        out = self.feed_forward(x)
        x = self.norm2(x + out)
        # return x, m, cm, attn_loss, self_weights
        return x, torch.zeros_like(x).to(x.device), torch.zeros_like(x).to(x.device), 0, self_weights


class DecoderLayer(nn.Module):
    def __init__(self, self_mem_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.self_mem_attn = self_mem_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(config["model"]["d_model"])
        self.norm2 = nn.LayerNorm(config["model"]["d_model"])
        self.norm3 = nn.LayerNorm(config["model"]["d_model"])

    def forward(self, x, trg_mask, src_mask, latent, memories, pos_emb):
        # x, m, cm, attn_loss, self_weights = self.self_mem_attn(x, memories=memories, input_mask=trg_mask, pos_emb=pos_emb)
        out, self_weights = self.self_mem_attn(x, key=x, value=x, mask=trg_mask, pos_emb=pos_emb)
        x = self.norm1(x + out)
        out, src_weights = self.src_attn(x, key=latent, value=latent, mask=src_mask)
        x = self.norm2(x + out)
        out = self.feed_forward(x)
        x = self.norm3(x + out)
        # return x, m, cm, attn_loss, self_weights, src_weights
        return x, torch.zeros_like(x).to(x.device), torch.zeros_like(x).to(x.device), 0, self_weights, src_weights


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
        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_out = d_model // h
        self.h = h
        self.linears = (clones(nn.Linear(d_model, d_model, bias=False), 4))  # TODO bias or not?
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key=None, value=None, mask=None, pos_emb=None):
        if mask is not None:
            if mask.dim() == 2:  # transform linear mask to square mask
                mask = mask[:, :, None] * mask[:, None, :]
            mask = mask.unsqueeze(1)  # apply same mask to all heads
        n_batches = query.size(0)
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_out).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, weights = full_attn(query, key, value, mask=mask, dropout=self.dropout, pos_emb=pos_emb)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_out)
        return self.linears[-1](x), weights


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
        pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * (dim ** 0.5)
        pos_dots = shift(pos_dots)  # left upper triangular has positional embedding of illegal token
        pos_dots = pos_dots[..., :dots.shape[-1]]  # TODO select useful embedding, confirm or remove
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
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]


def to(t):
    return {'dtype': t.dtype, 'device': t.device}


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