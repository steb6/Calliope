import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import copy
from collections import namedtuple
from config import config


Memory = namedtuple('Memory', ['mem', 'compressed_mem'])


class CompressiveEncoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 ff_mul=config["model"]["ff_mul"],
                 attn_layer_dropout=config["model"]["attn_layer_dropout"],
                 reconstruction_attn_dropout=config["model"]["reconstruction_attn_dropout"],
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

        self.pos_emb = nn.Parameter(torch.zeros(4, heads, seq_len + mem_len + cmem_len, d_model // heads, device=device,
                                                requires_grad=True))
        c = copy.deepcopy

        self_mem_attn = Residual(PreNorm(d_model, MemorySelfAttention(heads, d_model, seq_len,
                                                                    mem_len, cmem_len, cmem_ratio,
                                                                    attn_dropout=attn_layer_dropout,
                                                                    reconstruction_attn_dropout=reconstruction_attn_dropout)))

        ff = Residual(PreNorm(d_model, FeedForward(d_model, ff_mul, dropout=ff_dropout)))

        encoder = Encoder(EncoderLayer(c(self_mem_attn), c(ff)), layers, vocab_size, d_model)
        self.drums_encoder = c(encoder)
        self.bass_encoder = c(encoder)
        self.guitar_encoder = c(encoder)
        self.strings_encoder = c(encoder)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, mask, mems, cmems):
        d_z, d_mem, d_cmem, d_l, daw = self.drums_encoder(seq[0, ...], mask[0, ...], mems[0, ...],
                                                          cmems[0, ...], self.pos_emb[0, ...])
        b_z, b_mem, b_cmem, b_l, baw = self.bass_encoder(seq[1, ...], mask[1, ...], mems[1, ...],
                                                         cmems[1, ...], self.pos_emb[1, ...])
        g_z, g_mem, g_cmem, g_l, gaw = self.guitar_encoder(seq[2, ...], mask[2, ...], mems[2, ...],
                                                           cmems[2, ...], self.pos_emb[2, ...])
        s_z, s_mem, s_cmem, s_l, saw = self.strings_encoder(seq[3, ...], mask[3, ...], mems[3, ...],
                                                            cmems[3, ...], self.pos_emb[3, ...])
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        latents = torch.stack([d_z, b_z, g_z, s_z], dim=1)
        aux_loss = torch.stack((d_l, b_l, g_l, s_l)).mean()
        aws = torch.mean(torch.stack([daw, baw, gaw, saw], dim=0), dim=0)
        return latents, mems, cmems, aux_loss, aws


class CompressiveDecoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 ff_mul=config["model"]["ff_mul"],
                 ff_dropout=config["model"]["ff_dropout"],
                 reconstruction_attn_dropout=config["model"]["reconstruction_attn_dropout"],
                 attn_layer_dropout=config["model"]["attn_layer_dropout"],
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
        self_mem_attn = Residual(PreNorm(d_model, MemorySelfAttention(heads, d_model, seq_len,
                                                                    mem_len, cmem_len, cmem_ratio,
                                                                    attn_dropout=attn_layer_dropout,
                                                                    reconstruction_attn_dropout=reconstruction_attn_dropout)))

        ff = Residual(PreNorm(d_model, FeedForward(d_model, ff_mul, dropout=ff_dropout)))
        decoder = Decoder(DecoderLayer(c(self_mem_attn), c(ff)), layers, vocab_size, d_model)

        self.drums_decoder = c(decoder)
        self.bass_decoder = c(decoder)
        self.guitar_decoder = c(decoder)
        self.strings_decoder = c(decoder)
        self.generator = Generator(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, trg, trg_mask, d_mems, d_cmems, count):  # TODO pass compress memories
        d_out, dsw, d_mem, d_cmem, d_l = self.drums_decoder(trg[0, ...], trg_mask[0, ...],
                                                            d_mems[0, ...], d_cmems[0, ...],
                                                            self.pos_emb[0, ...], count)
        b_out, bsw, b_mem, b_cmem, b_l = self.bass_decoder(trg[1, ...], trg_mask[1, ...],
                                                           d_mems[1, ...], d_cmems[1, ...],
                                                           self.pos_emb[1, ...], count)
        g_out, gsw, g_mem, g_cmem, g_l = self.guitar_decoder(trg[2, ...], trg_mask[2, ...],
                                                             d_mems[2, ...], d_cmems[2, ...],
                                                             self.pos_emb[2, ...], count)
        s_out, ssw, s_mem, s_cmem, s_l = self.strings_decoder(trg[3, ...], trg_mask[3, ...],
                                                              d_mems[3, ...], d_cmems[3, ...],
                                                              self.pos_emb[3, ...], count)
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        output = torch.stack([d_out, b_out, g_out, s_out], dim=-1)
        output = self.generator(output)
        aux_loss = torch.stack((d_l, b_l, g_l, s_l))
        aux_loss = torch.mean(aux_loss)
        self_weights = torch.mean(torch.stack([dsw, bsw, gsw, ssw], dim=0), dim=0)
        return output, self_weights, mems, cmems, aux_loss


class Encoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N

    def forward(self, seq, mask, mems, cmems, pos_emb):
        attn_losses = torch.tensor(0., requires_grad=True, device=seq.device, dtype=torch.float32)
        seq = self.embed(seq)
        new_mem = []
        new_cmem = []
        self_weights = []
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            seq, new_memories, attn_loss, attn = layer(seq, (mem, cmem), mask, pos_emb)
            self_weights.append(attn)
            new_mem.append(new_memories[0])
            new_cmem.append(new_memories[1])
            attn_losses = attn_losses + attn_loss
        self_weights = torch.mean(torch.stack(self_weights, dim=0), dim=(0, 1, 2))
        mems = torch.stack(new_mem)
        cmems = torch.stack(new_cmem)
        attn_loss = attn_losses / self.N  # normalize w.r.t number of layers
        return seq, mems, cmems, attn_loss, self_weights


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N, vocab_size, d_model):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.N = N

    def forward(self, trg, trg_mask, mems, cmems, pos_emb, count):
        attn_losses = torch.tensor(0., requires_grad=True, device=trg.device, dtype=torch.float32)
        trg = self.embed(trg)
        new_mem = []
        new_cmem = []
        self_weights = []
        slot_size = config["model"]["max_bar_length"]//config["model"]["cmem_ratio"]
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            trg, self_weight, new_memories, attn_loss = layer(trg, trg_mask, (mem, cmem), pos_emb, count=count)
            self_weights.append(self_weight)
            new_mem.append(new_memories[0])
            new_cmem.append(new_memories[1])
            attn_losses = attn_losses + attn_loss
        self_weights = torch.mean(torch.stack(self_weights, dim=0), dim=(0, 1, 2))  # mn of layer batch instruments
        mems = torch.stack(new_mem)
        cmems = torch.stack(new_cmem)
        attn_losses = attn_losses / self.N  # normalize w.r.t number of layers
        return trg, self_weights, mems, cmems, attn_losses


class EncoderLayer(nn.Module):
    def __init__(self, mem_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward

    def forward(self, x, memories, input_mask, pos_emb):
        x, new_memories, attn_loss, attn = self.mem_attn(x, memories=memories, input_mask=input_mask, pos_emb=pos_emb)
        x, = self.feed_forward(x)
        return x, new_memories, attn_loss, attn


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, mem_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward

    def forward(self, x, trg_mask, memories, pos_emb):
        x, new_memories, attn_loss, self_weights = self.mem_attn(x, memories=memories, input_mask=trg_mask, pos_emb=pos_emb)
        x, = self.feed_forward(x)
        return x, self_weights, memories, attn_loss  # TODO new memories or memories?


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

    def forward(self, h, memories=None, mask=None, pos_emb=None):
        # Prepare mask
        if mask.dim() == 2:  # encoder mask, cover just pad
            mask = mask[:, :, None] * mask[:, None, :]
        mask = F.pad(mask, (self.cmem_len + self.mem_len, 0), value=True)
        # Algorithm from paper
        m, cm = memories
        mem = torch.cat((cm, m, h), dim=1)  # TODO x too?
        a, weights = self.multi_head_attention(h, key=mem, value=mem, mask=mask, pos_emb=pos_emb)
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
            hQ = torch.matmul(hh, Q)
            mK = torch.matmul(mm, K)
            mV = torch.matmul(mm, V)
            z = torch.einsum("bxd,byd->bxy", hQ, mK)
            z = z.softmax(dim=-1)
            z = self.reconstruction_attn_dropout(z)
            return torch.matmul(z, mV)

        new_cm = self.compress_mem_fn(old_mem)
        l_attn = F.mse_loss(attn(h_copy, old_mem), attn(h_copy, new_cm))

        return h, Memory(m, cm), l_attn, weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_out = d_model // h
        self.h = h
        self.linears = (clones(nn.Linear(d_model, d_model, bias=False), 4))  # TODO bias or not?
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key=None, value=None, mask=None, pos_emb=None):
        if mask is not None:  # apply same mask to all heads
            mask = mask.unsqueeze(1)
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


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj_drums = nn.Linear(d_model, vocab)
        self.proj_bass = nn.Linear(d_model, vocab)
        self.proj_guitar = nn.Linear(d_model, vocab)
        self.proj_strings = nn.Linear(d_model, vocab)

    def forward(self, x):
        out_drums = F.log_softmax(self.proj_drums(x[:, :, :, 0]), dim=-1)
        out_bass = F.log_softmax(self.proj_bass(x[:, :, :, 1]), dim=-1)
        out_guitar = F.log_softmax(self.proj_guitar(x[:, :, :, 2]), dim=-1)
        out_strings = F.log_softmax(self.proj_strings(x[:, :, :, 3]), dim=-1)
        out = torch.stack([out_drums, out_bass, out_guitar, out_strings], dim=-1)
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
        pos_emb = pos_emb[:, -(k.shape[-2] + v.shape[-2]):].type(q.dtype)
        pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * (q.shape[-1] ** 0.5)
        pos_dots = shift(pos_dots)  # TODO what does this do?
        dots = dots + pos_dots

    if mask is not None:
        dots = dots.masked_fill(mask == 0, -1e9)  # same mask for all heads
    attn = dots.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v), attn  # (Q K^T) V


def clones(module, N):
    """ Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def reshape_dim(t, dim, split_dims):
    """
    Reshape dimension dim of tensor t with split dims
    Ex: t = (2, 200, 16), dim = -1, split_dims = (-1, 4) ---> t = (2, 200, -1, 4)
    """
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim + 1] = split_dims
    return t.reshape(shape)


def shift(x):
    """
    Get an x matrix and return a matrix y with the same shape were
    y[..., -1, :] = x[..., 0, :]
    y[..., -2, :] = x[..., 1, 1:] + [0]
    y[..., -3, :] = x[..., 2, 2:] + [0, 0]
    ...
    y[..., 0, :] = x[..., -1, n:] + [0]*n
    """
    *_, i, j = x.shape  # 3 x 4 x 150 x 450
    zero_pad = torch.zeros((*_, i, i), **to(x))  # 3 x 4 x 150 x 150
    # add to all heads attention weights 150 pad token
    x = torch.cat([x, zero_pad], -1)  # 3 x 4 x 150 x 600
    # sum the dimensions along attention axis
    l = i + j - 1  # 599
    # Flat last 2 dimensions of x
    x = x.view(*_, -1)  # 3 x 4 x 90000
    # Create zero matrix with dimension (1, 8, 1023)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))  # 3 x 4 x 449
    # Concatenate x (1, 8, 2097152), which is formed by 1024 elem and 1024 zeros, and a zero matrix (1, 8, 1023)
    # and change dimension as (1, 8, , 12047)
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    # return last 1024 token and first 1023 dimension
    return shifted[..., :i, i - 1:]


def to(t):
    return {'dtype': t.dtype, 'device': t.device}


def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device=device)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    left = (*pre_slices, slice(None, index))
    right = (*pre_slices, slice(index, None))
    return t[left], t[right]


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


class MemorySelfAttention(nn.Module):
    def __init__(self, heads, dim, seq_len, mem_len, cmem_len, cmem_ratio = 4, attn_dropout = 0., dropout = 0., reconstruction_attn_dropout = 0., one_kv_head = False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_dim = self.dim_head if one_kv_head else dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)

    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        mem, cmem = memories

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            if input_mask.dim() == 2:
                mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            else:
                mask = input_mask.unsqueeze(1)
            mask = F.pad(mask, (mem_len + cmem_len, 0), value=True)
            dots.masked_fill_(~mask, mask_value)

        # total_mem_len = mem_len + cmem_len
        # mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal = 1 + total_mem_len).bool()
        # dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        weights = attn.detach().clone()
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad = True, **to(q))

        # if self.seq_len > t or not calc_memory:
        #     return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = queue_fifo(mem, x, length = self.mem_len, dim = 1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value = 0.)

        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss, weights

        compressed_mem = self.compress_mem_fn(old_mem)
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        # if not self.training:
        #     return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate compressed memory auxiliary loss if training
        old_mem = old_mem.detach()  # TODO detached
        compressed_mem = self.compress_mem_fn(old_mem)

        freezed = self.to_kv.requires_grad_(False)
        cmem_k, cmem_v = freezed(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        # q, old_mem_k, old_mem_v, cmem_k, cmem_v = map(torch.detach, (q, old_mem_k, old_mem_v, cmem_k, cmem_v))
        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        aux_loss = F.mse_loss(
            full_attn(q, old_mem_k, old_mem_v, dropout = self.reconstruction_attn_dropout)[0],
            full_attn(q, cmem_k, cmem_v, dropout = self.reconstruction_attn_dropout)[0]
        )

        return logits, Memory(new_mem, new_cmem), aux_loss, weights
