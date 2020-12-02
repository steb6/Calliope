import torch.nn as nn
import torch
import math
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd
import copy
from muspy_config import config
from functools import partial
from collections import namedtuple


Memory = namedtuple('Memory', ['mem', 'compressed_mem'])

# TODO add positional encoding coherent for encoder e decoder
# TODO use easy library to upload dataset

# TODO add all the other AE functionalities
class TransformerAutoencoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self,
                 d_model=256,
                 n_tracks=4,
                 heads=4,
                 d_ff=512,
                 dropout=0.0,
                 layers=3,
                 vocab_size=128 + 128 + 1000 + 32 + 1 + 1 + 1 + 1,
                 seq_len=200,
                 mem_len=200,
                 cmem_len=32,
                 cmem_ratio=4):
        super(TransformerAutoencoder, self).__init__()

        # Deepcopy creates new instances of the object, nothing is shared between layers
        c = copy.deepcopy
        # x + norm(attention())
        attn = Residual(PreNorm(d_model, MultiHeadedAttention(heads, d_model, seq_len, mem_len, cmem_len, cmem_ratio)))
        dec_attn = Residual(PreNorm(d_model, DecoderMultiHeadedAttention(heads, d_model)))
        # dec_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # x = x + norm(ff())
        ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff, dropout=0.1)))
        position = PositionalEncoding(d_model, dropout)
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layers)
        decoder = Decoder(DecoderLayer(d_model, c(dec_attn), c(dec_attn), c(ff), dropout), layers)

        # need one encoder and one decoder for each of the four tracks
        self.drums_encoder = c(encoder)
        self.bass_encoder = c(encoder)
        self.guitar_encoder = c(encoder)
        self.strings_encoder = c(encoder)

        self.drums_decoder = c(decoder)
        self.bass_decoder = c(decoder)
        self.guitar_decoder = c(decoder)
        self.strings_decoder = c(decoder)

        # TODO un embedding per ogni traccia o basta un emebdding layer unico ?
        self.src_embed = nn.Sequential(Embeddings(vocab_size, d_model), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(vocab_size, d_model), c(position))
        # self.src_embed = Embeddings(src_vocab, d_model)
        # self.tgt_embed = Embeddings(src_vocab, d_model)

        # positional embeddings
        # seq_and_mem_len = seq_len + mem_len + cmem_len
        # self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, d_model // heads))

        self.generator = Generator(d_model, vocab_size)

        # TODO per ora lasciare così ma in futuro non aggregare latents delle varie tracce
        self.linear_encoder = nn.Linear(d_model * n_tracks, d_model)
        # self.linear_decoder = nn.Linear(d_model * n_tracks, d_model)
        # additional info
        self.n_tracks = n_tracks
        self.layers = layers
        self.d_model = d_model
        self.seq_len = seq_len

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, memories=None):
        "Take in and process masked src and target sequences."
        # Check if memories are empty, in the case create new memories
        b, t, d = src.shape
        if memories is None:
            memories = (None, None)
        mem, cmem = memories
        # every memory has size n_tracks * n_layer * b * 0 * d
        if mem is None:
            mem = torch.empty(self.n_tracks, self.layers, b, 0, self.d_model, **to(src))
        if cmem is None:
            cmem = torch.empty(self.n_tracks, self.layers, b, 0, self.d_model, **to(src))
        # positional embeddings
        total_len = mem.shape[3] + cmem.shape[3] + self.seq_len
        # pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]
        pos_emb = None # TODO CORREGGERE
        # Set up new memories and loss
        # aux_loss = torch.tensor(0., requires_grad=True, **to(src))
        latents, next_memories, aux_loss = self.encode(src, src_mask, (mem, cmem), pos_emb)
        # detach memories
        next_mem, next_cmem = next_memories
        next_mem, next_cmem = map(torch.detach, (next_mem, next_cmem))
        output = self.decode(latents, src_mask, tgt, tgt_mask)
        return output, (next_mem, next_cmem), aux_loss

    def encode(self, src, src_mask, memories, pos_emb):
        mem, cmem = memories
        # For every track, encode
        drums_z, drums_mem, drums_cmem, drums_l = self.drums_encoder(self.src_embed(src[:, :, 0]), src_mask[:, :, 0], (mem[0], cmem[0]), pos_emb)
        bass_z, bass_mem, bass_cmem, bass_l = self.bass_encoder(self.src_embed(src[:, :, 1]), src_mask[:, :, 1],(mem[1], cmem[1]), pos_emb)
        guitar_z, guitar_mem, guitar_cmem, guitar_l = self.guitar_encoder(self.src_embed(src[:, :, 2]), src_mask[:, :, 2], (mem[2], cmem[2]), pos_emb)
        strings_z, strings_mem, strings_cmem, strings_l = self.strings_encoder(self.src_embed(src[:, :, 3]), src_mask[:, :, 3], (mem[3], cmem[3]), pos_emb)
        # Stack memories
        drums_mem, drums_cmem = map(torch.stack, (drums_mem, drums_cmem))
        bass_mem, bass_cmem = map(torch.stack, (bass_mem, bass_cmem))
        guitar_mem, guitar_cmem = map(torch.stack, (guitar_mem, guitar_cmem))
        strings_mem, strings_cmem = map(torch.stack, (strings_mem, strings_cmem))
        new_mem = torch.stack((drums_mem, bass_mem, guitar_mem, strings_mem))
        new_cmem = torch.stack((drums_cmem, bass_cmem, guitar_cmem, strings_cmem))
        # Sum up losses
        # aux_loss = torch.tensor(0., requires_grad=True, **to(src))
        aux_loss = torch.stack((drums_l, bass_l, guitar_l, strings_l))
        aux_loss = torch.mean(aux_loss)
        # Concatenate latents of every track
        latents = torch.cat([drums_z, bass_z, guitar_z, strings_z], dim=2)
        latents = self.linear_encoder(latents)  # TODO non aggregare
        return latents, (new_mem, new_cmem), aux_loss

    def decode(self, memory, src_mask, tgt, tgt_mask):
        drums_out = self.drums_decoder(self.tgt_embed(tgt[:, :, 0]), memory, src_mask[:, :, 0], tgt_mask[:, :, :, 0])
        bass_out = self.bass_decoder(self.tgt_embed(tgt[:, :, 1]), memory, src_mask[:, :, 1], tgt_mask[:, :, :, 1])
        guitar_out = self.guitar_decoder(self.tgt_embed(tgt[:, :, 2]), memory, src_mask[:, :, 2], tgt_mask[:, :, :, 2])
        strings_out = self.strings_decoder(self.tgt_embed(tgt[:, :, 3]), memory, src_mask[:, :, 3],
                                           tgt_mask[:, :, :, 3])
        output = torch.stack([drums_out, bass_out, guitar_out, strings_out], dim=-1)
        return self.generator(output)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)
        self.N = N

    def forward(self, x, mask, memories, pos_emb):
        "Pass the input (and mask) through each layer in turn."
        mems, cmems = memories
        new_mem = []
        new_cmem = []
        aux_loss = torch.tensor(0., requires_grad=True, **to(x))
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            x, new_memory, layer_loss = layer(x, mask, (mem, cmem), pos_emb)
            new_mem.append(new_memory[0])
            new_cmem.append(new_memory[1])
            aux_loss = aux_loss + layer_loss
        return x, new_mem, new_cmem, aux_loss


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, memories, pos_emb):
        "Follow Figure 1 (left) for connections."
        x, new_memories, aux_loss = self.self_attn(x, memories=memories, input_mas=mask, pos_emb=pos_emb)
        x, = self.feed_forward(x)
        # for memory in new_memories:
            # memory.detach()
        # x = self.sublayer[0](old_x, lambda k: x)
        return x, new_memories, aux_loss # self.sublayer[1](x, self.feed_forward), new_memories, aux_loss


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # self_attn è Residual(Prenorm(MultiHeadAttention))
        # quindi Reisdual prende x e gli altri argomenti che riceve e li passa a PreNorm
        x, = self.self_attn(x, key=x, value=x, mask=tgt_mask)
        x, = self.src_attn(x, key=m, value=m, mask=src_mask)
        x, = self.feed_forward(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, dim, seq_len, mem_len, cmem_len, cmem_ratio, dropout=0.1, attn_dropout=0.1,
                 reconstruction_attn_dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.dim_head = dim // h
        self.h = h
        # 4 projection layers: 1 for query, 2 for keys, 3 for values, and final at the end
        # self.linears = (clones(nn.Linear(dim, dim), 4))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        # Is 1/root(dim_head)
        self.scale = self.dim_head ** (-0.5)

        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias=False)

        kv_dim = dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)

    def forward(self, x, memories=None, pos_emb=None, input_mask=None, calc_memory=True, **kwargs):

        mem, cmem = memories

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        b, t, d = x.shape

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # Here we have q with dimension (1, 1024, 512) and k and v with dimensions (1, x, 512)
        # Now we want to have the form (1, 8, 1024, 64), 512/8 = 64, in order to do self attention
        # The following function is independent from t (1024 or x) and number of dimensions
        merge_heads = lambda x: reshape_dim(x, -1, (-1, self.dim_head)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))
        # Now we have q = (1, 8, 1024, 64) and k = v = (1, 8, x, 64)
        k, v = map(lambda x: x.expand(-1, self.h, -1, -1), (k, v))
        # Compute attention as Q K^T / root(dim_head)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # Lowest possible value of float32
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            # Take last kv_len positional embedding
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            # Like before, Q pos_emb^T / root(dim_head)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            # What does shift do?
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            # Create two tensor (1, 1, 1024, 1) and (1, 1, 1, 1024) and multiply them together to obtain
            # a tensor with size (1, 1, 1024, 1024)
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            # Pad the last dimension of the mask to cover the memories, those zeros will become true the line after
            mask = F.pad(mask, (mem_len + cmem_len, 0), value=True)
            # Where not mask is true, insert the mask value defined before
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + cmem_len
        # Create matrix of ones of size (1024, 1024) and return true only elements over the 1+total_mem_len diagonal
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal=1 + total_mem_len).bool()
        # Put elements under 1+total_mem_len diagonal to -inf
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # from (1, 8, 1024, 64) to (1, 1024, 8, 64) to (1, 1024, 512)
        out = out.transpose(1, 2).reshape(b, t, -1)
        # from dim to dim, from 512 to 512, necessary to aggregate all attention layers
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad=True, **to(q))

        # if seq_len > t means that the sequence is over, so no more memory update is needed
        if self.seq_len > t or not calc_memory:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = queue_fifo(mem, x, length=self.mem_len, dim=1)

        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value=0.)

        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss
        # STOP GRADIENT DETACHING MEMORY
        old_mem = old_mem.detach()
        compressed_mem = self.compress_mem_fn(old_mem)
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        if not self.training:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate compressed memory auxiliary loss if training

        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, self.h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))
        # TODO DANGER:  UNDERSTAND WHY IN ORDER TO TRAIN THE COMPRESSOR I NEEDED TO REMOVE THIS LINE
        # TODO WHY THIS LINE WAS HERE IN A MODEL THAT WORKS? IS THERE ANY OTHER WAY? DO I NEED TO STOP THE GRADIENT?
        # q, old_mem_k, old_mem_v, cmem_k, cmem_v = map(torch.detach, (q, old_mem_k, old_mem_v, cmem_k, cmem_v))
        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        attn_fn = partial(full_attn, dropout_fn=self.reconstruction_attn_dropout)

        aux_loss = F.mse_loss(
            attn_fn(q, old_mem_k, old_mem_v),
            attn_fn(q, cmem_k, cmem_v)
        )

        return logits, Memory(new_mem, new_cmem), aux_loss


class DecoderMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(DecoderMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_out = d_model // h
        self.h = h
        # 4 projection layers: 1 for query, 2 for keys, 3 for values, and final at the end
        self.linears = (clones(nn.Linear(d_model, d_model), 4))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key=None, value=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            # src_mask happends to be on one dimension. so we unsqueeze it to be applied to each attention result
            # but tgt mask is going to be of 2 dimension so se have no need to unsqueeze, and apply it directly
            # TODO: check if we should compute a square src_mask outside the model
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-2)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # here we use first three projection layers
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_out).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_out)

        # here we use final projection layer
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.):
        super().__init__()
        activation = nn.GELU

        self.w1 = nn.Linear(dim, hidden)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x, **kwargs):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    # apply out = fn(x, **kwargs) and return out+x and other results from fn
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
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj_drums = nn.Linear(d_model, vocab)
        self.proj_bass = nn.Linear(d_model, vocab)
        self.proj_guitar = nn.Linear(d_model, vocab)
        self.proj_strings = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x is (3, 199, 512), it is projected to (3, 199, 1292) and then softmax
        out_drums = F.log_softmax(self.proj_drums(x[:, :, :, 0]), dim=-1)
        out_bass = F.log_softmax(self.proj_bass(x[:, :, :, 1]), dim=-1)
        out_guitar = F.log_softmax(self.proj_guitar(x[:, :, :, 2]), dim=-1)
        out_strings = F.log_softmax(self.proj_strings(x[:, :, :, 3]), dim=-1)
        # at the end, for each time step we have one-hot of token
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
        try:
            aux = self.lut(x)
        except:
            print(torch.max(x))
        return aux * math.sqrt(self.d_model)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # create a tensor with size (597, 1292) and fill it with this value
        # size is token dimension, smoothing is 0.1
        true_dist.fill_(self.smoothing / (self.size - 2))
        # unsqueeze add a dimension in the specified position
        # put self.confidence value (0.9) in true dist in the positions indicated by target
        # so true_dist has the same size of x which have 0.9 in the right notes and a small value in all the others
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # set all padding position to 0 (because we have a lot of them)
        true_dist[:, self.padding_idx] = 0
        # return position of padding of target
        mask = torch.nonzero(target.data == self.padding_idx)
        # substitute position of pad in true_dist with 0
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.to(config.device))  # TODO mettere a posto in caso di multi gpu


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    " Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim + 1] = split_dims
    return t.reshape(shape)


def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    # Add a (1, 8, 1024, 1024) matrix of zero along last axis, so we have (1, 8, 1024, 2048)
    x = torch.cat([x, zero_pad], -1)
    # l is 2047
    l = i + j - 1
    # From (1, 8, 1024, 2048) to (1, 8, 2097152)
    x = x.view(*_, -1)
    # Create zero matrix with dimension (1, 8, 1023)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
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
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def full_attn(q, k, v, dropout_fn=None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v)


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


# TODO USELESS?
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def calc_gradient_penalty(model, real_data, gen_data):
    device = config.training["device"]
    batch_size = config.training["batch_size"]
    alpha = torch.rand(batch_size, 1)

    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = autograd.Variable(interpolates.to(device), requires_grad=True)
    score_interpolates = model(interpolates)

    gradients = autograd.grad(
        outputs=score_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(score_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # TODO ERA FALSE
        return self.dropout(x)