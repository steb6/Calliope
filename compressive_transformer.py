import torch.nn as nn
import torch
import math
from torch.nn import functional as F
from torch.autograd import Variable
import copy
from functools import partial
from collections import namedtuple
from config import config
from math import ceil

# from compress_latents import CompressLatents, DecompressLatents

Memory = namedtuple('Memory', ['mem', 'compressed_mem'])


class CompressiveEncoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 d_ff=config["model"]["d_ff"],
                 dropout=config["model"]["dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 seq_len=config["model"]["seq_len"],
                 mem_len=config["model"]["mem_len"],
                 cmem_len=config["model"]["cmem_len"],
                 cmem_ratio=config["model"]["cmem_ratio"],
                 pad_token=config["tokens"]["pad"],
                 z_i_dim=config["model"]["z_i_dim"],
                 ):
        super(CompressiveEncoder, self).__init__()
        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert cmem_len >= (mem_len // cmem_ratio), f'len of cmem should be at least ' f'{int(mem_len // cmem_ratio)}' \
                                                    f' but it is ' f'{int(cmem_len)}'
        c = copy.deepcopy
        e_mem_attn = Residual(PreNorm(d_model, MemoryMultiHeadedAttention(heads, d_model, seq_len,
                                                                          mem_len, cmem_len, cmem_ratio,
                                                                          mask_subsequent=False)))
        # e_mem_attn = Residual(PreNorm(d_model, MultiHeadedAttention(heads, d_model)))
        ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff, dropout=0.1)))
        encoder = Encoder(EncoderLayer(d_model, c(e_mem_attn), c(ff), dropout),
                          layers, vocab_size, d_model, pad_token, seq_len, z_i_dim)
        self.drums_encoder = c(encoder)
        self.bass_encoder = c(encoder)
        self.guitar_encoder = c(encoder)
        self.strings_encoder = c(encoder)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, mask, mems, cmems):
        d_z, d_mem, d_cmem, d_l, d_ae, daw = self.drums_encoder(seq[:, 0, :], mask[:, 0, :], mems[0, ...],
                                                                cmems[0, ...])
        b_z, b_mem, b_cmem, b_l, b_ae, baw = self.bass_encoder(seq[:, 1, :], mask[:, 1, :], mems[1, ...], cmems[1, ...])
        g_z, g_mem, g_cmem, g_l, g_ae, gaw = self.guitar_encoder(seq[:, 2, :], mask[:, 2, :], mems[2, ...],
                                                                 cmems[2, ...])
        s_z, s_mem, s_cmem, s_l, s_ae, saw = self.strings_encoder(seq[:, 3, :], mask[:, 3, :], mems[3, ...],
                                                                  cmems[3, ...])
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        mems, cmems = map(torch.detach, (mems, cmems))
        latents = torch.stack([d_z, b_z, g_z, s_z], dim=1)
        aux_loss = torch.stack((d_l, b_l, g_l, s_l)).mean()
        ae_loss = torch.stack((d_ae, b_ae, g_ae, s_ae)).mean()
        aws = torch.mean(torch.stack([daw, baw, gaw, saw], dim=0), dim=0)
        return latents, mems, cmems, aux_loss, ae_loss, aws


class CompressiveDecoder(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 heads=config["model"]["heads"],
                 d_ff=config["model"]["d_ff"],
                 dropout=config["model"]["dropout"],
                 layers=config["model"]["layers"],
                 vocab_size=config["tokens"]["vocab_size"],
                 seq_len=config["model"]["seq_len"],
                 mem_len=config["model"]["mem_len"],
                 cmem_len=config["model"]["cmem_len"],
                 cmem_ratio=config["model"]["cmem_ratio"],
                 pad_token=config["tokens"]["pad"],
                 z_i_dim=config["model"]["z_i_dim"],
                 ):
        super(CompressiveDecoder, self).__init__()
        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert cmem_len >= (mem_len // cmem_ratio), f'len of cmem should be at least ' f'{int(mem_len // cmem_ratio)}' \
                                                    f' but it is ' f'{int(cmem_len)}'
        c = copy.deepcopy
        # self_mem_attn = Residual(PreNorm(d_model, MemoryMultiHeadedAttention(heads, d_model, seq_len,
        #                                                                      mem_len, cmem_len, cmem_ratio,
        #                                                                      mask_subsequent=True)))
        self_mem_attn = Residual(PreNorm(d_model, MultiHeadedAttention(heads, d_model)))
        src_attn = Residual(PreNorm(d_model, MultiHeadedAttention(heads, d_model)))

        ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff, dropout=0.1)))
        decoder = Decoder(DecoderLayer(d_model, c(self_mem_attn), c(src_attn), c(ff), dropout, heads),
                          layers, vocab_size, d_model, pad_token, seq_len, z_i_dim)
        self.drums_decoder = c(decoder)
        self.bass_decoder = c(decoder)
        self.guitar_decoder = c(decoder)
        self.strings_decoder = c(decoder)
        self.generator = Generator(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, trg, latent, src_mask, trg_mask, d_mems, d_cmems):
        # d_out, d_mem, d_cmem, d_l, d_ae
        d_out, dlw, dsw, d_mem, d_cmem, d_l, d_ae = self.drums_decoder(trg[:, 0, :], latent[0, ...], src_mask[:, 0, :],
                                                                       trg_mask[:, 0, ...], d_mems[0, ...],
                                                                       d_cmems[0, ...])
        b_out, blw, bsw, b_mem, b_cmem, b_l, b_ae = self.bass_decoder(trg[:, 1, :], latent[1, ...], src_mask[:, 1, :],
                                                                      trg_mask[:, 1, ...], d_mems[1, ...],
                                                                      d_cmems[1, ...])
        g_out, glw, gsw, g_mem, g_cmem, g_l, g_ae = self.guitar_decoder(trg[:, 2, :], latent[2, ...], src_mask[:, 2, :],
                                                                        trg_mask[:, 2, ...], d_mems[2, ...],
                                                                        d_cmems[2, ...])
        s_out, slw, ssw, s_mem, s_cmem, s_l, s_ae = self.strings_decoder(trg[:, 3, :], latent[3, ...],
                                                                         src_mask[:, 3, :],
                                                                         trg_mask[:, 3, ...], d_mems[3, ...],
                                                                         d_cmems[3, ...])
        mems = torch.stack([d_mem, b_mem, g_mem, s_mem])
        cmems = torch.stack([d_cmem, b_cmem, g_cmem, s_cmem])
        mems, cmems = map(torch.detach, (mems, cmems))
        output = torch.stack([d_out, b_out, g_out, s_out], dim=-1)
        output = self.generator(output)
        aux_loss = torch.stack((d_l, b_l, g_l, s_l))
        aux_loss = torch.mean(aux_loss)
        ae_loss = torch.stack((d_ae, b_ae, g_ae, s_ae))
        ae_loss = torch.mean(ae_loss)
        latents_weight = torch.mean(torch.stack([dlw, blw, glw, slw], dim=0), dim=0)
        self_weights = torch.mean(torch.stack([dsw, bsw, gsw, ssw], dim=0), dim=0)
        return output, latents_weight, self_weights, mems, cmems, aux_loss, ae_loss


class Encoder(nn.Module):
    def __init__(self, layer, N, vocab_size, d_model, pad_token, seq_len, z_i_dim):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, 0.1)
        self.N = N
        self.pad_token = pad_token
        self.d_model = d_model
        self.compress_bar = nn.Linear(d_model * (seq_len + 1), z_i_dim)  # for eos token

    def forward(self, seq, mask, mems, cmems):
        attn_losses = torch.tensor(0., requires_grad=True, device=seq.device, dtype=torch.float32)
        ae_losses = torch.tensor(0., requires_grad=True, device=seq.device, dtype=torch.float32)
        seq = self.embed(seq)
        seq = self.position(seq)
        new_mem = []
        new_cmem = []
        attns = []
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            seq, new_memories, attn_loss, ae_loss, attn = layer(seq, (mem, cmem), mask)
            attns.append(attn)
            new_mem.append(new_memories[0])
            new_cmem.append(new_memories[1])
            attn_losses = attn_losses + attn_loss
            ae_losses = ae_losses + ae_loss
        # PAD ATTNS
        attns = torch.mean(torch.stack(attns, dim=0), dim=(0, 1, 2))
        # END PAD ATTNS
        mems = torch.stack(new_mem)
        cmems = torch.stack(new_cmem)
        attn_loss = attn_losses / self.N  # normalize w.r.t number of layers
        ae_loss = ae_losses / self.N  # normalize w.r.t number of layers
        return seq, mems, cmems, attn_loss, ae_loss, attns


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N, vocab_size, d_model, pad_token, seq_len, z_i_dim):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.pad_token = pad_token
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, 0.1)
        self.N = N
        self.d_model = d_model
        self.decompress_bar = nn.Linear(z_i_dim, d_model * (seq_len - 1))
        self.seq_len = seq_len

    def forward(self, trg, latent, src_mask, trg_mask, mems, cmems):
        attn_losses = torch.tensor(0., requires_grad=True, device=trg.device, dtype=torch.float32)
        ae_losses = torch.tensor(0., requires_grad=True, device=trg.device, dtype=torch.float32)
        trg = self.embed(trg)
        trg = self.position(trg)
        new_mem = []
        new_cmem = []
        latents_weight = []
        self_weights = []
        for layer, mem, cmem in zip(self.layers, mems, cmems):
            # for layer in self.layers:
            trg, latent_weight, self_weight, new_memories, attn_loss, ae_loss = layer(trg, latent, src_mask, trg_mask,
                                                                                      (mem, cmem))
            # trg, latent_weight, self_weight = layer(trg, latent, src_mask, trg_mask)
            latents_weight.append(latent_weight)
            self_weights.append(self_weight)
            new_mem.append(new_memories[0])
            new_cmem.append(new_memories[1])
            attn_losses = attn_losses + attn_loss
            ae_losses = ae_losses + ae_loss
        latents_weight = torch.mean(torch.stack(latents_weight, dim=0), dim=(0, 1, 2))  # mn of layer batch instruments
        self_weights = torch.mean(torch.stack(self_weights, dim=0), dim=(0, 1, 2))  # mn of layer batch instruments
        mems = torch.stack(new_mem)
        cmems = torch.stack(new_cmem)
        attn_losses = attn_losses / self.N  # normalize w.r.t number of layers
        ae_losses = ae_losses / self.N  # normalize w.r.t number of layers
        return trg, latents_weight, self_weights, mems, cmems, attn_losses, ae_losses


class EncoderLayer(nn.Module):
    def __init__(self, size, mem_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, memories, input_mask):
        x, new_memories, attn_loss, ae_loss, attn = self.mem_attn(x, memories=memories, input_mask=input_mask)
        # x, = self.mem_attn(x, key=x, value=x, mask=input_mask)
        x, = self.feed_forward(x)
        return x, new_memories, attn_loss, ae_loss, attn
        # return x, (torch.zeros(1, requires_grad=True, **to(x)), torch.zeros(1, requires_grad=True, **to(x))), \
        #        torch.zeros(1, requires_grad=True, **to(x)), torch.zeros(1, requires_grad=True, **to(x))


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, mem_attn, src_attn, feed_forward, dropout, heads):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.mem_attn = mem_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.heads = heads

    def forward(self, trg, latent, src_mask, trg_mask, memories):
        # x, new_memories, attn_loss, ae_loss, self_weights = self.mem_attn(trg, memories=memories, input_mask=trg_mask)
        attn_loss = torch.zeros(1, requires_grad=True, **to(trg))
        ae_loss = torch.zeros(1, requires_grad=True, **to(trg))
        new_memories = memories
        x, self_weights = self.mem_attn(trg, key=trg, value=trg, mask=trg_mask)

        src_mask = None  # we only have one latent for the full sequence
        x, latent_weights = self.src_attn(x, key=latent, value=latent, mask=src_mask)
        x, = self.feed_forward(x)
        return x, latent_weights, self_weights, new_memories, attn_loss, ae_loss
        # return x, (torch.zeros(1, requires_grad=True, **to(x)), torch.zeros(1, requires_grad=True, **to(x))), \
        #        torch.zeros(1, requires_grad=True, **to(x)), torch.zeros(1, requires_grad=True, **to(x))


class MemoryMultiHeadedAttention(nn.Module):
    def __init__(self, h, dim, seq_len, mem_len, cmem_len, cmem_ratio, dropout=0.1, attn_dropout=0.1,
                 reconstruction_attn_dropout=0.1, ae_dropout=0.1, mask_subsequent=None):
        super(MemoryMultiHeadedAttention, self).__init__()
        assert dim % h == 0
        self.dim_head = dim // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)  # 1/root(dim_head)
        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)
        self.to_q = nn.Linear(dim, dim, bias=False)
        kv_dim = dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ae_dropout = nn.Dropout(ae_dropout)
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)
        self.reconstruct_mem = nn.Linear(ceil(mem_len / cmem_ratio), mem_len)
        self.attn_imgs = 0
        self.mask_subsequent = mask_subsequent

    def forward(self, x, memories=None, input_mask=None, calc_memory=True):
        mem, cmem = memories
        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]
        b, t, d = x.shape
        # Compute dots
        q = self.to_q(x)
        kv_input = torch.cat((cmem, mem, x), dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        def merge_heads(matrix):
            return reshape_dim(matrix, -1, (-1, self.dim_head)).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # Prepare masks
        mask_value = max_neg_value(dots)
        if input_mask is not None:  # set -inf under and right of the mask
            if input_mask.dim() == 2:  # encoder mask, cover just pad
                mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            elif input_mask.dim() == 3:  # decoder mask, cover pad and subsequent
                mask = input_mask.unsqueeze(1)
            else:
                raise Exception("Wrong mask provided")
            mask = F.pad(mask, (mem_len + cmem_len, 0), value=True)
            dots.masked_fill_(~mask, mask_value)  # dots has values in the upper square, the rest is -inf
        # Compute weights
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        # Compute logits
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)
        # Memories and auxiliary losses
        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad=True, **to(q))
        ae_loss = torch.zeros(1, requires_grad=True, **to(q))
        if t < self.seq_len or not calc_memory:  # TODO set calc_memory to false when autoregressive
            return logits, Memory(new_mem, new_cmem), aux_loss, ae_loss, attn
        # Calculate memory and compressed memory
        old_mem, new_mem = queue_fifo(mem, x, length=self.mem_len, dim=1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio
        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value=0.)  # Pad old memory to match cmem_ratio
        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss, ae_loss, attn  # No memory to compress
        # STOP GRADIENT DETACHING MEMORY
        # old_mem = old_mem.detach()
        compressed_mem = self.compress_mem_fn(old_mem)
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        # if not self.training:  # TODO without this, it does not use memory in evaluating new song
        # return logits, Memory(new_mem, new_cmem), aux_loss, ae_loss

        # Compute attention loss
        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, self.h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))
        # TODO DANGER:  UNDERSTAND WHY IN ORDER TO TRAIN THE COMPRESSOR I NEEDED TO REMOVE THIS LINE
        # TODO WHY THIS LINE WAS HERE IN A MODEL THAT WORKS? IS THERE ANY OTHER WAY? DO I NEED TO STOP THE GRADIENT?
        q, old_mem_k, old_mem_v, cmem_k, cmem_v = map(torch.detach, (q, old_mem_k, old_mem_v, cmem_k, cmem_v))
        # q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))
        attn_fn = partial(full_attn, dropout=self.reconstruction_attn_dropout)
        aux_loss = F.mse_loss(
            attn_fn(q, old_mem_k, old_mem_v)[0],
            attn_fn(q, cmem_k, cmem_v)[0]
        )

        # Compute auto-encoding loss TODO how to do it?
        ae_loss = torch.zeros(1, requires_grad=True, **to(q))
        # reconstructed_mem = self.reconstruct_mem(compressed_mem.transpose(1, 2)).transpose(1, 2)
        # reconstructed_mem = self.ae_dropout(reconstructed_mem)
        # to_cut = min(reconstructed_mem.shape[1], old_mem.shape[1])
        # old_mem = old_mem[:, :to_cut, :]
        # reconstructed_mem = reconstructed_mem[:, :to_cut, :]
        # ae_loss = F.mse_loss(
        #     old_mem,
        #     reconstructed_mem
        # )

        return logits, Memory(new_mem, new_cmem), aux_loss, ae_loss, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_out = d_model // h
        self.h = h
        self.linears = (clones(nn.Linear(d_model, d_model), 4))  # Wq Wk Wv Wout
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key=None, value=None, mask=None):
        if mask is not None:  # apply same mask to all heads
            mask = mask.unsqueeze(1)
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-2)  # src_mask has 1 dimension, trg_mask has 2 dimension
        n_batches = query.size(0)
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_out).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, weights = full_attn(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_out)
        return self.linears[-1](x), weights


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
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
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def full_attn(q, k, v, mask=None, dropout=None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)  # Q K^T
    if mask is not None:  # TODO fix this in decoder
        dots = dots.masked_fill(mask == 0, -1e9)  # 2, 4, 199, 199 and 2, 1, 1, 199 same mask for all heads
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
    left = (*pre_slices, slice(None, index))
    right = (*pre_slices, slice(index, None))
    return t[left], t[right]


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)
