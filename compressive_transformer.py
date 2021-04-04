import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from iterate_dataset import SongIterator
from config import config, max_bar_length
from copy import deepcopy as c
seaborn.set_context(context="talk")


class MusicEncoder(nn.Module):
    def __init__(self, layers, N, emb_pos):
        super(MusicEncoder, self).__init__()
        self.drums_encoder = Encoder(c(layers), N, c(emb_pos))
        self.guitar_encoder = Encoder(c(layers), N, c(emb_pos))
        self.bass_encoder = Encoder(c(layers), N, c(emb_pos))
        self.strings_encoder = Encoder(c(layers), N, c(emb_pos))

    def forward(self, x, mask):
        z_drums = self.drums_encoder(x[0], mask[0])
        z_guitar = self.guitar_encoder(x[1], mask[1])
        z_bass = self.bass_encoder(x[2], mask[2])
        z_strings = self.strings_encoder(x[3], mask[3])
        return torch.stack([z_drums, z_guitar, z_bass, z_strings])


class MusicDecoder(nn.Module):
    def __init__(self, layers, N, emb_pos):
        super(MusicDecoder, self).__init__()
        self.drums_decoder = Decoder(c(layers), N, c(emb_pos))
        self.guitar_decoder = Decoder(c(layers), N, c(emb_pos))
        self.bass_decoder = Decoder(c(layers), N, c(emb_pos))
        self.strings_decoder = Decoder(c(layers), N, c(emb_pos))

    def forward(self, x, memory, src_mask, trg_mask):
        z_drums = self.drums_decoder(x[0], memory[0], src_mask[0], trg_mask[0])
        z_guitar = self.guitar_decoder(x[1], memory[1], src_mask[1], trg_mask[1])
        z_bass = self.bass_decoder(x[2], memory[2], src_mask[2], trg_mask[2])
        z_strings = self.strings_decoder(x[3], memory[3], src_mask[3], trg_mask[3])
        return torch.stack([z_drums, z_guitar, z_bass, z_strings])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj_drums = nn.Linear(d_model, vocab)
        self.proj_guitar = nn.Linear(d_model, vocab)
        self.proj_bass = nn.Linear(d_model, vocab)
        self.proj_strings = nn.Linear(d_model, vocab)

    def forward(self, x):
        out_drums = F.log_softmax(self.proj_drums(x[0]), dim=-1)
        out_guitar = F.log_softmax(self.proj_guitar(x[1]), dim=-1)
        out_bass = F.log_softmax(self.proj_bass(x[2]), dim=-1)
        out_strings = F.log_softmax(self.proj_strings(x[3]), dim=-1)
        return torch.stack([out_drums, out_guitar, out_bass, out_strings])




def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, emb_pos):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.emb_pos = emb_pos

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.emb_pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


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


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N, emb_pos):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.emb_pos = emb_pos

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.emb_pos(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
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


def make_model(src_vocab, tgt_vocab, N=config["model"]["layers"],
               d_model=config["model"]["d_model"], d_ff=config["model"]["d_model"]*config["model"]["ff_mul"],
               h=config["model"]["heads"], dropout=0.1, device=config["train"]["device"]):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
    position = PositionalEncoding(d_model, dropout).to(device)

    enc_emb_pos = nn.Sequential(Embeddings(d_model, src_vocab), c(position)).to(device)
    encoder = MusicEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, enc_emb_pos).to(device)
    dec_emb_pos = nn.Sequential(Embeddings(d_model, tgt_vocab).to(device), c(position)).to(device)
    decoder = MusicDecoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N, dec_emb_pos).to(device)
    generator = Generator(d_model, tgt_vocab).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for model in [encoder, decoder, generator]:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return encoder, decoder, generator


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[..., :-1]
            self.trg_y = trg[..., 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


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
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


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
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = torch.LongTensor(data.long()).to(config["train"]["device"])
        tgt = torch.LongTensor(data.long()).to(config["train"]["device"])
        yield Batch(src, tgt, 0)


def data_gen_2(loader):
    for batch in loader:
        src, trg, _, _, _ = batch
        src = src[:, 0, 0, :]
        trg = trg[:, 0, 0, :]
        src = torch.LongTensor(src.long()).to(config["train"]["device"])
        trg = torch.LongTensor(trg.long()).to(config["train"]["device"])
        yield Batch(src, trg, config["tokens"]["pad"])


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, enc_opt=None, dec_opt=None):
        self.generator = generator
        self.criterion = criterion
        self.enc_opt = enc_opt
        self.dec_opt = dec_opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        n_batch, n_track, seq_len, d_model = x.shape
        x = x.reshape(n_batch, n_track*seq_len, d_model)
        y = y.reshape(n_batch, n_track*seq_len)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if self.generator.training:
            loss.backward()
            if self.enc_opt is not None:
                self.enc_opt.step()
                self.enc_opt.optimizer.zero_grad()
            if self.dec_opt is not None:
                self.dec_opt.step()
                self.dec_opt.optimizer.zero_grad()
        # compute accuracy
        pad_mask = y != self.criterion.padding_idx
        accuracy = ((torch.max(x, dim=-1).indices == y) & pad_mask).sum().item()
        accuracy = accuracy / pad_mask.sum().item()
        return loss.item(), accuracy  #  * norm, accuracy


def run_epoch(data_iter, encoder, decoder, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_accuracy = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        latent = encoder.forward(batch.src, batch.src_mask)
        out = decoder.forward(batch.trg, latent, batch.src_mask, batch.trg_mask)
        loss, accuracy = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_accuracy += accuracy
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens, total_accuracy / (i + 1)


# TODO IN ORDER TO WORK
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

###################
# TRAINING RANDOM #
###################
if True:
    if __name__ == "__main__":
        device = config["train"]["device"]
        # Train the simple copy task.
        V = 11
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(device)
        encoder, decoder, generator = make_model(V, V, N=2, device=device)
        enc_opt = NoamOpt(512, 1, 400,  # TODO parametrize d_model
                          torch.optim.Adam(encoder.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        dec_opt = NoamOpt(512, 1, 400,  # TODO parametrize d_model
                          torch.optim.Adam(list(decoder.parameters()) + list(generator.parameters()),
                                           lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(10):
            encoder.train()
            decoder.train()
            run_epoch(data_gen(V, 30, 20), encoder, decoder,
                      SimpleLossCompute(generator, criterion, enc_opt, dec_opt))
            encoder.eval()
            decoder.eval()
            print(run_epoch(data_gen(V, 30, 5), encoder, decoder,
                            SimpleLossCompute(generator, criterion, None, None)))


        def greedy_decode(encoder, decoder, generator, src, src_mask, max_len, start_symbol):
            memory = encoder.forward(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
            for i in range(max_len - 1):
                out = decoder(Variable(ys), memory, src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
                prob = generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            return ys


        encoder.eval()
        decoder.eval()
        src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])).to(device)
        src_mask = Variable(torch.ones(1, 1, 10)).to(device)
        print("Given", src)
        print("Got", greedy_decode(encoder, decoder, generator, src, src_mask, max_len=10, start_symbol=1))

        ########################################### DRAW ######################################################################

        tgt_sent = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sent = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


        def draw(data, x, y, ax):
            seaborn.heatmap(data.cpu(),
                            xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                            cbar=False, ax=ax)


        for layer in range(1, 2, 2):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for h in range(4):
                draw(encoder.layers[layer].self_attn.attn[0, h].data,
                     sent, sent if h == 0 else [], ax=axs[h])
            plt.show()

        for layer in range(1, 2, 2):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Decoder Self Layer", layer + 1)
            for h in range(4):
                draw(decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
                     tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
            plt.show()
            print("Decoder Src Layer", layer + 1)
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            for h in range(4):
                draw(decoder.layers[layer].src_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
                     sent, tgt_sent if h == 0 else [], ax=axs[h])
            plt.show()

##################
# TRAINING DRUMS #
##################
if False:
    if __name__ == "__main__":
        device = config["train"]["device"]
        dataset = SongIterator(dataset_path=config["paths"]["dataset"],
                               test_size=config["train"]["test_size"],
                               batch_size=config["train"]["batch_size"],
                               n_workers=config["train"]["n_workers"])
        tr_loader, ts_loader = dataset.get_loaders()
        # Train the simple copy task.
        V = config["tokens"]["vocab_size"]
        criterion = LabelSmoothing(size=V, padding_idx=config["tokens"]["pad"], smoothing=0.1).to(device)
        encoder, decoder, generator = make_model(V, V, N=2, device=device)
        enc_opt = NoamOpt(config["model"]["d_model"], 1, 400,
                          torch.optim.Adam(encoder.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        dec_opt = NoamOpt(config["model"]["d_model"], 1, 400,  # TODO parametrize d_model
                          torch.optim.Adam(list(decoder.parameters()) + list(generator.parameters()),
                                           lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(10):
            encoder.train()
            decoder.train()
            run_epoch(data_gen_2(tr_loader), encoder, decoder,
                      SimpleLossCompute(generator, criterion, enc_opt, dec_opt))
            encoder.eval()
            decoder.eval()
            print(run_epoch(data_gen_2(ts_loader), encoder, decoder,
                            SimpleLossCompute(generator, criterion, None, None)))


        def greedy_decode(encoder, decoder, generator, src, src_mask, max_len, start_symbol):
            memory = encoder.forward(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
            for i in range(max_len - 1):
                out = decoder(Variable(ys), memory, src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
                prob = generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            return ys


        encoder.eval()
        decoder.eval()
        src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])).to(device)
        src_mask = Variable(torch.ones(1, 1, 10)).to(device)
        print("Given", src)
        print("Got", greedy_decode(encoder, decoder, generator, src, src_mask, max_len=10, start_symbol=1))

        ########################################### DRAW ######################################################################

        tgt_sent = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sent = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


        def draw(data, x, y, ax):
            seaborn.heatmap(data.cpu(),
                            xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                            cbar=False, ax=ax)


        for layer in range(1, 2, 2):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for h in range(4):
                draw(encoder.layers[layer].self_attn.attn[0, h].data,
                     sent, sent if h == 0 else [], ax=axs[h])
            plt.show()

        for layer in range(1, 2, 2):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Decoder Self Layer", layer + 1)
            for h in range(4):
                draw(decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
                     tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
            plt.show()
            print("Decoder Src Layer", layer + 1)
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            for h in range(4):
                draw(decoder.layers[layer].src_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
                     sent, tgt_sent if h == 0 else [], ax=axs[h])
            plt.show()
