from config import config
import torch
import numpy as np
from torch.nn import functional as f


def get_memories():
    a = 4
    b = config["model"]["layers"]
    c = config["train"]["batch_size"]
    e = config["model"]["d_model"]
    device = config["train"]["device"]
    mem_len = config["model"]["mem_len"]
    cmem_len = config["model"]["cmem_len"]
    e_mems = torch.zeros(a, b, c, mem_len, e, dtype=torch.float32, device=device)
    e_cmems = torch.zeros(a, b, c, cmem_len, e, dtype=torch.float32, device=device)
    d_mems = torch.zeros(a, b, c, mem_len, e, dtype=torch.float32, device=device)
    d_cmems = torch.zeros(a, b, c, cmem_len, e, dtype=torch.float32, device=device)
    return e_mems, e_cmems, d_mems, d_cmems


def create_trg_mask(trg):
    trg_mask = np.full(trg.shape + (trg.shape[-1],), True)
    for b in range(config["train"]["batch_size"]):
        for i in range(4):
            line_mask = trg[b][i] != config["tokens"]["pad"]
            pad_mask = np.matmul(line_mask[:, np.newaxis], line_mask[np.newaxis, :])
            subsequent_mask = np.expand_dims(np.tril(np.ones((trg.shape[-1], trg.shape[-1]))), (0, 1))
            subsequent_mask = subsequent_mask.astype(np.bool)
            trg_mask[b][i] = pad_mask & subsequent_mask
    trg_mask = torch.BoolTensor(trg_mask).to(config["train"]["device"])
    return trg_mask


def pad_attention(attentions):  # pad list of array to be the same size
    length = max([s.shape[-1] for s in attentions])
    for count, attention in enumerate(attentions):
        attentions[count] = f.pad(attention, (length - attention.shape[-1], 0))
    return torch.mean(torch.stack(attentions, dim=0), dim=0)