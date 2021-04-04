from config import config
import torch
import numpy as np
from torch.nn import functional as f
import os
import subprocess
from config import remote
from torch.autograd import Variable
import copy


def get_prior(shape):
    return Variable(torch.randn(*shape) * 5.).to(config["train"]["device"])  # single gaussian


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # TODO MASK
        # self.src_mask = (src != pad)
        # self.src_mask = self.src_mask[:, :, :, None] * self.src_mask[:, :, None, :]
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


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    sm = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sm) == 0


def min_max_scaling(value, old_interval, new_interval):
    """
    It scales a value with range [mn, mx] into a int value with range [a, b]
    """
    mn, mx = old_interval
    a, b = new_interval
    return round((((value - mn) * (b - a)) / (mx - mn)) + a)


def midi_to_wav(input_file, output_file):
    """
    - Manual is available in
        https://github.com/FluidSynth/fluidsynth/wiki/UserManual
    - Sound font can be downloaded from
        http://timtechsoftware.com/ad.html?keyword=sf2%20format?file_name=the%20General%20MIDI%20Soundfont?file_url=uploads/
        GeneralUser_GS_SoftSynth_v144.sf2
    - To install FluidSync for windows, download the executable, for unix use conda
    """
    subprocess.call(["fluidsynth" if remote else os.path.join("fl", "bin", "fluidsynth"),
                     "-F", output_file,
                     # "-i", "-n", "-T", "wav",  # those seems to be useless
                     # "-q",  # activate quiet mode
                     "-r", "8000",
                     # "-T", "raw",  # audio type
                     "sound_font.sf2" if remote else os.path.join("fl", "sound_font.sf2"),  # a sound font
                     input_file])


# def create_trg_mask(trg):
#     trg_mask = np.full(trg.shape + (trg.shape[-1],), True)
#     for i in range(trg.shape[0]):
#         for b in range(trg.shape[1]):
#             line_mask = trg[i][b] != config["tokens"]["pad"]
#
#             eos_index = np.argmax(trg[i][b] == config["tokens"]["sos"])
#             if eos_index != 0 and eos_index < len(line_mask) - 1:
#                 line_mask[(eos_index+1):] = config["tokens"]["pad"]
#
#             pad_mask = np.matmul(line_mask[:, np.newaxis], line_mask[np.newaxis, :])
#             subsequent_mask = np.expand_dims(np.tril(np.ones((trg.shape[-1], trg.shape[-1]))), (0, 1))
#             subsequent_mask = subsequent_mask.astype(np.bool)
#             trg_mask[i][b] = pad_mask & subsequent_mask
#     trg_mask = torch.BoolTensor(trg_mask).to(config["train"]["device"])
#     return trg_mask