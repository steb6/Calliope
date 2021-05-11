import torch
import os
from config import config
import numpy as np
from utilities import get_prior
from create_bar_dataset import NoteRepresentationManager
from config import n_bars
from torch.autograd import Variable
from config import max_bar_length
import matplotlib.pyplot as plt
from tqdm import tqdm


class Tester:
    def __init__(self, encoder, latent_compressor, latent_decompressor, decoder, generator):
        self.encoder = encoder.eval()
        self.latent_compressor = latent_compressor.eval()
        self.latent_decompressor = latent_decompressor.eval()
        self.decoder = decoder.eval()
        self.generator = generator.eval()

    @staticmethod
    def ifn(elem, i):
        return None if elem is None else elem[i]

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def greedy_decode2(self, latents, n_bars=n_bars, desc="greedy decoding"):

        if config["train"]["compress_latents"]:
            latents = self.latent_decompressor(latents)

        outs = []
        for i in range(n_bars):
            ys = torch.ones(4, 1, 1).fill_(config["tokens"]["sos"]).long().to(config["train"]["device"])
            for _ in range(max_bar_length - 1):
                trg_mask = Variable(self.subsequent_mask(ys.size(1)).type_as(latents[0].data)).repeat(4, 1, 1, 1).bool()
                src_mask = Variable(torch.ones(4, 1, 1).fill_(True)).to(config["train"]["device"]).bool()
                out = self.decoder(Variable(ys), latents[i], src_mask, trg_mask)  # TODO MASK
                prob = self.generator(out[:, :, -1, :])
                _, next_word = torch.max(prob, dim=-1)

                next_word = next_word.unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=-1)
            outs.append(ys)
        outs = torch.stack(outs)
        return outs

    def generate(self, note_manager):
        latent = get_prior((1, config["model"]["d_model"])).to(config["train"]["device"])
        outs = self.greedy_decode2(latent, n_bars, "generate")
        outs = outs.repeat(4, 1, 1, 1)
        outs = outs.transpose(0, 2)[0].cpu().numpy()
        return note_manager.reconstruct_music(outs)


if __name__ == "__main__":
    # load models
    print("Loading models")
    import wandb
    wandb.init()
    wandb.unwatch()
    checkpoint_name = os.path.join("musae_pretrained", str(n_bars)+"-bar")

    tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_decompressor.pt"),
                    torch.load(checkpoint_name + os.sep + "decoder.pt"),
                    torch.load(checkpoint_name + os.sep + "generator.pt"))

    nm = NoteRepresentationManager()

    print("Generating")
    with torch.no_grad():
        for i in tqdm(range(100)):
            gen = tester.generate(nm)
            gen.write_midi("results" + os.sep + "GEN_generated_"+str(i)+".mid")
            # plt.figure(figsize=(20, 10))
            # gen.show_pianoroll(yticklabel="off", xticklabel="off", label="off")
            # plt.savefig("results" + os.sep + "GEN_gen_pianoroll_"+str(i))
