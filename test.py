import torch
import os
from config import config
from tqdm import tqdm
import numpy as np
from utilities import get_prior, Batch
from iterate_dataset import SongIterator
from create_bar_dataset import NoteRepresentationManager
from config import remote, n_bars
import copy
from loss_computer import compute_accuracy
from torch.autograd import Variable
from config import max_bar_length
import matplotlib.pyplot as plt
import dill


class Tester:
    def __init__(self, encoder, latent_compressor, latent_decompressor, decoder, generator):
        self.encoder = encoder.eval()
        self.latent_compressor = latent_compressor.eval()
        self.latent_decompressor = latent_decompressor.eval()
        self.decoder = decoder.eval()
        self.generator = generator.eval()

    def interpolation(self, note_manager, first=None, second=None):
        # Encode first
        if first is not None:
            f, _ = first
            f = torch.LongTensor(f.long()).to(config["train"]["device"])[:1].transpose(0, 2)
            fs = [Batch(f[i], None, config["tokens"]["pad"]) for i in range(n_bars)]

            first_latents = []
            for f in fs:
                latent = self.encoder(f.src, f.src_mask)
                first_latents.append(latent)
            first_latent = self.latent_compressor(first_latents)
        else:
            first_latent = get_prior((1, 256))

        # Encode second
        if second is not None:
            s, _ = second
            s = torch.LongTensor(s.long()).to(config["train"]["device"])[:1].transpose(0, 2)
            ss = [Batch(s[i], None, config["tokens"]["pad"]) for i in range(n_bars)]

            second_latents = []
            for s in ss:
                latent = self.encoder(s.src, s.src_mask)
                second_latents.append(latent)
            second_latent = self.latent_compressor(second_latents)
        else:
            second_latent = get_prior((1, 256))

        # Create interpolation
        latents = []
        timesteps = config["train"]["interpolation_timesteps"] + 2
        for i in range(timesteps):
            i += 1
            first_amount = ((timesteps - i) / (timesteps - 1))
            second_amount = ((i - 1) / (timesteps - 1))
            first_part = first_latent * first_amount
            second_part = second_latent * second_amount
            interpolation = first_part + second_part
            latents.append(interpolation)

        # Create interpolated song
        outs = []
        for latent in latents:
            step_outs = self.greedy_decode2(latent, n_bars, "interpolating")
            outs.append(step_outs)
        outs = torch.stack(outs)
        tot_bars, step_bars, n_track, n_batch, n_tok = outs.shape
        outs = outs.reshape(tot_bars*step_bars, n_track, n_batch, n_tok)
        outs = outs[:, :, 0, :]  # select first batch
        outs = outs.transpose(0, 1).cpu().numpy()  # invert bars and instruments

        # src of batch, first batch
        if first is not None:
            one = note_manager.reconstruct_music(first[0][0, :, :, :].detach().cpu().numpy())
        else:
            one = note_manager.reconstruct_music(outs[:, :2, :])
        full = note_manager.reconstruct_music(outs)
        if second is not None:
            two = note_manager.reconstruct_music(second[0][0, :, :, :].detach().cpu().numpy())
        else:
            two = note_manager.reconstruct_music(outs[:, -2:, :])

        return one, full, two

    @staticmethod
    def ifn(elem, i):
        return None if elem is None else elem[i]

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def greedy_decode(self, encoder, decoder, generator, src, src_mask, max_len, start_symbol):
        # TODO deve funzionare multibar anzichè una barra sola er volta per la compressione e decompressione
        latents = encoder.forward(src, src_mask)

        if config["train"]["compress_latents"]:
            latent = self.latent_compressor([latents])
            latents = self.latent_decompressor(latent)

        ys = torch.ones(4, 1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            trg_mask = Variable(self.subsequent_mask(ys.size(1)).type_as(src.data)).repeat(4, 1, 1, 1)
            out = decoder(Variable(ys), latents[0], src_mask, trg_mask)  # TODO MASK
            prob = generator(out[:, :, -1, :])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=-1)
        return ys

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

    def generate(self, note_manager):  # TODO CHECK THIS
        latent = get_prior((1, config["model"]["d_model"])).to(config["train"]["device"])
        outs = self.greedy_decode2(latent, n_bars, "generate")  # TODO careful
        outs = outs.transpose(0, 2)[0].cpu().numpy()
        return note_manager.reconstruct_music(outs)

    def reconstruct(self, batch, note_manager):
        srcs, trgs = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)  # out: bar, 4, batch, t
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"])[:1].transpose(0, 2)

        # JOIN WITH NEW VERSION
        batches = [Batch(srcs[i], trgs[i], config["tokens"]["pad"]) for i in range(n_bars)]

        latents = []
        for batch in batches:
            latent = self.encoder(batch.src, batch.src_mask)
            latents.append(latent)

        if config["train"]["compress_latents"]:
            latents = self.latent_compressor(latents)  # in: 3, 4, 200, 256, out: 3, 256

        outs = self.greedy_decode2(latents, len(srcs), "reconstruct")  # TODO careful

        trg_y = torch.stack([b.trg_y for b in batches])
        accuracy = compute_accuracy(outs[..., 1:], trg_y, config["tokens"]["pad"]).item()
        print("Reconstruction accuracy:", accuracy)

        outs = outs.transpose(0, 2)[0].cpu().numpy()  # invert bars and batch and select first batch
        srcs = srcs.transpose(0, 2)[0].cpu().numpy()

        original = note_manager.reconstruct_music(srcs)
        reconstructed = note_manager.reconstruct_music(outs)
        return original, reconstructed, accuracy


if __name__ == "__main__":
    # load models
    print("Loading models")
    import wandb
    wandb.init()
    wandb.unwatch()
    checkpoint_name = os.path.join("remote", "90000")

    tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_decompressor.pt"),
                    torch.load(checkpoint_name + os.sep + "decoder.pt"),
                    torch.load(checkpoint_name + os.sep + "generator.pt"))

    # load songs
    print("Creating iterator")
    dataset = SongIterator(dataset_path=config["paths"]["dataset"] + os.sep + "test",
                           batch_size=config["train"]["batch_size"],
                           n_workers=config["train"]["n_workers"])
    tr_loader = dataset.get_loader()

    print("tr_loader_length", len(tr_loader))

    song1 = tr_loader.__iter__().__next__()
    song2 = tr_loader.__iter__().__next__()
    while torch.eq(song1[0], song2[0]).all():
        song2 = tr_loader.__iter__().__next__()

    nm = NoteRepresentationManager()

    # print("Reconstructing")
    # with torch.no_grad():
    #     origin, recon, accuracy = tester.reconstruct(song1, nm)
    #     origin.write_midi("results" + os.sep + "original.mid")
    #     recon.write_midi("results" + os.sep + "reconstructed.mid")
    #
    # print("Generating")
    # with torch.no_grad():
    #     gen = tester.generate(nm)
    #     gen.write_midi("results" + os.sep + "generated.mid")
    #
    # print("Interpolating")
    # with torch.no_grad():
    #     first, full, second = tester.interpolation(nm, song1, song2)
    #     first.write_midi("results" + os.sep + "first.mid")
    #     full.write_midi("results" + os.sep + "full.mid")
    #     second.write_midi("results" + os.sep + "second.mid")
    #
    print("Random interpolating")
    with torch.no_grad():
        first, full, second = tester.interpolation(nm)
        first.write_midi("results" + os.sep + "first.mid")
        full.write_midi("results" + os.sep + "full.mid")
        second.write_midi("results" + os.sep + "second.mid")

    # for i, instrument in zip(range(4), ["drums", "guitar", "bass", "strings"]):
    #     track = copy.deepcopy(origin)
    #     track.tracks = [track.tracks[i]]
    #     track.resolution = track.resolution * n_bars
    #     track.show_score(figsize=(30, 30), clef="bass" if instrument == "bass" else "treble")
    #     plt.savefig("results" + os.sep + "gen_" + instrument + "_spreadsheet")
