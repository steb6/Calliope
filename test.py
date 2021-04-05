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


class Tester:
    def __init__(self, encoder, latent_compressor, latent_decompressor, decoder, generator):
        self.encoder = encoder.eval()
        self.latent_compressor = latent_compressor.eval()
        self.latent_decompressor = latent_decompressor.eval()
        self.decoder = decoder.eval()
        self.generator = generator.eval()

    def interpolation(self, note_manager, first, second):
        # Encode first
        srcs, _, src_masks, _, _ = first

        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"])[:1].transpose(0, 2)

        first_latents = []
        for src, src_mask in zip(srcs, src_masks):
            latent = self.encoder(src, src_mask)
            first_latents.append(latent)
        first_latent = self.latent_compressor(first_latents)

        # Encode second
        srcs, _, src_masks, _, _ = second
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"])[:1].transpose(0, 2)

        second_latents = []
        for src, src_mask in zip(srcs, src_masks):
            latent = self.encoder(src, src_mask)
            second_latents.append(latent)
        second_latent = self.latent_compressor(second_latents)

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

        # src of batch, first batch, first bar
        one = note_manager.reconstruct_music(first[0][0, :, :1, :].detach().cpu().numpy())
        full = note_manager.reconstruct_music(outs)
        two = note_manager.reconstruct_music(second[0][0, :, :1, :].detach().cpu().numpy())

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
        srcs, trgs, _, _, _ = batch
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

        accuracy = compute_accuracy(outs[0, :, :, 1:], batch.trg_y, config["tokens"]["pad"]).item()
        print("Reconstruction accuracy:", accuracy)

        # outs = outs.transpose(0, 2)[0].cpu().numpy()[0]  # invert bars and batch and select first batch
        # srcs = srcs.transpose(0, 2)[0].cpu().numpy()

        original = note_manager.reconstruct_music(srcs[0].cpu().numpy())
        reconstructed = note_manager.reconstruct_music(outs[0].cpu().numpy())
        return original, reconstructed, accuracy


if __name__ == "__main__":
    # load models
    print("Loading models")
    run_name = "remote" if not remote else "/data/musae3.0/musae_model_checkpoints_8/2021-03-03_12-23-00"
    run_batch = "9000" if not remote else "9000"

    checkpoint_name = os.path.join("musae_model_checkpoints_8", run_name, run_batch)

    tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                    torch.load(checkpoint_name + os.sep + "decoder.pt"))

    # load songs
    print("Creating iterator")
    dataset = SongIterator(dataset_path=config["paths"]["dataset"],
                           test_size=0.3,
                           batch_size=config["train"]["batch_size"],
                           n_workers=config["train"]["n_workers"])
    tr_loader, ts_loader = dataset.get_loaders()

    print("tr_loader_length", len(tr_loader))
    print("ts_loader_length", len(ts_loader))

    song1 = tr_loader.__iter__().__next__()
    song2 = tr_loader.__iter__().__next__()

    # load representation manager
    nm = NoteRepresentationManager()

    print("Reconstructing")
    with torch.no_grad():
        origin, recon = tester.reconstruct(song1, nm)
        # gen = tester.generate(nm)

    # gen.write_midi("test" + os.sep + "generated.mid")
    origin.write_midi("test" + os.sep + "original.mid")
    recon.write_midi("test" + os.sep + "reconstructed.mid")
