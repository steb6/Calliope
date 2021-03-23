import torch
import os
from config import config
from utilities import get_memories
from tqdm import tqdm
import numpy as np
from utilities import get_prior
from iterate_dataset import SongIterator
from create_bar_dataset import NoteRepresentationManager
from utilities import create_trg_mask
from config import remote
import copy
from loss_computer import compute_accuracy


class Tester:
    def __init__(self, encoder, latent_compressor, latent_decompressor, decoder):
        self.encoder = encoder.eval()
        self.latent_compressor = latent_compressor.eval()
        self.latent_decompressor = latent_decompressor.eval()
        self.decoder = decoder.eval()

    def interpolation(self, note_manager, first, second):
        # Encode first
        e_cmems, e_mems = get_memories()
        srcs, _, src_masks, _, _ = first
        latent = None
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"])[:1].transpose(0, 2)
        for src, src_mask in zip(srcs, src_masks):
            latent, e_cmems, e_mems, e_attn_loss, sw = self.encoder(src, src_mask, e_cmems, e_mems)
        first_latent = self.latent_compressor(latent)

        # Encode second
        e_cmems, e_mem = get_memories()
        srcs, _, src_masks, _, _ = second
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"])[:1].transpose(0, 2)
        for src, src_mask in zip(srcs, src_masks):
            latent, e_cmems, e_mems, e_attn_loss, sw = self.encoder(src, src_mask, e_cmems, e_mems)
        second_latent = self.latent_compressor(latent)

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
        steps = config["train"]["interpolation_timesteps_length"]
        outs = []
        limited = []
        for latent in latents:
            latent = self.latent_decompressor(latent)
            latent = latent.transpose(0, 1)  # in: 1 4 200 256   out: 4 1 200 256
            step_outs, limit = self.greedy_decode(latent, steps, "interpolating")
            outs = outs + step_outs
            limited = limited + limit
        outs = torch.stack(outs)
        outs = outs[:, :, 0, :]  # select first batch
        outs = outs.transpose(0, 1).cpu().numpy()  # invert bars and instruments

        limited = torch.stack(limited)
        limited = limited[:, :, 0, :]  # select first batch
        limited = limited.transpose(0, 1).cpu().numpy()  # invert bars and instruments

        # src of batch, first batch, first bar
        one = note_manager.reconstruct_music(first[0][0, :, :1, :].detach().cpu().numpy())
        full = note_manager.reconstruct_music(outs)
        limited = note_manager.reconstruct_music(limited)
        two = note_manager.reconstruct_music(second[0][0, :, :1, :].detach().cpu().numpy())

        return one, full, limited, two

    @staticmethod
    def ifn(elem, i):
        return None if elem is None else elem[i]

    def greedy_decode(self, latent, n_bars, desc, trg_masks=None, src_masks=None):
        outs = []
        outs_limited = []
        cmems, mems = get_memories()

        for i in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])  # track batch tok
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            for j in range(config["model"]["seq_len"] - 1):  # for each token of each bar

                # TODO remove
                if trg_masks is None:
                    trg_mask = create_trg_mask(trg.cpu().numpy())
                else:
                    trg_mask = trg_masks[i, :, :, :(j+1), :(j+1)]
                if src_masks is None:
                    src_mask = None
                else:
                    src_mask = src_masks[i]
                # TODO remove

                out, _, _, _, _, _ = self.decoder(trg, trg_mask, src_mask, latent, cmems, mems)
                out = torch.max(out, dim=-1).indices
                trg = torch.cat((trg, out[..., -1:]), dim=-1)

            # TODO remove
            if trg_masks is None:
                trg_mask = create_trg_mask(trg.cpu().numpy())
            else:
                trg_mask = trg_masks[i]
            if src_masks is not None:
                src_mask = src_masks[i]
            else:
                src_mask = None
            # TODO remove

            out, cmems, mems, _, _, _ = self.decoder(trg, trg_mask, src_mask, latent, cmems, mems)
            out = torch.max(out, dim=-1).indices
            outs.append(copy.deepcopy(out))
            for t in range(len(out)):  # for each track
                eos_indices = torch.nonzero(out[t] == config["tokens"]["eos"])
                if len(eos_indices) == 0:
                    continue
                eos_index = eos_indices[0][1]  # first occurrence, column index
                out[t][eos_index:] = config["tokens"]["pad"]
            outs_limited.append(copy.deepcopy(out))
        return outs, outs_limited

    def generate(self, note_manager):  # TODO CHECK THIS
        latent = get_prior((1, config["model"]["d_model"])).to(config["train"]["device"])
        latent = self.latent_decompressor(latent)
        latent = latent.transpose(0, 1)
        outs, limited = self.greedy_decode(latent, config["train"]["generated_iterations"], "generate")  # TODO careful
        outs = torch.stack(outs)
        limited = torch.stack(limited)
        outs = outs.transpose(0, 2)[0].cpu().numpy()
        limited = limited.transpose(0, 2)[0].cpu().numpy()
        return note_manager.reconstruct_music(outs), note_manager.reconstruct_music(limited)

    def reconstruct(self, batch, note_manager):
        srcs, _, src_masks, trg_masks, _ = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"])[:1].transpose(0, 2)  # out: bar, 4, batch, t
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"])[:1].transpose(0, 2)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"])[:1].transpose(0, 2)

        e_cmems, e_mems = get_memories()
        latent = None
        for src, src_mask in zip(srcs, src_masks):
            latent, e_cmems, e_mems, _, _ = self.encoder(src, src_mask, e_cmems, e_mems)

        if config["train"]["compress_latents"]:
            latent = self.latent_compressor(latent)
            latent = self.latent_decompressor(latent)

        latent = latent.transpose(0, 1)  # in batch, 4, t, d out: 4, batch, t, d

        src_masks = None if not config["train"]["use_src_mask"] else src_masks
        trg_masks = None
        outs, outs_limited = self.greedy_decode(latent, len(srcs), "reconstruct",
                                                trg_masks=trg_masks, src_masks=src_masks)  # TODO careful
        outs = torch.stack(outs)
        outs_limited = torch.stack(outs_limited)

        accuracy = compute_accuracy(outs, srcs, config["tokens"]["pad"]).item()*100
        print("Reconstruction accuracy:", accuracy, "%")

        outs = outs.transpose(0, 2)[0].cpu().numpy()  # invert bars and batch and select first batch
        srcs = srcs.transpose(0, 2)[0].cpu().numpy()
        outs_limited = outs_limited.transpose(0, 2)[0].cpu().numpy()

        original = note_manager.reconstruct_music(srcs)
        reconstructed = note_manager.reconstruct_music(outs)
        limited = note_manager.reconstruct_music(outs_limited)
        return original, reconstructed, limited, accuracy


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
