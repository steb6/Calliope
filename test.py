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


class Tester:
    def __init__(self, encoder, latent_compressor, decoder):
        self.encoder = encoder.eval()
        self.latent_compressor = latent_compressor.eval()
        self.decoder = decoder.eval()

    def interpolation(self, note_manager, first, second):
        # Encode first
        e_mems, e_cmems, _, _ = get_memories()
        srcs, _, src_masks, _, _ = first
        latent = None
        srcs = srcs[0].unsqueeze(0)  # select first song of the batch
        src_masks = src_masks[0].unsqueeze(0)  # add batch dimension of size 1
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, sw = self.encoder(src, src_mask, e_mems, e_cmems)
            e_mems = e_mems.detach()
            e_cmems = e_cmems.detach()
        first_latent = self.latent_compressor(latent)

        # Encode second
        e_mems, e_cmems, _, _ = get_memories()
        srcs, _, src_masks, _, _ = second
        srcs = srcs[0].unsqueeze(0)  # select first song of the batch
        src_masks = src_masks[0].unsqueeze(0)  # add batch dimension of size 1
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, sw = self.encoder(src, src_mask, e_mems, e_cmems)
            e_mems = e_mems.detach()
            e_cmems = e_cmems.detach()
        second_latent = self.latent_compressor(latent)

        # Create interpolation
        latents = []
        timesteps = config["train"]["interpolation_timesteps"]+2
        for i in range(timesteps):
            i += 1
            first_amount = ((timesteps-i)/(timesteps-1))
            second_amount = ((i - 1) / (timesteps - 1))
            first_part = first_latent * first_amount
            second_part = second_latent * second_amount
            interpolation = first_part + second_part
            latents.append(interpolation)

        # Create interpolated song
        steps = config["train"]["interpolation_timesteps_length"]
        outs = []
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        for latent in latents:
            dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
                                        config["model"]["d_model"])
            step_outs = self.greedy_decode(dec_latent, steps, "interpolating")
            outs = outs + step_outs
        outs = torch.stack(outs)
        outs = outs[:, :, 0, :]
        outs = outs.transpose(0, 1).cpu().numpy()

        one = note_manager.reconstruct_music(first[0][0].detach().cpu().numpy())  # src of batch, first batch
        full = note_manager.reconstruct_music(outs)
        two = note_manager.reconstruct_music(second[0][0].detach().cpu().numpy())

        return one, full, two

    def greedy_decode(self, latent, n_bars, desc):
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        for _ in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            for _ in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                trg_mask = create_trg_mask(trg.cpu().numpy())
                out, _, _, _, _, _ = self.decoder(trg, None, None, latent, d_mems, d_cmems)
                out = torch.max(out, dim=-2).indices
                out = out.permute(2, 0, 1)
                trg = torch.cat((trg, out[..., -1:]), dim=-1)
            trg_mask = create_trg_mask(trg.cpu().numpy())
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, None, None, latent, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.permute(2, 0, 1)
            outs.append(out)
        return outs

    def beam_search_decode(self, latent, n_bars, desc, k=4):
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        for _ in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            candidates = [(trg, 0.)]
            for _ in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                new_candidates = []
                for candidate in candidates:
                    trg, score = candidate
                    # trg_mask = create_trg_mask(trg.cpu().numpy())
                    out, _, _, _, _, _ = self.decoder(trg, None, None, latent, d_mems, d_cmems)
                    out = torch.topk(out, k, dim=-2)
                    tok = out.indices
                    for i in range(k):
                        c = tok[:, :, i, :]
                        o = c.permute(2, 0, 1)
                        t = (torch.cat((trg, o[..., -1:]), dim=-1), score - torch.sum(out.values[:, -1:, i, :]).item())
                        new_candidates.append(t)
                new_candidates = sorted(new_candidates, key=lambda x: x[1])  # TODO is this sorting right?
                new_candidates = new_candidates[:k]
                candidates = new_candidates
            # trg_mask = create_trg_mask(trg.cpu().numpy())
            trg = candidates[0][0]
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, None, None, latent, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.permute(2, 0, 1)
            outs.append(out)
        return outs

    def generate(self, note_manager):  # TODO CHECK THIS
        latent = get_prior((1, config["model"]["n_latents"]*config["model"]["d_model"])).to(config["train"]["device"])
        dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
                                    config["model"]["d_model"])
        outs = self.beam_search_decode(dec_latent, config["train"]["generated_iterations"], "generate")  # TODO careful
        outs = torch.stack(outs)
        outs = outs.transpose(0, 2)[0].cpu().numpy()
        return note_manager.reconstruct_music(outs)

    def reconstruct(self, batch, note_manager):
        srcs, trgs, src_masks, trg_masks, _ = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"]).transpose(0, 2)
        e_mems, e_cmems, d_mems, d_cmems = get_memories()
        latent = None
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, _, _ = self.encoder(src, src_mask, e_mems, e_cmems)
        latent = self.latent_compressor(latent)
        dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
                                    config["model"]["d_model"])

        outs = self.beam_search_decode(dec_latent, len(trgs), "reconstruct")  # TODO careful
        # outs = []
        # for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
        #     out, self_weight, src_weight, d_mems, d_cmems, d_attn_loss = self.decoder(trg, trg_mask, src_mask,
        #                                                                               dec_latent,
        #                                                                               d_mems, d_cmems)
        #     out = torch.max(out, dim=-2).indices
        #     out = out.permute(2, 0, 1)
        #     outs.append(out)

        outs = torch.stack(outs)
        outs = outs.transpose(0, 2)[0].cpu().numpy()
        srcs = srcs.transpose(0, 2)[0].cpu().numpy()
        original = note_manager.reconstruct_music(srcs)
        reconstructed = note_manager.reconstruct_music(outs)
        return original, reconstructed


if __name__ == "__main__":
    # load models
    run_name = "2021-02-25_15-52-39"
    run_batch = "11500"

    checkpoint_name = os.path.join("musae_model_checkpoints", run_name, run_batch)

    tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                    torch.load(checkpoint_name + os.sep + "decoder.pt"))

    # load songs
    dataset = SongIterator(dataset_path=config["paths"]["dataset"],
                           test_size=config["train"]["test_size"],
                           batch_size=config["train"]["batch_size"],
                           n_workers=config["train"]["n_workers"])
    _, ts_loader = dataset.get_loaders()

    song1 = ts_loader.__iter__().__next__()
    song2 = ts_loader.__iter__().__next__()

    # load representation manager
    nm = NoteRepresentationManager()

    with torch.no_grad():
        origin, recon = tester.reconstruct(song1, nm)
        gen = tester.generate(nm)

    gen.write_midi("test" + os.sep + "generated.mid")
    origin.write_midi("test" + os.sep + "original.mid")
    recon.write_midi("test" + os.sep + "reconstructed.mid")
