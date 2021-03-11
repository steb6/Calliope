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

    def greedy_topk_decode(self, latent, n_bars, desc, k=5):
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        for _ in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])
            trg = torch.LongTensor(trg).to(config["train"]["device"]).unsqueeze(-1).repeat(1, 1, 1, k)
            prob = torch.full_like(trg, 1/k, device=config["train"]["device"], dtype=torch.float32)
            for _ in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                trg_mask = create_trg_mask(trg[..., 0].cpu().numpy())
                out, _, _, _, _, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems, emb_weights=prob)

                top_k = torch.topk(out, config["train"]["top_k_mixed_embeddings"], dim=-1)
                last_trg = top_k.indices  # 8 1 200 5 4
                last_prob = torch.exp(top_k.values)

                last_trg = last_trg[:, :, -1:, :]
                last_prob = last_prob[:, :, -1:, :]

                scaled_prob = torch.zeros_like(last_prob)
                for i, instrument in enumerate(last_prob):
                    for ba, batch in enumerate(instrument):
                        for t, token in enumerate(batch):
                            s = torch.sum(last_prob[i][ba][t])
                            for e, _ in enumerate(token):
                                scaled_prob[i][ba][t][e] = last_prob[i][ba][t][e] / s

                trg = torch.cat((trg, last_trg), dim=-2)
                prob = torch.cat((prob, scaled_prob), dim=-2)

            trg_mask = create_trg_mask(trg[..., 0].cpu().numpy())
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems, emb_weights=prob)
            out = torch.max(out, dim=-1).indices
            for i in range(len(out)):
                eos_indices = torch.nonzero(out[i] == config["tokens"]["eos"])
                if len(eos_indices) == 0:
                    continue
                eos_index = eos_indices[0][1]  # first element, column index
                out[i][0][eos_index:] = config["tokens"]["pad"]  # TODO careful, select just first batch
            outs.append(out)
        return outs

    def greedy_decode(self, latent, n_bars, desc):
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        outs_limited = []
        for _ in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            for _ in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                trg_mask = create_trg_mask(trg.cpu().numpy())
                out, _, _, _, _, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems)
                out = torch.max(out, dim=-1).indices
                trg = torch.cat((trg, out[..., -1:]), dim=-1)
            trg_mask = create_trg_mask(trg.cpu().numpy())
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems)
            out = torch.max(out, dim=-1).indices
            outs.append(copy.deepcopy(out))
            for i in range(len(out)):
                eos_indices = torch.nonzero(out[i] == config["tokens"]["eos"])
                if len(eos_indices) == 0:
                    continue
                eos_index = eos_indices[0][1]  # first element, column index
                out[i][0][eos_index:] = config["tokens"]["pad"]  # TODO careful, select just first batch
            outs_limited.append(out)
        return outs, outs_limited

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

    def individual_beam_search_decode(self, latent, n_bars, desc, k=4):
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        for _ in tqdm(range(n_bars), position=0, leave=True, desc=desc):
            trg = np.full((1, 1), config["tokens"]["sos"])
            pad = np.full((1, 1), config["tokens"]["pad"])
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            pad = torch.LongTensor(pad).to(config["train"]["device"])

            drums_candidates = [(trg, 0.)]
            bass_candidates = [(trg, 0.)]
            guitar_candidates = [(trg, 0.)]
            strings_candidates = [(trg, 0.)]
            instruments = [drums_candidates, bass_candidates, guitar_candidates, strings_candidates]

            for _ in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                idx = 0
                for candidates, name in zip(instruments, ["drums", "bass", "guitar", "strings"]):
                    new_candidates = []
                    for candidate in candidates:
                        trg, score = candidate
                        # if trg[0][-1] == config["tokens"]["pad"]:
                        #     t = (
                        #         torch.cat((trg, pad), dim=-1),
                        #         score + 1.5)
                        #     new_candidates.append(t)
                        #     continue
                        trg_mask = create_trg_mask(trg.unsqueeze(0).cpu().numpy())[0]  # TODO check
                        out, _, _, _, _, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems, just=name)
                        out = torch.topk(out, k, dim=-1)
                        tok = out.indices
                        for i in range(k):
                            c = tok[:, :, i]  # TODO if 3 then stop
                            # if c[0][-1:].item() == config["tokens"]["eos"]:
                            #     t = (
                            #         torch.cat((trg, pad), dim=-1),
                            #         score + 1.5)  # TODO wat
                            # else:
                            t = (
                                torch.cat((trg, c[..., -1:]), dim=-1),
                                score - torch.sum(out.values[:, -1:, i]).item())
                            new_candidates.append(t)
                    new_candidates = sorted(new_candidates, key=lambda x: x[1])  # TODO is this sorting right?
                    new_candidates = new_candidates[:k]
                    instruments[idx] = new_candidates
                    idx += 1
            trg = torch.stack((instruments[0][0][0], instruments[1][0][0], instruments[2][0][0], instruments[3][0][0]))
            trg_mask = create_trg_mask(trg.cpu().numpy())
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, trg_mask, None, latent, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.permute(2, 0, 1)
            outs.append(out)
        return outs

    def generate(self, note_manager):  # TODO CHECK THIS
        latent = get_prior((1, config["model"]["n_latents"] * config["model"]["d_model"])).to(config["train"]["device"])
        dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
                                    config["model"]["d_model"])
        outs = self.greedy_decode(dec_latent, config["train"]["generated_iterations"], "generate")  # TODO careful
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
        # dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
        #                             config["model"]["d_model"])
        dec_latent = latent

        outs, outs_limited = self.greedy_decode(dec_latent, len(trgs), "reconstruct")  # TODO careful
        # outs = []
        # for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
        #     out, self_weight, src_weight, d_mems, d_cmems, d_attn_loss = self.decoder(trg, trg_mask, src_mask,
        #                                                                               dec_latent,
        #                                                                               d_mems, d_cmems)
        #     out = torch.max(out, dim=-1).indices
        #     outs.append(out)

        outs = torch.stack(outs)
        outs_limited = torch.stack(outs_limited)

        outs = outs.transpose(0, 2)[0].cpu().numpy()  # invert bars and batch and select first batch
        srcs = srcs.transpose(0, 2)[0].cpu().numpy()
        outs_limited = outs_limited.transpose(0, 2)[0].cpu().numpy()

        original = note_manager.reconstruct_music(srcs)
        reconstructed = note_manager.reconstruct_music(outs)
        limited = note_manager.reconstruct_music(outs_limited)
        return original, reconstructed, limited


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
