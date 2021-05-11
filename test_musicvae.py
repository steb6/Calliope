import torch
import os
from test import Tester
from iterate_dataset import SongIterator
from config import config, set_freer_gpu, n_bars
from tqdm import tqdm
from utilities import Batch
from loss_computer import SimpleLossCompute, LabelSmoothing
import copy
from create_bar_dataset import NoteRepresentationManager
import os

if __name__ == "__main__":
    # load models
    set_freer_gpu()
    print("Loading models")
    import wandb

    wandb.init()
    wandb.unwatch()
    # checkpoint_name = os.path.join("remote", "fix")
    # BEST 1 bar
    # checkpoint_name = "/data/musae3.0/musae_model_checkpoints_1/2021-04-20_00-50-29/530000"
    # BEST 2 BAR
    # checkpoint_name = "/data/musae3.0/musae_model_checkpoints_2/2021-04-16_22-59-09/480000"
    # BEST 16 BAR
    checkpoint_name = "/data/musae3.0/musae_model_checkpoints_16/2021-04-16_22-53-12/50000"
    # LCOAL 2 bar
    # checkpoint_name = "pretrained" + os.sep + "1-bar"

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
    ts_loader = dataset.get_loader()

    print("tr_loader_length", len(ts_loader))


    def compute_accuracy_instrument(x, y, pad):
        assert x.shape == y.shape

        y_pad = y[:, 0, ...] != pad
        true = ((x[:, 0, ...] == y[:, 0, ...]) & y_pad).sum()
        count = y_pad.sum().item()
        drum_acc = true / count

        y_pad = y[:, 1, ...] != pad
        true = ((x[:, 1, ...] == y[:, 1, ...]) & y_pad).sum()
        count = y_pad.sum().item()
        guitar_acc = true / count

        y_pad = y[:, 2, ...] != pad
        true = ((x[:, 2, ...] == y[:, 2, ...]) & y_pad).sum()
        count = y_pad.sum().item()
        bass_acc = true / count

        y_pad = y[:, 3, ...] != pad
        true = ((x[:, 3, ...] == y[:, 3, ...]) & y_pad).sum()
        count = y_pad.sum().item()
        strings_acc = true / count

        return drum_acc, guitar_acc, bass_acc, strings_acc


    print("Reconstructing")
    i = 0
    total_accuracies = 0
    total_drums_acc = 0
    total_guitar_acc = 0
    total_bass_acc = 0
    total_strings_acc = 0

    criterion = LabelSmoothing(size=config["tokens"]["vocab_size"], padding_idx=0, smoothing=0.1).to(
        config["train"]["device"])

    nm = NoteRepresentationManager()

    with torch.no_grad():
        for x in tqdm(ts_loader):
            # TODO if GREEDY
            # origin, recon, accuracy = tester.reconstruct(x, nm)

            # TODO else NEXT STEP
            srcs, trgs = x
            srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
            trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)  # invert batch and bars

            latent = None
            batches = [Batch(srcs[i], trgs[i], config["tokens"]["pad"]) for i in range(n_bars)]
            ############
            # ENCODING #
            ############
            latents = []
            for batch in batches:
                latent = tester.encoder(batch.src, batch.src_mask)
                latents.append(latent)

            ############
            # COMPRESS #
            ############
            old_batches = copy.deepcopy(batches)
            if config["train"]["compress_latents"]:
                latent = tester.latent_compressor(latents)  # in: 3, 4, 200, 256, out: 3, 256

            if config["train"]["compress_latents"]:
                latents = tester.latent_decompressor(latent)  # in 3, 256, out: 3, 4, 200, 256
                for k in range(n_bars):
                    batches[k].src_mask = batches[k].src_mask.fill_(True)[:, :, :, :20]

            outs = []
            for batch, latent in zip(batches, latents):
                out = tester.decoder(batch.trg, latent, batch.src_mask, batch.trg_mask)
                outs.append(out)

            # Format results
            outs = torch.stack(outs, dim=0)
            #
            # # Loss and accuracy
            trg_ys = torch.stack([batch.trg_y for batch in batches])
            bars, n_track, n_batch, seq_len, d_model = outs.shape
            outs = outs.permute(1, 2, 0, 3, 4).reshape(n_track, n_batch, bars * seq_len, d_model)  # join bars
            trg_ys = trg_ys.permute(1, 2, 0, 3).reshape(n_track, n_batch, bars * seq_len)

            # loss, accuracy = SimpleLossCompute(tester.generator, criterion)(outs, trg_ys,
            #                                                                 batch.ntokens)  # join instr
            outs = tester.generator(outs)
            outs = torch.max(outs, dim=-1).indices

            outs = outs.reshape(4, config["train"]["batch_size"], n_bars, 199).transpose(0, 1)
            trg_ys = trg_ys.reshape(4, config["train"]["batch_size"], n_bars, 199).transpose(0, 1)

            recon = nm.reconstruct_music(outs[0])
            origin = nm.reconstruct_music(trg_ys[0])
            # # TODO endif

            accuracies = [0, 0, 0, 0]
            for e in range(4):
                i_original = copy.deepcopy(origin)
                i_original.tracks = [i_original.tracks[e]]
                i_reconstructed = copy.deepcopy(recon)
                i_reconstructed.tracks = [i_reconstructed.tracks[e]]
                try:
                    i_original = i_original.to_pianoroll_representation(encode_velocity=False)
                    i_reconstructed = i_reconstructed.to_pianoroll_representation(encode_velocity=False)
                except Exception:
                    continue
                all = i_original.size
                if i_original.size > i_reconstructed.size:
                    i_original = i_original[:i_reconstructed.shape[0], :i_reconstructed.shape[1]]
                if i_original.size < i_reconstructed.size:
                    i_reconstructed = i_reconstructed[:i_original.shape[0], :i_original.shape[1]]
                true = (i_original == i_reconstructed).sum()
                accuracy = true / all
                accuracies[e] = accuracy

            total_drums_acc += accuracies[0]
            total_guitar_acc += accuracies[1]
            total_bass_acc += accuracies[2]
            total_strings_acc += accuracies[3]

            total_accuracies += accuracy
            i += 1

            if i % (10 if n_bars == 16 else 100) == 0:
                print("Total accuracy:", total_accuracies / i)
                print("Drums accuracy:", total_drums_acc / i)
                print("Guitar accuracy:", total_guitar_acc / i)
                print("Bass accuracy:", total_bass_acc / i)
                print("Strings accuracy:", total_strings_acc / i)

    print("Total accuracy:", total_accuracies / i)
    print("Drums accuracy:", total_drums_acc / i)
    print("Guitar accuracy:", total_guitar_acc / i)
    print("Bass accuracy:", total_bass_acc / i)
    print("Strings accuracy:", total_strings_acc / i)
