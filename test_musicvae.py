import torch
import os
from test import Tester
from iterate_dataset import SongIterator
from config import config
from tqdm import tqdm


if __name__ == "__main__":
    # load models
    print("Loading models")
    import wandb
    wandb.init()
    wandb.unwatch()
    checkpoint_name = os.path.join("remote", "fix")

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

    with torch.no_grad():
        for x in tqdm(ts_loader):
            origin, recon, accuracy = tester.reconstruct(x, None)
            drum_acc, guitar_acc, bass_acc, strings_acc = compute_accuracy_instrument(recon, origin,
                                                                                      config["tokens"]["pad"])
            total_drums_acc += drum_acc.item()
            total_guitar_acc += guitar_acc.item()
            total_bass_acc += bass_acc.item()
            total_strings_acc += strings_acc.item()
            total_accuracies += accuracy
            i += 1

    print("Total accuracy:", total_accuracies/i)
    print("Drums accuracy:", total_drums_acc/i)
    print("Guitar accuracy:", total_guitar_acc/i)
    print("Bass accuracy:", total_bass_acc/i)
    print("Strings accuracy:", total_strings_acc/i)
