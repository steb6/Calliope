import wandb
from config import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from utilities import midi_to_wav
sns.set_theme()


class Logger:
    def __init__(self):
        self.reconstructions = []

    def log_reconstruction_accuracy(self, accuracy):
        self.reconstructions.append(accuracy)
        log = {"stuff/reconstruction accuracy": accuracy}
        print(self.reconstructions)
        wandb.log(log)

    @staticmethod
    def log_losses(losses, train):
        mode = "train/" if train else "eval/"
        log = {mode + "loss": losses[0],
               mode + "accuracy": losses[1],
               mode + "drums loss": losses[2],
               mode + "guitar loss": losses[3],
               mode + "bass loss": losses[4],
               mode + "strings loss": losses[5]}
        if config["train"]["aae"] and len(losses) == 12:
            log[mode + "discriminator real score"] = losses[6]
            log[mode + "discriminator fake score"] = losses[7]
            log[mode + "generator score"] = losses[8]
            log[mode + "loss_critic"] = losses[9]
            log[mode + "loss_gen"] = losses[10]
            log[mode + "wasserstain distance"] = losses[11]
        wandb.log(log)

    @staticmethod
    def log_stuff(lr, latent, disc=None, gen=None, beta=None, prior=None, tf_prob=None):
        log = {"stuff/lr": lr, "stuff/latent": latent[0]}
        if config["train"]["aae"]:
            log["stuff/disc lr"] = disc
            log["stuff/gen lr"] = gen
            log["stuff/beta"] = beta
            log["stuff/prior"] = prior.detach().cpu().numpy()[0]
        log["stuff/tf_prob"] = tf_prob
        wandb.log(log)

    @staticmethod
    def log_examples(e_in, d_in):
        enc_input = e_in.transpose(0, 2)[0].detach().cpu().numpy()
        dec_input = d_in.transpose(0, 2)[0].detach().cpu().numpy()
        columns = ["Encoder Input: " + str(enc_input.shape),
                   "Decoder Input: " + str(dec_input.shape)]
        inputs = (enc_input, dec_input)
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"Inputs": table})

    @staticmethod
    def log_attn_heatmap(enc_self_weights, dec_self_weights, dec_src_weights):
        instruments = ["drums" , "guitar", "bass", "strings"]
        weights = [enc_self_weights, dec_self_weights, dec_src_weights]
        weights_name = ["encoder self attention", "decoder self attention", "decoder source weights"]

        for i, instrument in enumerate(instruments):
            for w, weight in enumerate(weights):
                T = []
                condition1 = range(config["model"]["layers"])
                condition2 = range(config["model"]["heads"])
                for c1 in np.unique(condition1):
                    for c2 in np.unique(condition2):
                        T.append({'layer': c1,
                                  'head': c2,
                                  'picture': weight[i][c1][c2].detach().cpu().numpy(),
                                  })
                df = pd.DataFrame(T)
                if w != 2:
                    grid = sns.FacetGrid(df, row='layer', col='head',  # aspect=aspect,
                                         row_order=list(reversed(range(config["model"]["layers"]))))
                else:
                    grid = sns.FacetGrid(df, row='layer', col='head',
                                         row_order=list(reversed(range(config["model"]["layers"]))))
                grid.map(lambda x, **kwargs: (sns.heatmap(x.values[0]), plt.grid(False)), 'picture')
                title = instrument+' '+weights_name[w]
                wandb.log({title: [wandb.Image(plt, caption=title)]})
                plt.close()

    @staticmethod
    def log_latent(latent):  # track layer batch seq dim
        sns.heatmap(latent)
        title = "latent"
        wandb.log({title: [wandb.Image(plt, caption="latent")]})
        plt.close()

    @staticmethod
    def log_songs(prefix, songs, names, log_name):
        log = []
        for song, name in zip(songs, names):
            song.write_midi(os.path.join(wandb.run.dir, prefix + name + ".mid"))
            midi_to_wav(os.path.join(wandb.run.dir, prefix + name + ".mid"),
                        os.path.join(wandb.run.dir, prefix + name + ".wav"))
            log.append(wandb.Audio(os.path.join(wandb.run.dir, prefix + name + ".wav"),
                                   caption="original", sample_rate=32))

        wandb.log({log_name: log})
