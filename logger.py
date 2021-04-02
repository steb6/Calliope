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
        pass
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
               mode + "encoder attention loss": losses[2],
               mode + "decoder attention loss": losses[3],
               mode + "drums loss": losses[4],
               mode + "guitar loss": losses[5],
               mode + "bass loss": losses[6],
               mode + "strings loss": losses[7]}
        if config["train"]["aae"] and len(losses) == 14:
            log[mode + "discriminator real score"] = losses[8]
            log[mode + "discriminator fake score"] = losses[9]
            log[mode + "generator score"] = losses[10]
            log[mode + "loss_critic"] = losses[11]
            log[mode + "loss_gen"] = losses[12]
            log[mode + "wasserstain distance"] = losses[13]
        wandb.log(log)

    @staticmethod
    def log_stuff(step, lr, latent, disc=None, gen=None, beta=None, prior=None, tf_prob=None):
        log = {"stuff/step": step, "stuff/lr": lr, "stuff/latent": latent[0]}
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
        # seq, tracks, layer, batch, heads, attn1, attn2
        # enc_self_weights = torch.mean(enc_self_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
        # dec_self_weights = torch.mean(dec_self_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
        # dec_src_weights = torch.mean(dec_src_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
        # tracks, layer, heads, attn1, attn2

        instruments = ["drums", "bass", "guitar", "strings"]
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
                # true_height = weight[0][0][0].detach().cpu().numpy()
                # true_width = weight.shape[-1]
                # aspect = true_width/true_height
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
    def log_memories(e_mems, e_cmems, d_mems, d_cmems):  # track layer batch seq dim
        e_mems = e_mems[:, :, 0, ...].transpose(-2, -1).detach().cpu().numpy()
        e_cmems = e_cmems[:, :, 0, ...].transpose(-2, -1).detach().cpu().numpy()
        d_mems = d_mems[:, :, 0, ...].transpose(-2, -1).detach().cpu().numpy()
        d_cmems = d_cmems[:, :, 0, ...].transpose(-2, -1).detach().cpu().numpy()

        instruments = ["drums", "bass", "guitar", "strings"]

        memories = [e_mems, e_cmems, d_mems, d_cmems]
        mem_name = ["encoder memory", "encoder compressed memory", "decoder memory", "decoder compressed memory"]

        for i, mem in enumerate(memories):
            T = []
            condition1 = range(len(instruments))
            condition2 = range(config["model"]["layers"])
            for c1 in np.unique(condition1):
                for c2 in np.unique(condition2):
                    T.append({'instrument': c1,
                              'layer': c2,
                              'picture': mem[c1, c2, ...],
                              })
            df = pd.DataFrame(T)
            true_height = mem.shape[-2]
            true_width = mem.shape[-1]
            aspect = true_width/true_height
            grid = sns.FacetGrid(df, row='instrument', col='layer', aspect=aspect)
            grid.map(lambda x, **kwargs: (sns.heatmap(x.values[0]), plt.grid(False)), 'picture')
            title = mem_name[i]
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
