import wandb
from config import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
sns.set_theme()


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def log_losses(losses, lr, train):
        mode = "train/" if train else "eval/"
        log = {"stuff/lr": lr,
               mode + "loss": losses[0],
               mode + "accuracy": losses[1],
               mode + "encoder attention loss": losses[2],
               mode + "decoder attention loss": losses[3],
               mode + "drums loss": losses[4],
               mode + "guitar loss": losses[5],
               mode + "bass loss": losses[6],
               mode + "strings loss": losses[7]}
        if config["train"]["aae"] and len(losses) == 13:  # TODO careful
            if losses[8] is not None:
                log[mode + "discriminator loss"] = losses[8]
            if losses[9] is not None:
                log[mode + "generator loss"] = losses[9]
            if losses[10] is not None:
                log[mode + "discriminator real score"] = losses[10]
            if losses[11] is not None:
                log[mode + "discriminator fake score"] = losses[11]
            if losses[12] is not None:
                log[mode + "generator fake score"] = losses[12]
        wandb.log(log)

    @staticmethod
    def log_latent(latent):
        latent = latent[0].transpose(0, 1).detach().cpu().numpy()  # transpose sequence length and batch
        wandb.log({"Latent": latent[0]})
        # latent = latent[0].transpose(0, 1).detach().cpu().numpy()  # transpose sequence length and batch
        # T = [{'img': 0, 'picture': latent}]
        # df = pd.DataFrame(T)
        # true_height = latent.shape[-2]
        # true_width = latent.shape[-1]
        # aspect = true_width / true_height
        # # grid = sns.FacetGrid(df, row='img', aspect=aspect)
        # grid = sns.FacetGrid(df, col='img')
        # grid.map(lambda x, **kwargs: (sns.heatmap(x.values[0]), plt.grid(False)), 'picture')
        # wandb.log({"Latent": [wandb.Image(plt, caption="Latent")]})
        # plt.close()
        # # sns.heatmap(latent)
        # # # plt.show()
        # # wandb.log({"Latent": [wandb.Image(plt, caption="Latent")]})
        # # plt.close()

    @staticmethod
    def log_examples(e_in, d_in, pred, exp):
        enc_input = e_in.transpose(0, 2)[0].detach().cpu().numpy()
        dec_input = d_in.transpose(0, 2)[0].detach().cpu().numpy()
        columns = ["Encoder Input: " + str(enc_input.shape),
                   "Decoder Input: " + str(dec_input.shape)]
        inputs = (enc_input, dec_input)
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"Inputs": table})

        # predicted = torch.max(pred[0], dim=-2).indices.permute(0, 1).reshape(4, enc_input.shape[1], -1).transpose(0, 1).detach().cpu().numpy()
        # expected = exp[0].transpose(0, 1).reshape(4, enc_input.shape[1], -1).transpose(0, 1).detach().cpu().numpy()
        # T = []
        # step = 0
        # for b in range(predicted.shape[0]):
        #     T.append({'bar': step,
        #               'picture': predicted[b, :, :],
        #               })
        #     step += 1
        #     T.append({'bar': step,
        #               'picture': expected[b, :, :],
        #               })
        #     step += 1
        # df = pd.DataFrame(T)
        # true_height = predicted.shape[0]
        # true_width = predicted.shape[-1]
        # aspect = true_width/true_height
        # grid = sns.FacetGrid(df, row='bar', aspect=aspect)
        # grid.map(lambda x, **kwargs: (sns.heatmap(x.values[0], annot=True, fmt="d"), plt.grid(False)), 'picture')
        # wandb.log({"predicted and expected": [wandb.Image(plt, caption="predicted and expected")]})
        # plt.close()

    @staticmethod
    def log_attn_heatmap(enc_self_weights, dec_self_weights, dec_src_weights):
        # seq, tracks, layer, batch, heads, attn1, attn2
        enc_self_weights = torch.mean(enc_self_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
        dec_self_weights = torch.mean(dec_self_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
        dec_src_weights = torch.mean(dec_src_weights[:, :, :, 0, ...], dim=0).detach().cpu().numpy()
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
                                  'picture': weight[i, c1, c2, ...],
                                  })
                df = pd.DataFrame(T)
                true_height = weight.shape[-2]
                true_width = weight.shape[-1]
                aspect = true_width/true_height
                if w != 2:
                    grid = sns.FacetGrid(df, row='layer', col='head', aspect=aspect,
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
