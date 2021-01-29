import wandb
import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def log_losses(losses, lr, train):
        mode = "train/" if train else "eval/"
        log = {"stuff/lr": lr,
               mode + "loss": losses[0],
               mode + "encoder attention loss": losses[1],
               mode + "decoder attention loss": losses[2],
               mode + "drums loss": losses[3],
               mode + "guitar loss": losses[4],
               mode + "bass loss": losses[5],
               mode + "strings loss": losses[6]}
        wandb.log(log)

    @staticmethod
    def log_latent(latent):

        latent = latent.transpose(0, 1).detach().cpu().numpy()  # transpose sequence length and batch

        indices = pd.MultiIndex.from_product((range(latent.shape[0]), range(latent.shape[1]), range(latent.shape[2])),
                                             names=('batch', 'seq_len', 'dim'))
        data = pd.DataFrame(latent.reshape(-1), index=indices, columns=('value',)).reset_index()

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)

        fg = sns.FacetGrid(data, col='batch')
        fg.map_dataframe(draw_heatmap, 'seq_len', 'dim', 'value', cbar=False)
        # plt.show()
        wandb.log({"Latent": [wandb.Image(plt, caption="Latent")]})
        plt.clf()

    @staticmethod
    def log_examples(e_in, d_in, pred, exp, lat=None, z=None, r_lat=None):
        enc_input = e_in.transpose(0, 2)[0].reshape(4, -1).detach().cpu().numpy()
        dec_input = d_in.transpose(0, 2)[0].reshape(4, -1).detach().cpu().numpy()
        predicted = torch.max(pred, dim=-2).indices.permute(0, 2, 1)[0].reshape(4, -1).detach().cpu().numpy()
        expected = exp[0].transpose(0, 1).detach().cpu().numpy()
        columns = ["Encoder Input: " + str(enc_input.shape),
                   "Decoder Input: " + str(dec_input.shape),
                   "Predicted: " + str(predicted.shape),
                   "Expected: " + str(expected.shape)]
        inputs = (enc_input[:, :10], dec_input[:, :10], predicted[:, :10], expected[:, :10])
        if lat is not None and type(lat) is not list:
            latent = lat.detach().cpu().numpy()
            inputs = inputs + (latent,)
            columns.append("Latents: " + str(latent.shape))
        if z is not None:
            zeta = z.detach().cpu().numpy()
            inputs = inputs + (zeta,)
            columns.append("Z: " + str(z.shape))
        if r_lat is not None:
            recon_l = r_lat.detach().cpu().numpy()
            inputs = inputs + (recon_l,)
            columns.append("Real latents: " + str(recon_l.shape))
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"out": table})

    @staticmethod
    def log_attn_heatmap(enc_self_weights, dec_self_weights, dec_src_weights):
        enc_self_img = enc_self_weights.detach().cpu().numpy()
        ax1 = sns.heatmap(enc_self_img)
        wandb.log({"Encoder self attention": [wandb.Image(plt, caption="Encoder self attention")]})
        # plt.show()
        plt.clf()
        dec_self_img = dec_self_weights.detach().cpu().numpy()
        ax2 = sns.heatmap(dec_self_img)
        wandb.log({"Decoder self attention": [wandb.Image(plt, caption="Decoder self attention")]})
        # plt.show()
        plt.clf()
        dec_src_img = dec_src_weights.detach().cpu().numpy()
        ax3 = sns.heatmap(dec_src_img)
        wandb.log({"Decoder source attention": [wandb.Image(plt, caption="Decoder source attention")]})
        # plt.show()
        plt.clf()
