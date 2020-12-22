from config import set_freer_gpu
from create_dataset import NoteRepresentationManager
from config import config
from compressive_transformer import TransformerAutoencoder
from train import Trainer
from train_aae import AAETrainer
import shutil
import torch
import os

create_dataset = False


# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE\*.py berti@131.114.137.168:MusAE
# TO CONNECT: ssh berti@131.114.137.168
# TO ATTACH TO TMUX: tmux attach -t Training
# TO RESIZE TMUX: tmux attach -d -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi
# TO GET RESULTS: scp -r berti@131.114.137.168:MusAE/2020* C:\Users\berti\PycharmProjects\MusAE\remote_results
def train_aae():
    set_freer_gpu()
    ct = torch.load(os.path.join("remote_results", "checkpoint_16.pt"))
    trainer = AAETrainer(model=ct, dataset_path=config["paths"]["dataset_path"], **config["train"], config=config)
    trainer.train()


def train_ct():
    set_freer_gpu()

    if create_dataset:
        shutil.rmtree(config["paths"]["dataset_path"], ignore_errors=True)
        notes = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
        notes.convert_dataset()

    m = TransformerAutoencoder(**config["model"])

    trainer = Trainer(model=m,
                      dataset_path=config["paths"]["dataset_path"],
                      **config["train"],
                      config=config,
                      )
    trainer.train()


if __name__ == "__main__":
    train_ct()
    train_aae()
