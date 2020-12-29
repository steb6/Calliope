from config import set_freer_gpu
from create_dataset import NoteRepresentationManager
from config import config
from train import Trainer
# from train_aae import AAETrainer
import shutil
import torch
import os


# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE\*.py berti@131.114.137.168:MusAE
# TO CONNECT: ssh berti@131.114.137.168
# TO ATTACH TO TMUX: tmux attach -t Training
# TO RESIZE TMUX: tmux attach -d -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi
# TO GET RESULTS: scp -r berti@131.114.137.168:MusAE/2020* C:\Users\berti\PycharmProjects\MusAE\remote_results


def train_ct():
    set_freer_gpu()

    if config["train"]["create_dataset"]:
        shutil.rmtree(config["paths"]["dataset"], ignore_errors=True)
        notes = NoteRepresentationManager()
        notes.convert_dataset()

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":  # TODO put here assert I guess
    assert config["model"]["total_seq_len"] % config["model"]["seq_len"] == 0, 'adjust total_seq_len seq_len'
    train_ct()
    # train_aae()
