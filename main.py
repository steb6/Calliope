from config import set_freer_gpu
from create_bar_dataset import NoteRepresentationManager
from config import config
from train import Trainer
import shutil

# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE\*.py berti@131.114.137.168:MusAE
# TO CONNECT: ssh berti@131.114.137.168
# TO GET: scp -r berti@131.114.137.168:/data/musae3.0/musae_model_checkpoints_2/2021-04-16_22-59-09/200000 C:\Users\berti\PycharmProjects\MusAE\2bar
# TO START SERVER: python -m http.server 1378
# TO ATTACH TO TMUX: tmux attach -t Training
# TO RESIZE TMUX: tmux attach -d -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi
# TO GET RESULTS: scp -r berti@131.114.137.168:MusAE/2020* C:\Users\berti\PycharmProjects\MusAE\remote_results


if __name__ == "__main__":  # TODO put here assert I guess
    print("Use create_bar_dataset to create the dataset, then use train to train the model")
    # set_freer_gpu()
    #
    # answer = ""
    # while answer not in ["y", "n"]:
    #     answer = input("Dataset will be created from zero, do you want to proceed?").lower()
    #
    # shutil.rmtree(config["paths"]["dataset"], ignore_errors=True)
    # notes = NoteRepresentationManager()
    # notes.convert_dataset()
    #
    # trainer = Trainer()
    # trainer.train()
