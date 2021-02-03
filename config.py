import os
import numpy as np

# MAXIMUM ON REMOTE MACHINE: 16GB with 5 layers and 23 bars, 3450 on 3750

remote = os.getcwd() != 'C:\\Users\\berti\\PycharmProjects\\MusAE'

max_bar_length = 150  # for preprocessing, seq_len, mem_len e cmem_len

config = {
    "train": {
        "verbose": False,
        "do_eval": False,
        "aae": True,
        "create_dataset": False,
        "device": "cuda" if remote else "cpu",
        "batch_size": 3 if remote else 3,
        "test_size": 0.001 if remote else 0.3,  # 100 on remote  it was 0.0001 in remote
        "truncated_bars": 16 if remote else 6,  # To truncate the song along bars
        "n_workers": 0,
        "n_epochs": 25000,
        "label_smoothing": 0.1,
        "mb_before_eval": 1000 if remote else 1000,  # if >= early_stopping, happens at each epoch
        "after_mb_log_attn_img": 1000 if remote else 1000,
        "after_mb_log_examples": 1000 if remote else 1000,
        "after_mb_log_memories": 1000 if remote else 1000,
        "warmup_steps": 4000,
        "lr_min": 1e-4,
        "lr_max": 1e-3,
        "decay_steps": 50000,
        "minimum_lr": 1e-4,
        "generated_iterations": 16,
        "test_loss": False,
    },
    "model": {
        "seq_len": max_bar_length,
        "d_model": 32,
        "heads": 4,
        "ff_mul": 2,
        "layers": 4 if remote else 2,  # if remote else 1,  # 3 GB each
        "mem_len": max_bar_length,  # keep last 2 seq
        "cmem_len": max_bar_length,  # keep 4 compression
        "cmem_ratio": 4,
        "reconstruction_attn_dropout": 0.1,
        "attn_layer_dropout": 0.1,
        "ff_dropout": 0.1,
        "discriminator_dropout": 0.1
    },
    "data": {  # Parameters to create and listen the note representation
        "max_bar_length": max_bar_length,
        "max_bars": 100,
        "use_velocity": True,
        "reconstruction_programs": [0, 0, 32, 40],
        "early_stop": 0 if remote else 10,  # set this to 0 to disable early stop
        "resolution": 24,
        "tempo": 120,
        "velocities_total": (0, 127),  # using min max scaling, limits are inclusive
        "velocities_compact": (0, 31),  # using min max scaling, limits are inclusive
    },
    "tokens": {
        "pad": 0,
        "bar": 1,
        "sos": 2,
        "eos": 3,
        # Values
        "time_n_values": 128,
        "pitch_n_values": 128,
        "duration_n_values": 128,
        "velocity_n_values": 32,
        "time_first":     4,
        "pitch_first":    4 + 128,
        "duration_first": 4 + 128*2,
        "velocity_first": 4 + 128*3,
        "vocab_size":     4 + 128*3 + 32
    },
    "paths": {
        "raw_midi": "/data" if remote else "D:",
        "dataset": ("/data" if remote else "D:") + os.sep + "lmd_matched_converted",
    }
}


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def set_freer_gpu():
    if remote:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu = str(get_freer_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("Remote execution on gpu ", gpu)
    else:
        print("Local execution")
