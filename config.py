import os
import numpy as np

remote = os.getcwd() != 'C:\\Users\\berti\\PycharmProjects\\MusAE'

max_bar_length = 150  # for preprocessing, seq_len, mem_len e cmem_len
max_bars = 100  # for cmem_len and data creation

config = {
    "train": {
        "do_eval": False,
        "aae": False,
        "create_dataset": False if remote else False,
        "device": "cuda" if remote else "cpu",
        "batch_size": 3 if remote else 3,
        "test_size": 0.3 if remote else 0.3,  # 100 on remote  it was 0.0001 in remote
        "truncated_bars": 16,  # To truncate the song along bars
        "n_workers": 0,
        "n_epochs": 25000,
        "label_smoothing": 0.1,
        "mb_before_eval": 1000 if remote else 10,  # if >= early_stopping, happens at each epoch
        "after_mb_log_attn_img": 1000 if remote else 10,
        "after_mb_log_examples": 1000 if remote else 10,
        "warmup_steps": 4000 if remote else 10,
        "lr_min": 1e-4 if remote else 1e-2,
        "lr_max": 1e-3 if remote else 1e-2,
        "decay_steps": 50000 if remote else 1000000,
        "minimum_lr": 1e-4 if remote else 1e-2,
        "generated_iterations": 10,
        "test_loss": False,
    },
    "model": {
        # "total_seq_len": 300 if remote else 600,
        "seq_len": max_bar_length,
        "d_model": 32,
        "heads": 4,
        "ff_mul": 2,
        "layers": 2 if remote else 1,  # if remote else 1,  # 3 GB each
        "dropout": 0.1,
        "mem_len": max_bar_length,  # keep last 2 seq
        "cmem_len": max_bar_length,  # keep 4 compression
        "cmem_ratio": 4,
        "z_i_dim": 512 if remote else 64,
        # max_track_length / seq_len = n_latents, n_latents * z_i_dim are compressed into z_tot_dim
        "z_tot_dim": 2048 if remote else 256,
        "fixed_positional_encoding": False,
        "compress": True
    },
    "data": {  # Parameters to create and listen the note representation
        "max_bar_length": 150,
        "max_bars": 100,
        # "max_track_length": 10000,
        "use_velocity": True,
        "reconstruction_programs": [0, 0, 32, 40],
        "early_stop": 10000 if remote else 10,  # set this to 0 to disable early stop
        "resolution": 24,
        "tempo": 120,
        "velocities_total": (0, 127),  # using min max scaling, limits are inclusive
        "velocities_compact": (0, 31),  # using min max scaling, limits are inclusive
        "tempos_total": (60, 180),  # using min max scaling, limits are inclusive
        "tempos_compact": (0, 31)  # using min max scaling, limits are inclusive
    },
    "tokens": {
        "pad": 0,
        "bar": 1,
        "sos": 2,
        "eos": 3,
        # Time signature
        # x/2
        "two_two": 4,
        # x/4
        "one_four": 5,
        "two_four": 6,
        "three_four": 7,
        "four_four": 8,
        "five_four": 9,
        "six_four": 10,
        "seven_four": 11,
        "eight_four": 12,
        # x/8
        "three_eight": 13,
        "five_eight": 14,
        "six_eight": 15,
        "seven_eight": 16,
        "nine_eight": 17,
        "twelve_eight": 18,
        # Values
        "time_n_values": 256,  # it has 128*2 values, because of 8/4 time signature
        "pitch_n_values": 128,
        "duration_n_values": 128,
        "velocity_n_values": 32,
        "tempos_first":   19,
        "time_first":     19 + 32,
        "pitch_first":    19 + 32 + 128*2,
        "duration_first": 19 + 32 + 128*3,
        "velocity_first": 19 + 32 + 128*4,
        "vocab_size":     19 + 32 + 128*4 + 32
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
