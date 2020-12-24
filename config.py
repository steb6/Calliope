import os
import numpy as np

remote = os.getcwd() != 'C:\\Users\\berti\\PycharmProjects\\MusAE'

config = {
    "train": {
        "create_dataset": True,
        "device": "cuda" if remote else "cpu",
        "batch_size": 1,
        "test_size": 0.1,
        "n_workers": 0,
        "n_epochs": 250,
        "label_smoothing": 0.1,
        "mb_before_eval": 10,
        "warmup_steps": 1000,
        "lr_min": 1e-6,
        "lr_max": 1e-4,
        "generated_iterations": 10,
    },
    "model": {
        "seq_len": 500,
        "d_model": 32,
        "n_tracks": 4,
        "heads": 4,
        "d_ff": 128,
        "layers": 4 if remote else 1,  # 3 GB each
        "dropout": 0.1,
        "mem_len": 1024,  # keep last 2 bars
        "cmem_len": 32,  # keep one vector for each bar
        "cmem_ratio": 32,
        "z_dim": 512
    },
    "data": {  # Parameters to create and listen the note representation
        # "max_bars": 50,
        "max_track_length": 5000,
        "use_velocity": True,
        "reconstruct_programs": [0, 0, 32, 40],
        "early_stop": 1000,  # set this to 0 to disable early stop
        "resolution": 24,
        "tempo": 120,
        "velocities_total": (0, 127),  # using min max scaling, limits are inclusive
        "velocities_compat": (0, 31),  # using min max scaling, limits are inclusive
        "tempos_total": (60, 180),  # using min max scaling, limits are inclusive
        "tempos_compat": (0, 31)  # using min max scaling, limits are inclusive
    },
    "tokens": {
        "pad_token": 0,
        "bar_token": 1,
        "sos_token": 2,
        "eos_token": 3,
        "sob_token": 4,
        "eob_token": 5,
        # Time signature
        "two_two_token": 6,
        # x/4
        "one_four_token": 7,
        "two_four_token": 8,
        "three_four_token": 9,
        "four_four_token": 10,
        "five_four_token": 11,
        "six_four_token": 12,
        "seven_four_token": 13,
        "eight_four_token": 14,
        # x/8
        "three_eight_token": 15,
        "five_eight_token": 16,
        "six_eight_token": 17,
        "seven_eight_token": 18,
        "nine_eight_token": 19,
        "twelve_eight_token": 20,
        # Values
        "num_values": 128,
        "tempos_first_token": 21,
        "time_first_token":     21 + 32,  # it has 128*2 values, because of 8/4 time signature
        "pitch_first_token":    21 + 32 + 128*2,  # 132-259
        "duration_first_token": 21 + 32 + 128*2 + 128,  # 260-387
        "velocity_first_token": 21 + 32 + 128*2 + 128 + 128,  # this has 32 values
        "vocab_size": 21 + 128*4 + 32*2
    },
    "paths": {
        "raw_midi_path": "/data" if remote else "D:",
        "dataset_path": ("/data" if remote else "D:") + os.sep + "lmd_matched_converted",
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
