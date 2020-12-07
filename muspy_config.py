import os

remote = True

max_bars = 100
max_bar_length = 100
vocab_size = 21 + 128*4 + 32*2 + 1 #TODO check  # 579 token + tempos + time*2, pitch, duration + velocities

config = {
    "data": {  # Parameters to create and listen the note representation
        "max_bars": max_bars,  # because of the histogram of the lengths and the memory limits
        "max_bar_length": max_bar_length,
        "use_velocity": True,
        "reconstruct_programs": [0, 0, 32, 40],
        "early_stop": 10,
        "resolution": 24,
        "tempo": 120,
        "velocities_interval": (0, 127),
        "velocities_values": 32,
        "tempos_interval": (60, 180),
        "tempos_values": 32
    },
    "train": {
        "vocab_size": vocab_size,
        "device": "cuda",
        "batch_size": 1,
        "test_size": 0.3,
        "n_workers": 1,
        "n_epochs": 250,
        "label_smoothing": 0.1,
    },
    "tokens": {
        "pad_token": 0,
        "bar_token": 1,
        "sos_token": 2,
        "eos_token": 3,
        "sob_token": 4,
        "eob_token" : 5,
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
    },
    "model": {
        "vocab_size": vocab_size,  # this depends by config.tokens
        "d_model": 64,
        "n_tracks": 4,
        "heads": 4,
        "d_ff": 256,
        "layers": 1,
        "dropout": 0.0,
        "mem_len": max_bar_length,  # 512, before was 512
        "cmem_len": 25,
        "cmem_ratio": 4,
        "seq_len": max_bar_length,
        "pad_token": 0
    },
    "paths": {
        "raw_midi_path": "/data" if remote else "D:",
        "dataset_path": ("/data" if remote else "D:") + os.sep + "dataset",
    },
    "names": {
        "dataset_converter_log_file": "dataset_converter_log.txt",
        "plot_name": "learning_curve",
        "model_name": "checkpoint_epoch",
    }
}

# config["data"] = SimpleNamespace(**config["data"])
# config["model"] = SimpleNamespace(**config["model"])
# config["tokens"] = SimpleNamespace(**config["tokens"])
# config["train"] = SimpleNamespace(**config["train"])
# config["paths"] = SimpleNamespace(**config["paths"])
# config["names"] = SimpleNamespace(**config["names"])
# config = SimpleNamespace(**config)

# config = {
#     # Where to download Lakh Midi dataset and where to save extracted songs
#     "raw_midi_path": "D:",
#     "dataset_path": "D:" + os.sep + "dataset",
#     # To create and use dataset
#     "early_stop": 1000,
#     "resolution": 24,
#     "tempo": 120,
#     "max_bar_length": 24*4*4,  # number of time steps x number of quarter note in bar x note representation length
#     "max_track_length": 1000,  # because of the histogram of the lengths and the memory limits
#     "use_velocity": True,
#     "reconstruct_programs": [0, 0, 32, 40],
#     # Generation
#     "trained_model": os.path.join("new_musae", "checkpoint.pt"),
#     "sampled_path": "new_musae",
#     # Device
#     "device": "cuda",
#     # Vocab and seq len
#     "vocab_size": 516,
#     "seq_len": 200,  # resolution x 4 (quarter note) x token for each note x number of max_bars
#     "max_bars": 5,
#     # Train
#     "batch_size": 1,
#     "test_size": 0.3,
#     "n_workers": 1,
#     "n_epochs": 250,
#     "label_smoothing": 0.1,
#     # Model
#     "d_model": 64,
#     "n_tracks": 4,
#     "n_heads": 4,
#     "d_ff": 256,
#     "layers": 1,
#     "dropout": 0.0,
#     "mem_len": 200, # 512, before was 512
#     "cmem_len": 32,
#     "cmem_ratio": 4,
#     # Tokens, change this value is not enough for change the representation
#     "pad_token": 0,
#     "bar_token": 1,
#     "sos_token": 2,
#     "eos_token": 3,
#     "num_values": 128,
#     "time_first_token":     4,  # 4-131
#     "pitch_first_token":    4 + 128,  # 132-259
#     "duration_first_token": 4 + 128 + 128,  # 260-387
#     "velocity_first_token": 4 + 128 + 128 + 128,  # 388-515
#     # File names
#     "dataset_converter_log_file": "dataset_converter_log.txt",
#     "plot_name": "learning_curve",
#     "model_name": "checkpoint_epoch",
# }
# config = SimpleNamespace(**config)
