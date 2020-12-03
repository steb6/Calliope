from types import SimpleNamespace
import os
max_track_length = 1000
vocab_size = 516
config = {
    "data": {  # Parameters to create and listen the note representation
        "max_track_length": max_track_length,  # because of the histogram of the lengths and the memory limits
        "use_velocity": True,
        "reconstruct_programs": [0, 0, 32, 40],
        "early_stop": 1000,
        "resolution": 24,
        "tempo": 120,
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
        "num_values": 128,
        "time_first_token": 4,  # 4-131
        "pitch_first_token": 4 + 128,  # 132-259
        "duration_first_token": 4 + 128 + 128,  # 260-387
        "velocity_first_token": 4 + 128 + 128 + 128,  # 388-515
    },
    "model": {
        "vocab_size": vocab_size,  # this depends by config.tokens
        "d_model": 64,
        "n_tracks": 4,
        "heads": 4,
        "d_ff": 256,
        "layers": 1,
        "dropout": 0.0,
        "mem_len": max_track_length,  # 512, before was 512
        "cmem_len": 100,
        "cmem_ratio": 10,
        "seq_len": max_track_length
    },
    "paths": {
        "raw_midi_path": "D:",
        "dataset_path": "D:" + os.sep + "dataset",
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
