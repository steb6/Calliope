import os
import torch
from test import Tester
from create_bar_dataset import NoteRepresentationManager
from tqdm import tqdm


# TODO remember to use 1 bar model for this comparison!


if __name__ == "__main__":
    # load models
    print("Loading models")
    import wandb

    wandb.init()
    wandb.unwatch()
    # BEST MODEL 1 BAR
    checkpoint_name = "/data/musae3.0/musae_model_checkpoints_1/2021-04-20_00-50-29/280000"

    tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                    torch.load(checkpoint_name + os.sep + "latent_decompressor.pt"),
                    torch.load(checkpoint_name + os.sep + "decoder.pt"),
                    torch.load(checkpoint_name + os.sep + "generator.pt"))

    nm = NoteRepresentationManager()

    print("Generating")
    i = 0

    empty_drums_bar = 0
    empty_guitar_bar = 0
    empty_bass_bar = 0
    empty_strings_bar = 0

    # upc_drums = 0
    upc_guitar = 0
    upc_bass = 0
    upc_strings = 0

    total_notes_guitar = 0
    total_notes_bass = 0
    total_notes_strings = 0
    total_notes_drums = 0

    qn_guitar = 0
    qn_bass = 0
    qn_strings = 0

    eight_sixteen_notes = 0

    with torch.no_grad():
        for _ in tqdm(list(range(20000))):
            gen = tester.generate(nm)
            # EB
            if len(gen[0]) == 0:
                empty_drums_bar += 1
            if len(gen[1]) == 0:
                empty_guitar_bar += 1
            if len(gen[2]) == 0:
                empty_bass_bar += 1
            if len(gen[3]) == 0:
                empty_strings_bar += 1

            for note in gen[0]:
                if note.time % 6 == 0:
                    eight_sixteen_notes += 1
            total_notes_drums += len(gen[0])

            guitar_pitches = set()
            for note in gen[1]:
                guitar_pitches.add(note.pitch)
                if note.duration > 1:
                    qn_guitar += 1
            upc_guitar += len(guitar_pitches)
            total_notes_guitar += len(gen[1])

            bass_pitches = set()
            for note in gen[2]:
                bass_pitches.add(note.pitch)
                if note.duration > 1:
                    qn_bass += 1
            upc_bass += len(bass_pitches)
            total_notes_bass += len(gen[2])

            strings_pitches = set()
            for note in gen[3]:
                strings_pitches.add(note.pitch)
                if note.duration > 1:
                    qn_strings += 1
            upc_strings += len(strings_pitches)
            total_notes_strings += len(gen[3])

            i += 1

            if i % 1000 == 0:
                print("EB drums", (empty_drums_bar / i) * 100)
                print("EB guitar", (empty_guitar_bar / i) * 100)
                print("EB bass", (empty_bass_bar / i) * 100)
                print("EB strings", (empty_strings_bar / i) * 100)

                print("UPC guitar", upc_guitar / i)
                print("UPC bass", upc_bass / i)
                print("UPC strings", upc_strings / i)

                print("QN guitar", (qn_guitar / total_notes_guitar) * 100)
                print("QN bass", (qn_bass / total_notes_bass) * 100)
                print("QN strings", (qn_strings / total_notes_strings) * 100)

                print("DP", (eight_sixteen_notes / total_notes_drums) * 100)

    print("EB drums", (empty_drums_bar/i)*100)
    print("EB guitar", (empty_guitar_bar/i)*100)
    print("EB bass", (empty_bass_bar/i)*100)
    print("EB strings", (empty_strings_bar/i)*100)

    print("UPC guitar", upc_guitar/i)
    print("UPC bass", upc_bass/i)
    print("UPC strings", upc_strings/i)

    print("QN guitar", (qn_guitar/total_notes_guitar)*100)
    print("QN bass", (qn_bass/total_notes_bass)*100)
    print("QN strings", (qn_strings/total_notes_strings)*100)

    print("DP", (eight_sixteen_notes/total_notes_drums)*100)



