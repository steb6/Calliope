import muspy
import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
from config import config
import shutil
import time
from utilities import min_max_scaling


# 100'000 songs take 400 GB


class NoteRepresentationManager:
    """
    This class has all the function needed to process a Note Representation and convert it
    """

    def __init__(self):
        self.log = None
        self.log_file = "dataset_converter_log.txt"
        if config["data"]["use_velocity"]:
            self.offsets = [config["tokens"]["time_first"], config["tokens"]["pitch_first"],
                            config["tokens"]["duration_first"], config["tokens"]["velocity_first"]]
        else:
            self.offsets = [config["tokens"]["time_first"], config["tokens"]["pitch_first"],
                            config["tokens"]["duration_first"]]
        self.count = 0
        self.bar_lengths = []
        self.song_lengths = []

    def filter_song(self, s):
        """
        :param s: Muspy song
        :return: filtered song or None if s is invalid
        """
        for t in s.time_signatures:  # check time signature
            if t.numerator != 4 or t.denominator != 4:
                self.log.write(str(self.count) + ": Song with weird time skipped: {} / {}\n".
                               format(t.numerator, t.denominator))
                return None
        old_stdout = sys.stdout  # backup current stdout because adjust_resolution is too verbose
        sys.stdout = open(os.devnull, "w")
        s.adjust_resolution(config["data"]["resolution"])  # computationally heavy
        s.clip()  # clip velocities into 0-127
        sys.stdout = old_stdout
        drum = None
        guitar = None
        bass = None
        strings = None
        for track in s.tracks:
            if track.is_drum:  # is a drum
                if drum is None or len(track.notes) > len(drum.notes):  # and is better than the others
                    drum = track
            elif 0 <= track.program <= 31:
                if guitar is None or len(track.notes) > len(guitar.notes):
                    guitar = track
            elif 32 <= track.program <= 39:
                if bass is None or len(track.notes) > len(bass.notes):
                    bass = track
            elif strings is None or len(track.notes) > len(strings.notes):
                strings = track
        if drum is None or guitar is None or bass is None or strings is None:
            return None
        new = muspy.Music(tempos=[muspy.Tempo(time=0, qpm=120)],  # default tempo
                          time_signatures=[muspy.TimeSignature(time=0, numerator=4, denominator=4)],
                          resolution=config["data"]["resolution"],  # default resolution
                          )
        new.tracks = [drum, guitar, bass, strings]
        return new

    def transform_track(self, s):
        """
        :param s: Muspy song
        :return: tensor version of the song or None if invalid
        """
        track = np.full((config["data"]["max_bars"], config["data"]["max_bar_length"]), config["tokens"]["pad"],
                        dtype=np.int16)
        time_signature = 4  # default time signature, if none starts from 0
        bar_steps = config["data"]["resolution"] * time_signature
        use_velocity = config["data"]["use_velocity"]
        try:
            notes = s.to_note_representation()
        except Exception as e:
            self.log.write(str(self.count) + ': ' + e.__str__() + '\n')
            return None
        i = 0  # bar index
        j = 0  # token index
        n = 0  # notes index
        while n < len(notes):
            # add empty bar till note time value is in range
            while notes[n][0] >= bar_steps:
                if i < config["data"]["max_bars"] - 1:
                    i += 1
                    self.bar_lengths.append(j)  # empty bar
                    j = 0
                    notes[n:, 0] -= round(bar_steps)  # decrease all time measures
                else:
                    # log bars length and return song
                    bars_left = sorted(notes[n:], key=lambda x: x[0])[-1][0] % bar_steps
                    self.log.write(str(self.count) + ": Maximum number of bars reached, total was "+str(i+bars_left))
                    self.song_lengths.append(i + bars_left)
                    return track  # truncate song: no more space for bars
            # check note values
            if not notes[n][0] < config["tokens"]["time_n_values"]:  # check value of time
                self.log.write(str(self.count) + ": Invalid time: " + str(notes[n][0]) + '\n')
                n += 1
                continue  # skip note: invalid time
            if not notes[n][1] < config["tokens"]["pitch_n_values"]:  # check value of pitch
                self.log.write(str(self.count) + ": Invalid pitch: " + str(notes[n][1]))
                n += 1
                continue  # skip note: invalid pitch
            if not notes[n][2] < config["tokens"]["duration_n_values"]:  # clip duration if > 127
                notes[n][2] = config["tokens"]["duration_n_values"] - 1
            if use_velocity:  # if velocity, use min-max normalization with new interval
                notes[n][3] = min_max_scaling(notes[n][3], config["data"]["velocities_total"],
                                              config["data"]["velocities_compact"])  # no need to check values
            # add note
            if j + (4 if use_velocity else 3) < config["data"]["max_bar_length"] - 1:  # if enough space for note
                track[i][j] = notes[n][0] + self.offsets[0]
                track[i][j + 1] = notes[n][1] + self.offsets[1]
                track[i][j + 2] = notes[n][2] + self.offsets[2]
                if use_velocity:
                    track[i][j + 3] = notes[n][3] + self.offsets[3]
                j += (4 if use_velocity else 3)
                previous_time = notes[n][0]
                n += 1
            else:
                # no more space inside the bar, count tokens left and skip bar notes
                notes_left = (notes[n:, 0] < bar_steps).sum()
                tok_left = notes_left * (4 if use_velocity else 3)
                self.log.write(str(self.count) + ": Reached max bar length, left tokens: " + str(j + tok_left) + "\n")
                self.bar_lengths.append(j + tok_left)
                n += notes_left
        self.song_lengths.append(i)
        return track

    def transform_song(self, s):
        """
        It takes a filtered song and return the final tensor version of it, making a deep copy of it
        """
        processed = []
        for track in s.tracks:
            just_track = muspy.Music(resolution=config["data"]["resolution"], tempos=copy.deepcopy(s.tempos),
                                     time_signatures=copy.deepcopy(s.time_signatures))
            just_track.append(copy.deepcopy(track))
            adjusted_track = self.transform_track(just_track)
            if adjusted_track is None:  # unknown time signature
                return None
            processed.append(adjusted_track)
        return np.array(processed).astype(np.int16)

    @staticmethod
    def it_is_a_note(t, p, d, v):
        """
        :param t: note time
        :param p: note pitch
        :param d: note duration
        :param v: note velocities
        :return: True if it is a valid note, False otherwise
        """
        x1 = config["tokens"]["time_first"]
        x2 = config["tokens"]["pitch_first"]
        x3 = config["tokens"]["duration_first"]
        x4 = config["tokens"]["velocity_first"]
        x5 = config["tokens"]["vocab_size"]
        if config["data"]["use_velocity"]:
            return x1 <= t < x2 <= p < x3 <= d < x4 <= v < x5
        else:
            return x1 <= t < x2 <= p < x3 <= d < x4

    def reconstruct_music(self, s):
        """
        :param s: Tensor song to reconstruct
        :return: Muspy song
        """
        use_velocity = config["data"]["use_velocity"]
        music = muspy.Music(resolution=config["data"]["resolution"], tempos=[muspy.Tempo(qpm=120., time=0)])
        for i, instrument in enumerate(s):  # for each encoder track
            time_cursor = 0
            track = muspy.Track(is_drum=(i == 0), program=config["data"]["reconstruction_programs"][i])
            for bar in instrument:
                # track.notes.append(muspy.Note(time=time, pitch=60, duration=12, velocity=127))  # bar sound
                while len(bar) > (4 if use_velocity else 3):
                    if self.it_is_a_note(bar[0], bar[1], bar[2], (bar[3] if use_velocity else 0)):
                        velocity = 100  # default one
                        if use_velocity:  # use the encoded one
                            velocity = min_max_scaling(bar[3] - config["tokens"]["velocity_first"],
                                                       config["data"]["velocities_compact"],
                                                       config["data"]["velocities_total"])
                        note = muspy.Note(bar[0] + time_cursor - config["tokens"]["time_first"],
                                          bar[1] - config["tokens"]["pitch_first"],
                                          bar[2] - config["tokens"]["duration_first"],
                                          velocity
                                          )
                        track.append(note)
                        bar = bar[(4 if use_velocity else 3):]
                    else:
                        bar = bar[1:]
                time_cursor += round(config["data"]["resolution"] * 4)
            music.append(track)
        music.tempos.append(muspy.Tempo(time=0, qpm=120))
        music.time_signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        return music

    def convert_dataset(self):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        # download raw dataset if needed
        if not os.path.exists(os.path.join(config["paths"]["raw_midi"], "lmd_matched")):
            print("Downloading Lakh Dataset into " + config["paths"]["raw_midi"])
            muspy.LakhMIDIMatchedDataset(config["paths"]["raw_midi"], download_and_extract=True)
        # counting songs if needed for progbar
        print("Converting Lakh Dataset from " + config["paths"]["raw_midi"] + " in " + config["paths"]["dataset"])
        raw_midi = config["paths"]["raw_midi"] + os.sep + "lmd_matched"
        if config["data"]["early_stop"] == 0:
            print("No early_stop, counting all files...")
            dataset_length = sum([len(files) for _, _, files in os.walk(raw_midi)])
            print("Found", dataset_length, "raw song audio.")
        else:
            dataset_length = config["data"]["early_stop"]
        time.sleep(1.)  # sleep one second for a correct output presentation
        progbar = tqdm(total=dataset_length, leave=True, position=0, desc="Dataset creation")
        self.count = 0
        os.makedirs(config["paths"]["dataset"])
        # setting up log file
        self.log = open(self.log_file, "w")
        self.log.write("Log of dataset_converter, to check if it is working right\n")
        # main loop
        for subdir, dirs, files in os.walk(raw_midi):  # iterate over all subdirectories
            for filename in files:  # iterate over all files
                if config["data"]["early_stop"] == 0 and self.count > 0:  # if not early stop, update bar anyway
                    progbar.update()
                filepath = subdir + os.sep + filename
                try:
                    song = muspy.read(filepath)
                except Exception as e:  # invalid song format
                    self.log.write(str(self.count) + ": Invalid song format: " + e.__str__() + '\n')
                    continue
                filtered_song = self.filter_song(song)
                if filtered_song is None:  # if the song has 4 valid tracks
                    continue
                tensor_song = self.transform_song(filtered_song)
                if tensor_song is None:
                    continue

                # TODO test
                # try:
                #     song.write_midi("before.mid")
                # except Exception as e:
                #     self.log.write(e.__str__() + '\n')
                #     print(e.__str__() + '\n')
                # try:
                #     filtered_song.write_midi("middle.mid")
                # except Exception as e:
                #     self.log.write(e.__str__() + '\n')
                #     print(e.__str__() + '\n')
                # try:
                #     reconstructed_music = self.reconstruct_music(tensor_song)
                #     reconstructed_music.write_midi("after.mid")
                # except Exception as e:
                #     self.log.write(e.__str__() + '\n')
                #     print(e.__str__() + '\n')
                # TODO end test

                # save only sequence of bars if no empty bar
                tensor_song = np.swapaxes(tensor_song, 0, 1)
                # skip empty bars at the beginning or with just drums (stick for tempos)
                while (tensor_song[0, 1:, :] == 0).all():
                    tensor_song = tensor_song[1:, ...]
                # divide song into sequence of bars and save them
                while True:
                    no_empty_bars = True
                    candidate = tensor_song[:config["data"]["bars"], ...]
                    if len(candidate) < config["data"]["bars"]:
                        break
                    for bar in candidate:
                        if (bar == 0).all():
                            no_empty_bars = False
                    if no_empty_bars:
                        with open(os.path.join(config["paths"]["dataset"], str(self.count) + '.pickle'), 'wb') as f:
                            candidate = np.swapaxes(candidate, 0, 1)
                            self.reconstruct_music(candidate).write_midi("test.mid")  # TODO remove test
                            pickle.dump(candidate, f)
                        self.count += 1
                        # if early stop, update bar only after a success
                        if config["data"]["early_stop"] != 0:
                            progbar.update()
                            if self.count >= config["data"]["early_stop"]:
                                self.log.close()
                                self.plot_lengths()
                                progbar.close()
                                return
                        tensor_song = tensor_song[config["data"]["bars"]:, ...]
                    else:
                        break

        self.plot_lengths()
        self.log.close()
        progbar.close()

    def plot_lengths(self):
        plt.hist(self.bar_lengths, density=True, bins=30)  # `density=False` would make counts
        plt.ylabel('Number of bar with that length')
        plt.xlabel('Bar length')
        plt.savefig("bar_length_distribution.png")
        plt.close()
        plt.hist(self.song_lengths, density=True, bins=30)  # `density=False` would make counts
        plt.ylabel('Number of song with that bars')
        plt.xlabel('Number of bars')
        plt.savefig("song_length_distribution.png")
        plt.close()


if __name__ == "__main__":
    answer = input(config["paths"]["dataset"] + " will be removed and dataset will be created from zero, "
                                                "do you want to proceed?").lower()
    if answer not in ["y", "yes"]:
        exit()
    shutil.rmtree(config["paths"]["dataset"], ignore_errors=True)
    notes = NoteRepresentationManager()
    notes.convert_dataset()
