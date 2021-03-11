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
from config import max_bar_length


class NoteRepresentationManager:
    """
    This class has all the function needed to process a Note Representation and convert it
    """

    def __init__(self):
        self.log = None
        self.log_file = "dataset_converter_log.txt"
        self.count = 0
        self.bar_lengths = []
        self.song_lengths = []
        self.pad = config["tokens"]["pad"]
        self.resolution = config["data"]["resolution"]

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
        s.adjust_resolution(self.resolution)  # computationally heavy
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
                          resolution=self.resolution,  # default resolution
                          )
        new.tracks = [drum, guitar, bass, strings]
        return new

    def reconstruct_music(self, events):  # TODO make it possible to reconstruct faulty song
        """
        :param events: array events representation (4, bars, tokens)
        :return: Muspy song
        """
        assert events.shape[0] == 4
        events = np.reshape(events, (4, -1))
        music = muspy.Music(resolution=self.resolution, tempos=[muspy.Tempo(qpm=120., time=0)],
                            time_signatures=[muspy.TimeSignature(time=0, numerator=4, denominator=4)])
        for i, instrument in enumerate(events):
            instrument = instrument[instrument != self.pad]
            # clean track
            active = []
            to_remove = []
            for n, event in enumerate(instrument):
                # append note on to active
                if event < 128:
                    # avoid to activate a note if it is already active
                    if event in [elem[1] for elem in active]:
                        to_remove.append(n)
                    else:
                        active.append((n, event))
                elif event < 256:
                    pitch = event-128
                    # remove note on from active
                    if pitch in [elem[1] for elem in active]:
                        index_to_remove = [elem[1] for elem in active].index(pitch)
                        del active[index_to_remove]
                    else:
                        to_remove.append(n)
            to_remove = to_remove + [elem[0] for elem in active]
            for elem in to_remove:
                instrument[elem] = self.pad
            instrument = instrument[instrument != self.pad]
            # end clean track
            try:
                instrument = muspy.from_event_representation(instrument)[0]
            except Exception as e:
                print(e.__str__())
                print("BAU")
                instrument = muspy.from_event_representation(instrument)[0]
            instrument.program = config["data"]["reconstruction_programs"][i]
            instrument.is_drum = (i == 0)
            music.append(instrument)
        return music

    def divide_into_bars(self, song):
        """
        :param song: Muspy song
        :return: list of list of Muspy track, (4, bars, track)
        """
        full_time = self.resolution * 4
        divided = []
        for i, instrument in enumerate(song):
            bars = []
            track = muspy.Track()
            for j, note in enumerate(instrument):  # TODO take only lowest note
                # time to add the bar
                while note.time >= full_time:  # 96 should go to 0
                    track.append(muspy.Note(time=full_time, pitch=60, duration=0))  # time padding note
                    track = sorted(track, key=lambda x: (x.time, x.pitch))
                    # take only lowest note for each time
                    clean_track = muspy.Track()
                    t = -1
                    for n in track:
                        if n.time != t:
                            clean_track.append(n)
                        t = n.time
                    # end
                    bars.append(clean_track)
                    track = muspy.Track()
                    # decrease all other times
                    for elem in instrument:
                        elem.time -= full_time
                track.append(copy.deepcopy(note))
            divided.append(bars)
        # all instrument must have the same number of bars
        max_bars = max([len(x) for x in divided])  # TODO check
        for instrument in divided:
            while len(instrument) < max_bars:
                empty_bar = muspy.Track()
                empty_bar.append(muspy.Note(time=full_time, pitch=60, duration=0))  # time padding note
                instrument.append(empty_bar)
        self.song_lengths.append(max_bars)  # log number of bars
        return divided

    def from_song_to_events(self, song):
        """
        :param song: list of list of sequence, (4, bars, tokens)
        :return: events array with shape (4, bars, tokens)
        """
        n_bars = len(song[0])

        song_events = np.full((4, n_bars, max_bar_length), self.pad)
        for i, instrument in enumerate(song):
            for j, bar in enumerate(instrument):
                music = muspy.Music(resolution=self.resolution, tempos=[muspy.Tempo(qpm=120., time=0)],
                                    time_signatures=[muspy.TimeSignature(time=0, numerator=4, denominator=4)])
                music.append(bar)
                bar_events = muspy.to_event_representation(music)
                bar_events = np.reshape(bar_events, (-1))[:-2]
                self.bar_lengths.append(len(bar_events))  # log length of bar
                if len(bar_events) > max_bar_length:
                    bar_events = bar_events[:max_bar_length]
                song_events[i, j, :len(bar_events)] = bar_events
        return song_events

    def convert_dataset(self):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        # download raw dataset if needed
        print("LETS CREATE DATASET")
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
                tensor_song = self.divide_into_bars(filtered_song)
                tensor_song = self.from_song_to_events(tensor_song)

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

                # invert bars and instrument to skip some bar
                tensor_song = np.swapaxes(tensor_song, 0, 1)
                # skip empty bars at the beginning or with just drums (stick for tempos)
                # skip drums and first token (tempo pad)
                while (tensor_song[0, 1:, 1:] == self.pad).all() and len(tensor_song) > 0:
                    tensor_song = tensor_song[1:, ...]
                # divide song into sequence of bars and save them
                while True:
                    no_empty_bars = True
                    candidate = tensor_song[:config["data"]["truncated_bars"], ...]
                    if len(candidate) < config["data"]["truncated_bars"]:
                        break
                    for bar in candidate:
                        if (bar == self.pad).all():
                            no_empty_bars = False
                    if no_empty_bars:
                        with open(os.path.join(config["paths"]["dataset"], str(self.count) + '.pickle'), 'wb') as f:
                            candidate = np.swapaxes(candidate, 0, 1)  # invert again bars and instruments
                            reconstructed = self.reconstruct_music(candidate)  # TODO REMOVE TEST
                            reconstructed.write_midi("reconstructed.mid")  # TODO REMOVE TEST
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
                        tensor_song = tensor_song[config["data"]["truncated_bars"]:, ...]
                    else:
                        break
        print("Song converted, plotting histograms...")
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
    # answer = input(config["paths"]["dataset"] + " will be removed and dataset will be created from zero, "
    #                                             "do you want to proceed?").lower()
    # if answer not in ["y", "yes"]:
    #     exit()
    print("Removing " + config["paths"]["dataset"] + "...")
    shutil.rmtree(config["paths"]["dataset"], ignore_errors=True)
    notes = NoteRepresentationManager()
    notes.convert_dataset()
