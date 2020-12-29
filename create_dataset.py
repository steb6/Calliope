import muspy
import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
from manage_time_signature import TimeSignatureManager
import copy
from operator import add
from config import config
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
        self.time_signature_manager = TimeSignatureManager()  # it uses only signature
        self.count = 0

    def filter_song(self, s):
        """
        Get a muspy.Music object
        return the song with the 4 instruments with more notes, preserving tempos and time signatures
        it returns None if the song does not have enough instruments
        """
        for time in s.time_signatures:  # check time signature
            if not self.time_signature_manager.is_valid_time_signature(time.numerator, time.denominator):
                self.log.write("Song with weird time skipped: {} / {}\n".format(time.numerator, time.denominator))
                return None
        for time in s.tempos:  # clip tempo
            if time.qpm < 60:
                time.qpm = 60
            if time.qpm > 180:
                time.qpm = 180
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
        new = muspy.Music(tempos=s.tempos if len(s.tempos) > 0 else [muspy.Tempo(time=0, qpm=120)],  # default tempo
                          time_signatures=s.time_signatures if len(s.time_signatures) > 0 else
                          [muspy.TimeSignature(time=0, numerator=4, denominator=4)],  # default time signature
                          resolution=config["data"]["resolution"],  # default resolution
                          )
        new.tracks = [drum, guitar, bass, strings]
        return new

    def add_tokens(self, t, i, tokens):
        if i + len(tokens) < config["data"]["max_track_length"]:
            for tk in tokens:
                t[i] = tk
                i += 1
            return t, i, True
        self.log.write(str(self.count) + ": Reached max length of track\n")
        return t, i, False

    @staticmethod
    def min_max_scaling(value, old_interval, new_interval):
        """
        It scales a value with range [mn, mx] into a int value with range [a, b]
        """
        mn, mx = old_interval
        a, b = new_interval
        return round((((value - mn)*(b - a)) / (mx - mn)) + a)

    def transform_track(self, s):
        track = np.full(config["data"]["max_track_length"], config["tokens"]["pad"], dtype=np.int16)
        time_signature = 4  # default time signature, if none starts from 0
        signatures = list(s.time_signatures)
        tempos = list(s.tempos)
        if len(signatures) == 0:
            signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        if len(tempos) == 0:
            tempos.append(muspy.Tempo(time=0, qpm=120.))
        try:
            notes = s.to_note_representation()
        except Exception as e:
            self.log.write(str(self.count)+': '+e.__str__()+'\n')
            return None
        idx = 0
        for n, note in enumerate(notes):  # for all note in the track
            for signature in signatures:  # If there is an activated signature
                if signature.time == 0:
                    num = signature.numerator
                    den = signature.denominator
                    res = self.time_signature_manager.from_fraction_to_time_and_token(num, den)
                    if res is None:
                        self.log.write(
                            str(self.count) +
                            ": Time signature not known: {} / {}".format(signature.numerator, signature.denominator))
                        return None
                    time_signature, tok = res
                    signatures.remove(signature)
                    track, idx, success = self.add_tokens(track, idx, (tok,))
                    if not success:
                        return track
            for tempo in tempos:  # if time is activated, min-max scaling and print token
                if tempo.time == 0:
                    tok = self.min_max_scaling(tempo.qpm, config["data"]["tempos_total"],
                                               config["data"]["tempos_compact"])
                    tempos.remove(tempo)
                    tok = tok + config["tokens"]["tempos_first"]
                    track, idx, success = self.add_tokens(track, idx, (tok,))
                    if not success:
                        return track
            while note[0] >= config["data"]["resolution"] * time_signature:  # add bar token
                track, idx, success = self.add_tokens(track, idx, (config["tokens"]["bar"],))
                if not success:
                    return track
                notes[n:, 0] -= round(config["data"]["resolution"] * time_signature)  # decrease all time measures
                for signature in signatures:
                    signature.time -= config["data"]["resolution"] * time_signature
                for tempo in tempos:
                    tempo.time -= config["data"]["resolution"] * time_signature
            if not note[0] < config["tokens"]["time_n_values"] * 2:  # check value of time
                self.log.write(str(self.count) + ": Invalid time: " + str(note) + '\n')
                continue  # skip note: invalid time
            if not note[1] < config["tokens"]["pitch_n_values"]:  # check value of pitch
                self.log.write(str(self.count) + ": Invalid pitch: " + str(note[1]))
                continue  # skip note: invalid pitch
            if not note[2] < config["tokens"]["duration_n_values"]:  # clip duration if > 127
                note[2] = config["tokens"]["duration_n_values"] - 1
            if config["data"]["use_velocity"]:  # if velocity, use min-max normalization with new interval
                note[3] = self.min_max_scaling(note[3], config["data"]["velocities_total"],  # it was clipped, so
                                               config["data"]["velocities_compact"])  # no need to check values
            track, idx, success = self.add_tokens(track, idx, list(map(add, self.offsets, note)))
            if not success:
                return track
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
        x1 = config["tokens"]["time_first"]
        x2 = config["tokens"]["pitch_first"]
        x3 = config["tokens"]["duration_first"]
        x4 = config["tokens"]["velocity_first"]
        return x1 <= t < x2 <= p < x3 <= d < x4 <= v

    def reconstruct_music(self, s):
        """
        It takes a tensor song and return a muspy.Music element, music can be even in bar format,
        it use time_offset to move the beginning of the notes after n bars
        """
        music = muspy.Music(resolution=config["data"]["resolution"])
        time_signature = 4
        for i, instrument in enumerate(s):  # for each encoder track
            time = 0
            track = muspy.Track(is_drum=i == 0, program=config["data"]["reconstruction_programs"][i])
            notes = instrument.flat
            while len(notes) > 4 and notes[0] != config["tokens"]["pad"]:
                # If it is a note
                if self.it_is_a_note(notes[0], notes[1], notes[2], notes[3]):
                    note = muspy.Note(notes[0] + time - config["tokens"]["time_first"],
                                      notes[1] - config["tokens"]["pitch_first"],
                                      notes[2] - config["tokens"]["duration_first"],
                                      self.min_max_scaling(notes[3] - config["tokens"]["velocity_first"],
                                                           config["data"]["velocities_compact"],
                                                           config["data"]["velocities_total"]))
                    track.append(note)
                    notes = notes[4:]
                elif notes[0] == config["tokens"]["bar"]:  # if it is a bar
                    time += round(config["data"]["resolution"] * time_signature)  # round just to return int
                    notes = notes[1:]
                    # track.notes.append(muspy.Note(time=time, pitch=60, duration=12, velocity=127))  # bar sound
                # check if it is a time signature
                # TODO avoid to add signature or tempos 4 times
                elif self.time_signature_manager.from_token_to_time_and_fraction(notes[0]) is not None:
                    time_signature, n, d = self.time_signature_manager.from_token_to_time_and_fraction(notes[0])
                    music.time_signatures.append(muspy.TimeSignature(time=time, numerator=n, denominator=d))
                    notes = notes[1:]
                elif config["tokens"]["tempos_first"] <= notes[0] <= config["tokens"]["tempos_first"] \
                        + config["data"]["tempos_compact"][1]:  # tempos_compact is 31, so we have <=
                    tempo = self.min_max_scaling(notes[0] - config["tokens"]["tempos_first"],
                                                 config["data"]["tempos_compact"],
                                                 config["data"]["tempos_total"])
                    music.tempos.append(muspy.Tempo(time=time, qpm=tempo))
                    notes = notes[1:]
                else:  # unknown combination, just skip (maybe undertrained model)
                    notes = notes[1:]
            music.append(track)
        if len(music.tempos) == 0:  # default tempo if none met
            music.tempos.append(muspy.Tempo(time=0, qpm=120))
        if len(music.time_signatures) == 0:  # default time signature if none met
            music.time_signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        return music

    @staticmethod
    def cut_song(song, end_time):
        cut_song = muspy.Music()
        for track in song.tracks:
            cut_track = muspy.Track(program=track.program, is_drum=track.is_drum)
            for note in track.notes:
                if note.time < end_time:
                    cut_track.notes.append(note)
            cut_song.tracks.append(cut_track)
        for tempo in song.tempos:
            if tempo.time < end_time:
                cut_song.tempos.append(tempo)
        for signature in song.time_signatures:
            if signature.time < end_time:
                cut_song.time_signatures.append(signature)
        return cut_song

    def convert_dataset(self):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        if not os.path.exists(os.path.join(config["paths"]["raw_midi"], "lmd_matched")):
            print("Downloading Lakh Dataset into "+config["paths"]["raw_midi"])
            muspy.LakhMIDIMatchedDataset(config["paths"]["raw_midi"], download_and_extract=True)
        print("Converting Lakh Dataset from " + config["paths"]["raw_midi"] + " in " + config["paths"]["dataset"])
        raw_midi = config["paths"]["raw_midi"] + os.sep + "lmd_matched"
        if config["data"]["early_stop"] == 0:
            dataset_length = sum([len(files) for _, _, files in os.walk(raw_midi)])
        else:
            dataset_length = config["data"]["early_stop"]
        progbar = tqdm(total=dataset_length, leave=True, position=0, desc="Dataset creation")
        self.count = 0
        os.makedirs(config["paths"]["dataset"])
        self.log = open(self.log_file, "w")
        self.log.write("Log of dataset_converter, to check if it is working right\n")
        # lengths = []
        for subdir, dirs, files in os.walk(raw_midi):  # iterate over all subdirectories
            for filename in files:  # iterate over all files
                if config["data"]["early_stop"] == 0 and self.count > 0:  # if not early stop, update bar anyway
                    progbar.update()
                filepath = subdir + os.sep + filename
                try:
                    song = muspy.read(filepath)
                except Exception as e:  # invalid song format
                    self.log.write(str(self.count) + " invalid song format: " + e.__str__() + '\n')
                    continue
                filtered_song = self.filter_song(song)
                if filtered_song is None:  # if the song has 4 valid tracks
                    continue
                tensor_song = self.transform_song(filtered_song)
                if tensor_song is None:
                    continue
                # TODO to save length
                # for elem in tensor_song:
                # lengths.append(len(elem))
                # TODO test
                # try:
                #     song.write_midi("before.mid")
                # except Exception as e:
                #     self.log.write(e.__str__()+'\n')
                # try:
                #     filtered_song.write_midi("middle.mid")
                # except Exception as e:
                #     self.log.write(e.__str__()+'\n')
                # try:
                #     reconstructed_music = self.reconstruct_music(tensor_song)
                #     reconstructed_music.write_midi("after.mid")
                # except Exception as e:
                #     self.log.write(e.__str__()+'\n')
                # TODO end test
                with open(os.path.join(config["paths"]["dataset"], str(self.count) + '.pickle'), 'wb') as f:
                    pickle.dump(tensor_song, f)
                self.count += 1
                if config["data"]["early_stop"] != 0:  # if early_stop, we update progbar only after a success
                    progbar.update()
                    if self.count >= config["data"]["early_stop"]:
                        self.log.close()
                        # import matplotlib.pyplot as plt
                        # plt.hist(lengths, density=True, bins=30)  # `density=False` would make counts
                        # plt.ylabel('Probability')
                        # plt.xlabel('Data')
                        # plt.show()
                        return
        self.log.close()
