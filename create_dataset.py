import muspy
import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
from manage_time_signature import TimeSignatureManager
import copy
from operator import add
# 100'000 songs take 400 GB


class NoteRepresentationManager:
    """
    This class has all the function needed to process a Note Representation and convert it
    """

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.log = None
        self.log_file = "dataset_converter_log.txt"
        if self.use_velocity:
            self.offsets = [self.time_first_token, self.pitch_first_token,
                            self.duration_first_token, self.velocity_first_token]
        else:
            self.offsets = [self.time_first_token, self.pitch_first_token, self.duration_first_token]
        self.time_signature_manager = TimeSignatureManager(**kwargs)  # it uses only signature

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
        old_stdout = sys.stdout  # backup current stdout
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
        new = muspy.Music(tempos=s.tempos if len(s.tempos) > 0 else [muspy.Tempo(time=0, qpm=120)],  # default tempo
                          time_signatures=s.time_signatures if len(s.time_signatures) > 0 else
                          [muspy.TimeSignature(time=0, numerator=4, denominator=4)],  # default time signature
                          resolution=self.resolution,  # default resolution
                          )
        new.tracks = [drum, guitar, bass, strings]
        return new

    def add_tokens(self, t, i, j, tokens):
        """
        Given a matrix track, row index, column index and tuple with tokens to add,
        it add tokens into line if possible, otherwise it add what eob_token, goes in new bar and add sob and tokens
        and if it is not possible to add another line, it adds eos.
        """
        if j + len(tokens) < self.max_bar_length:  # 95 + 4 < 100
            for tk in tokens:
                t[i][j] = tk
                j += 1
        elif i + 1 < self.max_bars:  # if adding a row, we are in limits
            t[i][j] = self.eob_token
            i += 1
            j = 0
            t[i][0] = self.sob_token
            self.log.write("Reached max length of bar\n")
            return self.add_tokens(t, i, j, tokens)  # new line and recursion
        else:
            t[i][j] = self.eos_token
            self.log.write("Reached max length of song\n")
        return t, i, j

    @staticmethod
    def min_max_scaling(value, old_interval, new_interval):
        """
        It scales a value with range [mn, mx] into a int value with range [a, b]
        """
        mn, mx = old_interval
        a, b = new_interval
        return round((((value - mn)*(b - a)) / (mx - mn)) + a)

    def transform_track(self, s):
        """
        It transform the notes notation into a list of numbers which are the bars
        """
        track = np.zeros((self.max_bars, self.max_bar_length), dtype=np.int16)
        time_signature = 4  # default time signature, if none starts from 0
        signatures = list(s.time_signatures)
        tempos = list(s.tempos)
        if len(signatures) == 0:
            signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        if len(tempos) == 0:
            tempos.append(muspy.Tempo(time=0, qpm=120.))
        notes = s.to_note_representation()
        track[0, 0] = self.sos_token
        row = 0
        col = 1
        for n, note in enumerate(notes):  # for all note in the track
            for signature in signatures:  # If there is an activated signature
                if signature.time == 0:
                    num = signature.numerator
                    den = signature.denominator
                    res = self.time_signature_manager.from_fraction_to_time_and_token(num, den)
                    if res is None:
                        self.log.write(
                            "Time signature not known: {} / {}".format(signature.numerator, signature.denominator))
                        return None
                    time_signature, tok = res
                    signatures.remove(signature)
                    track, row, col = self.add_tokens(track, row, col, (tok,))
            for tempo in tempos:  # if time is activated, min-max scaling and print token
                if tempo.time == 0:
                    t = self.min_max_scaling(tempo.qpm, self.tempos_total, self.tempos_compat)
                    tempos.remove(tempo)
                    tok = self.tempos_first_token + t
                    track, row, col = self.add_tokens(track, row, col, (tok,))
            while note[0] >= self.resolution * time_signature:  # add bar till note has time less than bar length
                if row < self.max_bars - 1:  # if we can add another bar
                    track[row][col] = self.eob_token
                    col = 2
                    row += 1
                    track[row][0] = self.sob_token
                    track[row][1] = self.bar_token
                else:
                    return track
                notes[n:, 0] -= round(self.resolution * time_signature)  # decrease all time measures
                for signature in signatures:
                    signature.time -= self.resolution * time_signature
                for tempo in tempos:
                    tempo.time -= self.resolution * time_signature
            if not note[0] < self.num_values * 2:  # check value of time
                self.log.write("Invalid time: " + str(note) + '\n')
                continue
            if not note[1] < self.num_values:  # check value of pitch
                self.log.write("Invalid pitch: " + str(note[1]))
                continue
            if not note[2] < self.num_values:  # clip duration if > 127
                note[2] = self.num_values - 1
                continue
            if self.use_velocity:  # if velocity, use min-max normalization with new interval
                note[3] = self.min_max_scaling(note[3], self.velocities_total, self.velocities_compat)
            else:
                note = note[:-1]  # dont take last note
            track, row, col = self.add_tokens(track, row, col, list(map(add, self.offsets, note)))
        track[row][col] = self.eos_token
        return track

    def transform_song(self, s):
        """
        It takes a filtered song and return the final tensor version of it, making a deep copy of it
        """
        processed = []
        longest_track = 0

        for track in s.tracks:
            just_track = muspy.Music(resolution=self.resolution, tempos=copy.deepcopy(s.tempos),
                                     time_signatures=copy.deepcopy(s.time_signatures))
            just_track.append(copy.deepcopy(track))
            adjusted_track = self.transform_track(just_track)
            if adjusted_track is None:  # if one track is not valid, return None  # TODO this does not happen right?
                return None
            processed.append(adjusted_track)
            if len(adjusted_track) > longest_track:
                longest_track = len(adjusted_track)
        return processed

    def reconstruct_music(self, s):
        """
        It takes a tensor song and return a muspy.Music element, music can be even in bar format,
        it use time_offset to move the beginning of the notes after n bars
        """
        music = muspy.Music(resolution=self.resolution)
        time_signature = 4
        for i, instrument in enumerate(s):  # for each encoder track
            time = 0
            track = muspy.Track(is_drum=i == 0, program=self.reconstruct_programs[i])
            # for bar in instrument:  # for each bar
            notes = instrument.flat
            while len(notes) > 4:  # and bar[0] != self.pad_token:
                # If it is a note
                if self.time_first_token <= notes[0] < self.pitch_first_token <= notes[1] < \
                        self.duration_first_token <= notes[2] < self.velocity_first_token <= notes[3]:
                    note = muspy.Note(notes[0] + time - self.time_first_token,
                                      notes[1] - self.pitch_first_token,
                                      notes[2] - self.duration_first_token,
                                      self.min_max_scaling(notes[3], self.velocities_compat, self.velocities_total))
                    track.append(note)
                    notes = notes[4:]
                elif notes[0] == self.bar_token:  # if it is a bar
                    time += round(self.resolution * time_signature)
                    notes = notes[1:]
                    # track.notes.append(muspy.Note(time=time, pitch=60, duration=12, velocity=127))  # TODO bar sound
                # check if it is a time signature
                # TODO avoid to add signature or tempos 4 times
                elif self.time_signature_manager.from_token_to_time_and_fraction(notes[0]) is not None:
                    time_signature, n, d = self.time_signature_manager.from_token_to_time_and_fraction(notes[0])
                    music.time_signatures.append(muspy.TimeSignature(time=time, numerator=n, denominator=d))
                    notes = notes[1:]
                elif self.tempos_first_token <= notes[0] < self.tempos_first_token + self.tempos_compat[1]+1:  # tempo
                    tempo = self.min_max_scaling(notes[0], self.tempos_compat, self.tempos_total)
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

    def convert_dataset(self):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        if not os.path.exists(os.path.join(self.raw_midi_path, "lmd_matched")):
            print("Downloading Lakh Dataset into "+self.raw_midi_path)
            muspy.LakhMIDIMatchedDataset(self.raw_midi_path, download_and_extract=True)
        print("Converting Lakh Dataset from " + self.raw_midi_path + " in " + self.dataset_path)
        raw_midi = self.raw_midi_path + os.sep + "lmd_matched"
        if self.early_stop == 0:
            dataset_length = sum([len(files) for _, _, files in os.walk(raw_midi)])
        else:
            dataset_length = self.early_stop
        progbar = tqdm(total=dataset_length, leave=True, position=0, desc="Dataset creation")
        count = 0
        os.makedirs(self.dataset_path)
        self.log = open(self.log_file, "w")
        self.log.write("Log of dataset_converter, to check if it is working right\n")
        # lengths = []
        for subdir, dirs, files in os.walk(raw_midi):  # iterate over all subdirectories
            for filename in files:  # iterate over all files
                if self.early_stop == 0 and count > 0:  # if not early stop, update bar anyway
                    progbar.update()
                filepath = subdir + os.sep + filename
                try:
                    song = muspy.read(filepath)
                except Exception as e:  # invalid song format
                    self.log.write(str(count) + " invalid song format: " + e.__str__() + '\n')
                    continue
                filtered_song = self.filter_song(song)
                if filtered_song is None:  # if the song has 4 valid tracks
                    continue
                tensor_song = self.transform_song(filtered_song)
                if tensor_song is None:
                    continue
                tensor_song = np.array(tensor_song).astype(np.int16)
                reconstructed_music = self.reconstruct_music(tensor_song)
                # TODO to save length
                # for elem in tensor_song:
                # lengths.append(len(elem))
                # TODO test
                try:
                    song.write_midi("before.mid")
                except Exception as e:
                    self.log.write(e.__str__()+'\n')
                try:
                    filtered_song.write_midi("middle.mid")
                except Exception as e:
                    self.log.write(e.__str__()+'\n')
                try:
                    reconstructed_music.write_midi("after.mid")
                except Exception as e:
                    self.log.write(e.__str__()+'\n')
                with open(os.path.join(self.dataset_path, str(count) + '.pickle'), 'wb') as f:
                    pickle.dump(tensor_song, f)
                count += 1
                if self.early_stop != 0:  # if early_stop, we update progbar only after a success
                    progbar.update()
                    if count >= self.early_stop:
                        self.log.close()
                        # import matplotlib.pyplot as plt
                        # plt.hist(lengths, density=True, bins=30)  # `density=False` would make counts
                        # plt.ylabel('Probability')
                        # plt.xlabel('Data')
                        # plt.show()
                        return
        self.log.close()


if __name__ == "__main__":
    from muspy_config import config
    manager = NoteRepresentationManager(**config["data"], **config["tokens"], **config["paths"])
