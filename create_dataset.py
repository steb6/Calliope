import muspy
import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
from manage_time_signature import TimeSignatureManager
import copy


# now everything takes the double because I passed from int8 to int16
# each song take 200 KB
# 5000 songs take 1 GB
# 100'000 songs take 200 GB
# TODO add time signatures and times to representation

class NoteRepresentationManager:
    """
    This class has all the function needed to process a Note Representation and convert it
    """

    def __init__(self, resolution=None, tempo=None, time_first_token=None, pitch_first_token=None,
                 duration_first_token=None, velocity_first_token=None, pad_token=None, num_values=None,
                 use_velocity=None, bar_token=None, sos_token=None, eos_token=None,
                 reconstruct_programs=None, max_bar_length=None, max_track_length=None, early_stop=None,
                 raw_midi_path=None, dataset_path=None, log_file="dataset_converter_log.txt",
                 one_four_token=None, two_four_token=None, three_four_token=None, four_four_token=None,
                 five_four_token=None, six_four_token=None, seven_four_token=None, eight_four_token=None,
                 three_eight_token=None, five_eight_token=None, six_eight_token=None, seven_eight_token=None,
                 nine_eight_token=None, twelve_eight_token=None, two_two_token=None,
                 velocities_interval=None, velocities_values=None,
                 tempos_interval=None, tempos_values=None, tempos_first_token=None
                 ):
        self.resolution = resolution
        self.tempo = tempo
        self.max_bar_length = max_bar_length  # 1 note for resolution x note info
        self.pad_token = pad_token
        self.bar_token = bar_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.time_first_token = time_first_token
        self.pitch_first_token = pitch_first_token
        self.duration_first_token = duration_first_token
        self.velocity_first_token = velocity_first_token
        self.use_velocity = use_velocity
        self.num_values = num_values
        self.log_file = log_file
        self.log = None
        if self.use_velocity:
            self.offsets = [self.time_first_token, self.pitch_first_token,
                            self.duration_first_token, self.velocity_first_token]
        else:
            self.offsets = [self.time_first_token, self.pitch_first_token, self.duration_first_token]
        self.reconstruct_programs = reconstruct_programs
        self.max_track_length = max_track_length
        self.early_stop = early_stop
        self.raw_midi_path = raw_midi_path
        self.dataset_path = dataset_path
        self.velocities_interval = velocities_interval
        self.velocities_values = velocities_values
        self.tempos_interval = tempos_interval
        self.tempos_values = tempos_values
        self.tempos_first_token = tempos_first_token
        self.time_signature_manager = TimeSignatureManager(two_two_token=two_two_token,
                                                           one_four_token=one_four_token,
                                                           two_four_token=two_four_token,
                                                           three_four_token=three_four_token,
                                                           four_four_token=four_four_token,
                                                           five_four_token=five_four_token,
                                                           six_four_token=six_four_token,
                                                           seven_four_token=seven_four_token,
                                                           eight_four_token=eight_four_token,
                                                           three_eight_token=three_eight_token,
                                                           five_eight_token=five_eight_token,
                                                           six_eight_token=six_eight_token,
                                                           seven_eight_token=seven_eight_token,
                                                           nine_eight_token=nine_eight_token,
                                                           twelve_eight_token=twelve_eight_token)

    def filter_song(self, s):
        """
        Get a muspy.Music object
        return the song with the 4 instruments with more notes, with resolution of 24 and tempo of 120
        or None if the song does not have enough instruments
        """
        for time in s.time_signatures:  # check time signature
            if not self.time_signature_manager.is_valid_time_signature(time.numerator, time.denominator):
                self.log.write("Song with weird time skipped: {} / {}\n".format(time.numerator, time.denominator))
                return None
        for time in s.tempos:  # adjust tempo
            if time.qpm < 60:
                time.qpm = 60
            if time.qpm > 180:
                time.qpm = 180
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        s.adjust_resolution(self.resolution)
        s.clip()
        sys.stdout = old_stdout
        drum = None
        guitar = None
        bass = None
        strings = None
        for track in s.tracks:
            if track.is_drum:
                if drum is None or len(track.notes) > len(drum.notes):
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
        new = muspy.Music(tempos=s.tempos if len(s.tempos) > 0 else [muspy.Tempo(time=0, qpm=120)],
                          time_signatures=s.time_signatures if len(s.time_signatures) > 0 else
                          [muspy.TimeSignature(time=0, numerator=4, denominator=4)],
                          resolution=self.resolution,
                          )
        new.tracks = [drum, guitar, bass, strings]
        return new

    def transform_track(self, s):
        """
        It transform the notes notation into a list of numbers which are the bars
        """
        bars = [self.sos_token]
        time_signature = 4
        signatures = list(s.time_signatures)
        tempos = list(s.tempos)
        if len(signatures) == 0:
            signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        notes = s.to_note_representation()
        for i, note in enumerate(notes):  # for all note in the track
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
                    bars.append(tok)
            for tempo in tempos:  # if time is activated, min-max scaling and print token
                if tempo.time == 0:
                    t = round(((tempo.qpm - self.tempos_interval[0]) /
                               (self.tempos_interval[1] - self.tempos_interval[0])) * self.tempos_values)
                    tempos.remove(tempo)
                    bars.append(self.tempos_first_token + t)
            while note[0] >= self.resolution * time_signature:  # add bar till note has time less than bar length
                bars.append(self.bar_token)
                notes[i:, 0] -= round(self.resolution * time_signature)  # decrease all time measures
                for signature in signatures:
                    signature.time -= self.resolution * time_signature
                for tempo in tempos:
                    tempo.time -= self.resolution * time_signature
            if note[2] > self.num_values - 1:  # clip duration if > 127
                note[2] = self.num_values - 1
            if note[0] > self.num_values * 2:  # check value of time
                self.log.write("Invalid time: " + str(note) + '\n')
                return None
            if self.use_velocity:  # if velocity, use min-max normalization with new interval
                note[3] = round((((note[3] - self.velocities_interval[0]) /
                                  (self.velocities_interval[1] - self.velocities_interval[
                                      0]))) * self.velocities_values)
            if not all(value <= self.num_values - 1 for value in note[1:]):  # check if one value has value too high
                self.log.write("Invalid note: " + str(note) + '\n')
                return None
            for offset, value in zip(self.offsets, note):  # append note
                bars.append(value + offset)
        bars.append(self.eos_token)
        if len(bars) > self.max_track_length:
            # return None TODO here I returned None to avoid to cut the track, but now i trying to cut it
            bars = bars[:self.max_track_length - 1]
            bars.append(self.eos_token)
        if len(bars) < self.max_track_length:
            bars = bars + [self.pad_token] * (self.max_track_length - len(bars))
        return bars

    def transform_song(self, s):
        """
        It takes a filtered song and return the final tensor version of it
        """
        processed = []
        longest_track = 0

        for track in s.tracks:
            just_track = muspy.Music(resolution=self.resolution, tempos=copy.deepcopy(s.tempos),
                                     time_signatures=copy.deepcopy(s.time_signatures))
            just_track.append(track)
            adjusted_track = self.transform_track(just_track)
            if adjusted_track is None:  # if one track is not valid, return None
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
            while len(instrument) > 4:  # and bar[0] != self.pad_token:
                # If it is a note
                if self.time_first_token <= instrument[0] < self.pitch_first_token <= instrument[1] < \
                        self.duration_first_token <= instrument[2] < self.velocity_first_token <= instrument[3]:
                    note = muspy.Note(instrument[0] + time - self.time_first_token,
                                      instrument[1] - self.pitch_first_token,
                                      instrument[2] - self.duration_first_token,
                                      round(((instrument[3] - self.velocity_first_token) *
                                             (self.velocities_interval[1] - self.velocities_interval[0])) /
                                            self.velocities_values) + self.velocities_interval[0]
                                      )
                    track.append(note)
                    instrument = instrument[4:]
                elif instrument[0] == self.bar_token:  # if it is a bar
                    time += round(self.resolution * time_signature)
                    instrument = instrument[1:]
                    track.notes.append(muspy.Note(time=time, pitch=127, duration=24, velocity=127)) # TODO sound of bar
                # check if it is a time signature
                elif self.time_signature_manager.from_token_to_time_and_fraction(instrument[0]) is not None:
                    time_signature, n, d = self.time_signature_manager.from_token_to_time_and_fraction(instrument[0])
                    music.time_signatures.append(muspy.TimeSignature(time=time, numerator=n, denominator=d))
                    instrument = instrument[1:]
                elif self.tempos_first_token <= instrument[0] < self.tempos_first_token + self.tempos_values:  # tempo
                    tempo = round(((instrument[0] - self.tempos_first_token) *
                                  (self.tempos_interval[1] - self.tempos_interval[0])) / self.tempos_values) \
                            + self.tempos_interval[0]
                    music.tempos.append(muspy.Tempo(time=time, qpm=tempo))  # TODo avoid to add tempos 4 times
                    instrument = instrument[1:]
                else:  # unknown combination, just skip
                    instrument = instrument[1:]
            music.append(track)
        if len(music.tempos) == 0:
            music.tempos.append(muspy.Tempo(time=0, qpm=120))
        if len(music.time_signatures) == 0:
            music.time_signatures.append(muspy.TimeSignature(time=0, numerator=4, denominator=4))
        return music

    def convert_dataset(self):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        if not os.path.exists(os.path.join(self.raw_midi_path, "lmd_matched")):
            muspy.LakhMIDIMatchedDataset(self.raw_midi_path, download_and_extract=True)
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
                except Exception:  # invalid song format
                    self.log.write(str(count) + " invalid song format\n")
                    continue
                filtered_song = self.filter_song(song)
                if filtered_song is None:  # if the song has 4 valid tracks
                    continue
                tensor_song = self.transform_song(filtered_song)
                if tensor_song is None:
                    continue
                tensor_song = np.array(tensor_song).astype(np.int16)
                # TODO to save length
                # for elem in tensor_song:
                # lengths.append(len(elem))
                # TODO test
                try:
                    song.write_midi("before.mid")
                except Exception as e:
                    self.log.write(e.__str__()+'\n')
                reconstructed_music = self.reconstruct_music(tensor_song)
                try:
                    reconstructed_music.write_midi("after.mid")
                except Exception as e:
                    self.log.write(e.__str__()+'\n')
                # TODO save entire song
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

# TODO Save each bar separately with pickle
# while tensor_song.shape[1] % batch_size != 0:  # append bar till we have batch dimension
#     pad = np.full((4, 1, self.max_bar_length), self.pad_token, dtype=np.int16)
#     tensor_song = np.append(tensor_song, pad, axis=1)
# tensor_song = np.swapaxes(tensor_song, 0, 1)
# for i, bar in enumerate(tensor_song):
#     bar_name = os.path.join(dataset_path, str(count)+'-'+str(i)+'.pickle')
#     with open(bar_name, 'wb') as file:
#         pickle.dump(tensor_song[i], file)
# TODO with hdf5
# for i, b in enumerate(range(0, len(tensor_song), batch_size)):  # iterate over batches
#     hf = h5py.File(os.path.join(destination_path, str(count)+'-'+str(i)+".h5"), 'w')
#     hf.create_dataset('tensor', data=tensor_song[b:b+batch_size])
#     hf.close()
# hf = h5py.File(os.path.join(destination, str(count)+".h5"), 'r')
# n1 = np.array(hf.get('tensor'))
