import muspy
import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
from muspy_config import config
# now everything takes the double because I passed from int8 to int16
# each song take 200 KB
# 5000 songs take 1 GB
# 100'000 songs take 200 GB


class NoteRepresentationManager:
    """
    Tokens are expected one after the other and should be the greatest admissible values
    """
    def  __init__(self, resolution=None, tempo=None, time_first_token=None, pitch_first_token=None,
                 duration_first_token=None, velocity_first_token=None, pad_token=None, num_values=None,
                 use_velocity=None, log_file=None, bar_token=None, sos_token=None, eos_token=None,
                 reconstruct_programs=None, max_bar_length=None):
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

    def filter_song(self, s):
        """
        Get a muspy.Music object
        return the song with the 4 instruments with more notes, with resolution of 24 and tempo of 120
        or None if the song does not have enough instruments
        """
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
        new = muspy.Music(tempos=[muspy.Tempo(0, self.tempo)])
        new.tracks = [drum, guitar, bass, strings]
        return new

    def transform_track(self, n):
        """
        It transform the notes notation into a list of numbers which are the bars
        """
        bars = []
        bar = []
        for i, note in enumerate(n):  # for all note in the track
            while note[0] >= self.resolution * 4:  # add bar till note has time less than resolution * 4
                if len(bar) > 0:  # TODO add bar event when I will give entire songs
                    bar.append(self.eos_token)
                while len(bar) < self.max_bar_length:  # pad previous bar to max_bar_length
                    bar.append(self.pad_token)
                bars.append(bar)
                bar = []
                n[i:, 0] -= self.resolution * 4
            if note[2] > self.num_values-1:  # clip duration if > 127
                note[2] = self.num_values-1
            if not all(value <= self.num_values-1 for value in note):  # check if one value has value too high
                self.log.write("Invalid note: "+str(note))
                return None
            if len(bar) == 0:
                bar.append(self.sos_token)
            for offset, value in zip(self.offsets, note):  # append note
                bar.append(value+offset)
            if len(bar) > self.max_bar_length:  # it enters here if it keeps adding notes and notes[0] is never >= 128
                self.log.write("Bar too long: "+str(len(bar))+'\n')
                return None  # bar is too long, skip this song
        return bars

    def transform_song(self, s):
        """
        It takes a filtered song and return the final tensor version of it
        """
        processed = []
        longest_track = 0

        for track in s.tracks:
            just_track = muspy.Music(resolution=self.resolution, tempos=[muspy.Tempo(0, self.tempo)])
            just_track.append(track)
            notes = just_track.to_note_representation()
            adjusted_track = self.transform_track(notes)
            if adjusted_track is None:  # if one track is not valid, return None
                return None
            processed.append(adjusted_track)
            if len(adjusted_track) > longest_track:
                longest_track = len(adjusted_track)
        #  add bar to match longest track
        for track in processed:
            while(len(track)) < longest_track:
                track.append([self.pad_token] * self.max_bar_length)

        return processed

    def reconstruct_music(self, s, time_offset=0):
        """
        It takes a tensor song and return a muspy.Music element, music can be even in bar format,
        it use time_offset to move the beginning of the notes after n bars
        """
        music = muspy.Music(resolution=self.resolution, tempos=[muspy.Tempo(0, self.tempo)])

        for i, instrument in enumerate(s):  # for each encoder track
            time = self.resolution*4*time_offset  # this has default value 0
            track = muspy.Track(is_drum=i == 0, program=self.reconstruct_programs[i])
            for bar in instrument:  # for each bar
                while len(bar) > 4:  # and bar[0] != self.pad_token:
                    if self.time_first_token <= bar[0] < self.pitch_first_token <= bar[1] < self.duration_first_token <= \
                            bar[2] < self.velocity_first_token <= bar[3]:
                        note = muspy.Note(bar[0]+time-self.time_first_token, bar[1]-self.pitch_first_token,
                                          bar[2]-self.duration_first_token, bar[3]-self.velocity_first_token)
                        track.append(note)
                        bar = bar[4:]
                    else:
                        bar = bar[1:]
                time += self.resolution*4
            music.append(track)
        return music

    def convert_dataset(self, raw_midi=None, dataset_path=None, early_stop=None, batch_size=1):
        """
        Given a dataset path and a destination path, it walks all directories of dataset
        and for each song create a tensor
        """
        if not os.path.exists(os.path.join(raw_midi, "lmd_matched")):
            muspy.LakhMIDIMatchedDataset(raw_midi, download_and_extract=True)
        raw_midi = raw_midi + os.sep + "lmd_matched"
        if early_stop == 0:
            dataset_length = sum([len(files) for _, _, files in os.walk(raw_midi)])
        else:
            dataset_length = early_stop
        progbar = tqdm(total=dataset_length, leave=True, position=0, desc="Dataset creation")
        count = 0
        os.makedirs(dataset_path)
        self.log = open(self.log_file, "w")
        self.log.write("Log of dataset_converter, to check if it is working right\n")
        for subdir, dirs, files in os.walk(raw_midi):  # iterate over all subdirectories
            for filename in files:  # iterate over all files
                if early_stop == 0 and count > 0:  # if not early stop, update bar anyway
                    progbar.update()
                filepath = subdir + os.sep + filename
                try:
                    song = muspy.read(filepath)
                except Exception:  # invalid song format
                    self.log.write(str(count)+" invalid song format\n")
                    continue
                filtered_song = self.filter_song(song)
                if filtered_song is None:  # if the song has 4 valid tracks
                    continue
                tensor_song = self.transform_song(filtered_song)
                if tensor_song is None:
                    continue
                tensor_song = np.array(tensor_song).astype(np.int16)
                # Test
                # song.write_midi("before.mid")
                # reconstructed_music = self.reconstruct_music(tensor_song)
                # reconstructed_music.write_midi("after.mid")
                # Save entire song
                with open(os.path.join(dataset_path, str(count)+'.pickle'), 'wb') as f:
                    pickle.dump(tensor_song, f)
                # Save each bar
                # while tensor_song.shape[1] % batch_size != 0:  # append bar till we have batch dimension
                #     pad = np.full((4, 1, self.max_bar_length), self.pad_token, dtype=np.int16)
                #     tensor_song = np.append(tensor_song, pad, axis=1)
                # tensor_song = np.swapaxes(tensor_song, 0, 1)
                # for i, bar in enumerate(tensor_song):  # TODO save bar so loader can load batch, is it right?
                #     bar_name = os.path.join(dataset_path, str(count)+'-'+str(i)+'.pickle')
                #     with open(bar_name, 'wb') as file:
                #         pickle.dump(tensor_song[i], file)
                # for i, b in enumerate(range(0, len(tensor_song), batch_size)):  # iterate over batches
                #     hf = h5py.File(os.path.join(destination_path, str(count)+'-'+str(i)+".h5"), 'w')
                #     hf.create_dataset('tensor', data=tensor_song[b:b+batch_size])
                #     hf.close()
                # hf = h5py.File(os.path.join(destination, str(count)+".h5"), 'r')
                # n1 = np.array(hf.get('tensor'))
                count += 1
                if early_stop != 0:  # if early_stop, we update progbar only after a success
                    progbar.update()
                    if count >= early_stop:
                        self.log.close()
                        return
        self.log.close()


if __name__ == "__main__":
    data = NoteRepresentationManager(resolution=config.resolution,
                                     tempo=config.tempo,
                                     pad_token=config.pad_token,
                                     sos_token=config.sos_token,
                                     bar_token=config.bar_token,
                                     eos_token=config.eos_token,
                                     num_values=config.num_values,
                                     time_first_token=config.time_first_token,
                                     pitch_first_token=config.pitch_first_token,
                                     duration_first_token=config.duration_first_token,
                                     velocity_first_token=config.velocity_first_token,
                                     use_velocity=config.use_velocity,
                                     log_file=config.dataset_converter_log_file,
                                     reconstruct_programs=config.reconstruct_programs,
                                     max_bar_length=config.max_bar_length
                                     )
    data.convert_dataset(raw_midi=config.raw_midi_path, dataset_path=config.dataset_path,
                         early_stop=config.early_stop)
