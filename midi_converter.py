import os
import subprocess
"""
Manual is available in
https://github.com/FluidSynth/fluidsynth/wiki/UserManual
Sound font can be downloaded from 
http://timtechsoftware.com/ad.html?keyword=sf2%20format?file_name=the%20General%20MIDI%20Soundfont?file_url=uploads/
GeneralUser_GS_SoftSynth_v144.sf2
"""


def midi_to_wav(input_file, output_file):
    # Get kind of execution
    if os.getcwd() == 'C:\\Users\\berti\\PycharmProjects\\MusAE':
        remote = False
    else:
        remote = True
    # Check
    # if not os.path.exists(os.path.join("fl", "bin")):
    #     raise Exception("To convert MIDI to WAV you need to install fluidsynth and soundfont, "
    #                     "see midi_converter.py for more explanation")
    subprocess.call(["fluidsynth" if remote else os.path.join("fl", "bin", "fluidsynth"),
                     "-F", output_file,
                     # "-i", "-n", "-T", "wav",  # those seems to be useless
                     # "-q",  # activate quiet mode
                     "-r", "8000",
                     # "-T", "raw",  # audio type
                     "sound_font.sf2" if remote else os.path.join("fl", "sound_font.sf2"),  # a sound font
                     input_file])


if __name__ == "__main__":
    midi_to_wav("prova.mid", "out.wav")
