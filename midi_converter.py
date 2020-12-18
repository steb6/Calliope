import os
import subprocess
from config import remote
"""
- Manual is available in
    https://github.com/FluidSynth/fluidsynth/wiki/UserManual
- Sound font can be downloaded from 
    http://timtechsoftware.com/ad.html?keyword=sf2%20format?file_name=the%20General%20MIDI%20Soundfont?file_url=uploads/
    GeneralUser_GS_SoftSynth_v144.sf2
- To install FluidSync for windows, download the executable, for unix use conda
"""


def midi_to_wav(input_file, output_file):
    subprocess.call(["fluidsynth" if remote else os.path.join("fl", "bin", "fluidsynth"),
                     "-F", output_file,
                     # "-i", "-n", "-T", "wav",  # those seems to be useless
                     # "-q",  # activate quiet mode
                     "-r", "8000",
                     # "-T", "raw",  # audio type
                     "sound_font.sf2" if remote else os.path.join("fl", "sound_font.sf2"),  # a sound font
                     input_file])
