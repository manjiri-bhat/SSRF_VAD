# %%
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import IPython.display as ipd
import pathlib
import os
import glob
import numpy as np
import IPython.display as ipd
import soundfile as sf

# %%
data_path = ("") #add path to data folder
l= os.listdir(data_path)
file_ext="*.wav"



# %% [markdown]
# Time_stretch
# 

# %%
output_path = ("./fsdd_time_stretched/")
for filename in glob.glob(os.path.join(data_path, file_ext)):
    print(filename)
    signal, sr = librosa.load(filename)
    wav_time_stch = librosa.effects.time_stretch(signal,rate=0.5)
    fl = filename.split('\\')[1]
    op_file_name = os.path.join(output_path,fl)
    sf.write(op_file_name, wav_time_stch, 22050 )
    
    

# %% [markdown]
# Pitch Shift

# %%
output_path = ("./fsdd_pitch_shifted/") #Add your output folder path
for filename in glob.glob(os.path.join(data_path, file_ext)):
    print(filename)
    signal, sr = librosa.load(filename)
    wav_pitch_sf = librosa.effects.pitch_shift(signal,sr=sr,n_steps=-2) #-2 semitone shift
    fl = 'negative_2_' + filename.split('\\')[1] 
    op_file_name = os.path.join(output_path,fl)
    sf.write(op_file_name, wav_pitch_sf, 22050 )

# %%


# %% [markdown]
# 


