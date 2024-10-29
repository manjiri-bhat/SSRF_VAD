# %%
import numpy as np
import soundfile as sf


# %%
import os
import glob
import librosa

# %%
def mix_audio(signal, noise, snr):
    # if the audio is longer than the noise
    # play the noise in repeat for the duration of the audio
    noise = noise[np.arange(len(signal)) % len(noise)]
    
    # if the audio is shorter than the noise
    # this is important if loading resulted in 
    # uint8 or uint16 types, because it would cause overflow
    # when squaring and calculating mean
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)
    
    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise 
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    
    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    print(g, a, b)
    # mix the signals
    return a * signal + b * noise

# %%
speech_path = "/FSDD/recordings" #full path to speech audio
noise_path = "/Urbansound8K/audio" #full path to noise audio
Output_dir = "/overlap_at_snr_10/" #Path for files mixed at specified SNR
file_ext="*.wav"

# %%
files_speech = os.listdir(speech_path)
files_noise = os.listdir(noise_path)

# %%
for file1, file2 in zip(files_speech, files_noise):
    file1_path = os.path.join(speech_path, file1)
    file2_path = os.path.join(noise_path, file2)
    signal, sr = librosa.load(file1_path, sr=22050)
    noise, nsr = librosa.load(file2_path, sr=22050)
    noisy = mix_audio(signal, noise, 10)
    op_file_name = file1 + file2
    op_file = os.path.join(Output_dir,op_file_name)
    sf.write(op_file,noisy,22050)


# Example run
#op_file_name = './overlap_at_snr_0/trial.wav'
#signal, sr = librosa.load('FSDD/recordings/0_george_27.wav', sr=22050)
#noise, nsr = librosa.load('Urbansound8K/audio/344-3-0-0.wav', sr=22050)
#noisy = mix_audio(signal, noise, 0)
#sf.write(op_file_name,noisy,22050)


