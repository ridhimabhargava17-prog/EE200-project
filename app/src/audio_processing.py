import librosa
import numpy as np

def load_and_preprocess_audio(file_path, target_sr=11025):
    """Loads audio, forces it to mono, and downsamples it."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr

def compute_spectrogram(y, sr, n_fft=1024, hop_length=256):
    """Computes the Short-Time Fourier Transform (STFT) in dB magnitude."""
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    return spectrogram_db