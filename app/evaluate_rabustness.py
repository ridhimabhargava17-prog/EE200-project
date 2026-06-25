import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from src.audio_processing import load_and_preprocess_audio, compute_spectrogram
from src.fingerprinting import get_2d_peaks, generate_hashes
from src.database import load_database
from src.matching import match_query

def create_base_query(source_song_path, duration=15, start_sec=30):
    """Cuts out a clean 15-second fragment from a database song to use as a baseline query."""
    y, sr = librosa.load(source_song_path, sr=11025, mono=True, offset=start_sec, duration=duration)
    return y, sr

def add_white_noise(y, noise_level=0.05):
    """Mixes random Gaussian white noise into the audio signal."""
    noise = np.random.normal(0, noise_level, len(y))
    return y + noise

def apply_pitch_shift(y, sr, n_steps=2):
    """Shifts the pitch up by a specified number of semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def run_test_pipeline(y, sr, db, test_label):
    """Runs a target audio array through the fingerprinting and matching engine."""
    spec = compute_spectrogram(y, sr)
    peaks = get_2d_peaks(spec)
    hashes = generate_hashes(peaks)
    matches = match_query(hashes, db)
    
    print(f"\n--- Test Result for [{test_label}] ---")
    if matches:
        best_song, details = matches[0]
        print(f"Top Matched Song: {best_song}")
        print(f"Histogram Match Score: {details['score']}")
    else:
        print("No match found in the database.")

if __name__ == "__main__":
    # Load your indexed database
    db = load_database()
    if not db:
        print("Error: Database is empty. Please run and index tracks via app.py first.")
        exit()
        
    # Pick one of your actual songs from your directory to test against
    song_folder = "data/database_songs"
    songs = [f for f in os.listdir(song_folder) if f.endswith(('.mp3', '.wav'))]
    
    if not songs:
        print(f"Please place audio files in '{song_folder}' first.")
        exit()
        
    target_song = songs[0] # Test with the first song found
    target_path = os.path.join(song_folder, target_song)
    print(f"Using '{target_song}' to generate experimental fragments...")
    
    # 1. Generate clean baseline query
    y_clean, sr = create_base_query(target_path)
    run_test_pipeline(y_clean, sr, db, "Clean Baseline Clip")
    
    # 2. Experiment A: Add light noise
    y_light_noise = add_white_noise(y_clean, noise_level=0.02)
    run_test_pipeline(y_light_noise, sr, db, "Light Noise (Sigma=0.02)")
    
    # 3. Experiment B: Add heavy noise (breaking point test)
    y_heavy_noise = add_white_noise(y_clean, noise_level=0.15)
    run_test_pipeline(y_heavy_noise, sr, db, "Heavy Noise (Sigma=0.15)")
    
    # 4. Experiment C: Pitch shift up by 2 semitones
    print("\nApplying pitch shift (this might take a few seconds)...")
    y_shifted = apply_pitch_shift(y_clean, sr, n_steps=2)
    run_test_pipeline(y_shifted, sr, db, "Pitch-Shifted (+2 Semitones)")