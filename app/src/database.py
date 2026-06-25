import os
import pickle
from .audio_processing import load_and_preprocess_audio, compute_spectrogram
from .fingerprinting import get_2d_peaks, generate_hashes

DB_FILE_PATH = "data/fingerprints.pkl"

def create_empty_database():
    """Initializes empty maps linking Hashes -> list of (Song_ID, Time_Offset)."""
    return {}

def save_database(database):
    """Serializes the index lookup hash map to disk storage."""
    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)
    with open(DB_FILE_PATH, 'wb') as f:
        pickle.dump(database, f)

def load_database():
    """Loads serialized index lookup maps from disk."""
    if not os.path.exists(DB_FILE_PATH):
        return create_empty_database()
    with open(DB_FILE_PATH, 'rb') as f:
        return pickle.load(f)

def index_directory(directory_path):
    """Processes an entire folder of tracks and builds the hash database."""
    database = create_empty_database()
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist yet.")
        return database

    supported_extensions = ('.mp3', '.wav', '.flac', '.m4a')
    audio_files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported_extensions)]
    
    print(f"Found {len(audio_files)} tracks to process...")
    
    for song_id in audio_files:
        full_path = os.path.join(directory_path, song_id)
        print(f"Indexing: {song_id}")
        
        try:
            y, sr = load_and_preprocess_audio(full_path)
            spec = compute_spectrogram(y, sr)
            peaks = get_2d_peaks(spec)
            hashes = generate_hashes(peaks)
            
            # Map every single extracted hash to the current song reference map
            for hash_key, t1 in hashes:
                if hash_key not in database:
                    database[hash_key] = []
                database[hash_key].append((song_id, t1))
        except Exception as e:
            print(f"Error indexing track {song_id}: {e}")
            
    save_database(database)
    print("Database processing sequence finalized successfully.")
    return database