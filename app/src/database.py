import os
import pickle
from .audio_processing import load_and_preprocess_audio, compute_spectrogram
from .fingerprinting import get_2d_peaks, generate_hashes

# 1. Dynamically compute the absolute path based on this script's location
# __file__ is at:   /mount/src/ee200-project/app/src/database.py
# BASE_DIR is at:   /mount/src/ee200-project/app/src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to:        /mount/src/ee200-project/app/data/fingerprints.pkl
DB_FILE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "fingerprints.pkl"))

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
        print(f"Database file not found at {DB_FILE_PATH}. Creating an empty database.")
        return create_empty_database()
    try:
        with open(DB_FILE_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")
        return create_empty_database()

def index_directory(directory_path):
    """Processes an entire folder of tracks and builds the hash database."""
    # Start fresh or load existing data if you want to append
    database = create_empty_database()
    
    # Handle both relative or absolute inputs safely
    if not os.path.isabs(directory_path):
        directory_path = os.path.normpath(os.path.join(BASE_DIR, "..", directory_path))
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist yet.")
        return database

    supported_extensions = ('.mp3', '.wav', '.flac', '.m4a')
    audio_files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported_extensions)]
    
    print(f"Found {len(audio_files)} tracks to process inside {directory_path}...")
    
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
    print(f"Database processing sequence finalized successfully. Saved to: {DB_FILE_PATH}")
    return database