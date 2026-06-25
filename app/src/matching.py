import numpy as np
from collections import Counter

def match_query(query_hashes, database):
    """
    Compares query hashes against database, computes absolute time 
    offsets (t_base - t_query), and flags relative counts.
    """
    # Key = Song ID, Value = List of matches (t_base - t_query)
    matches_per_song = {}
    
    for query_hash, t_query in query_hashes:
        if query_hash in database:
            for song_id, t_base in database[query_hash]:
                offset = t_base - t_query
                if song_id not in matches_per_song:
                    matches_per_song[song_id] = []
                matches_per_song[song_id].append(offset)
                
    results = {}
    for song_id, offsets in matches_per_song.items():
        # Count identical occurrences of specific delta time values
        counter = Counter(offsets)
        if counter:
            best_offset, highest_peak_count = counter.most_common(1)[0]
            results[song_id] = {
                'score': highest_peak_count,
                'offset': best_offset,
                'all_offsets': offsets
            }
            
    # Sort results showing descending relevance hierarchy based on histogram peaks
    sorted_matches = sorted(results.items(), key=lambda item: item[1]['score'], reverse=True)
    return sorted_matches