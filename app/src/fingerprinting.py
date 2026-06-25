import numpy as np
import scipy.ndimage as ndimage

def get_2d_peaks(spectrogram_db, threshold_db=-45, neighborhood_size=15):
    """
    Finds local maxima in the 2D spectrogram (The Constellation Map).
    Neighborhood_size controls the minimum distance between adjacent peak stars.
    """
    # Create a mask where each pixel is the local maximum in its neighborhood
    local_max = ndimage.maximum_filter(spectrogram_db, size=neighborhood_size) == spectrogram_db
    
    # Filter background out based on quiet zones
    background = (spectrogram_db < threshold_db)
    erased_background = local_max ^ background
    detected_peaks = local_max & erased_background
    
    # Extract structural coordinate indices where peaks exist
    frequencies_idx, times_idx = np.where(detected_peaks)
    return list(zip(times_idx, frequencies_idx))

def generate_hashes(peaks, target_zone_lookahead=30, target_zone_width=15, fan_out=3):
    """
    Pairs an anchor peak with neighboring peaks in a forward 'Target Zone' 
    to build robust combinatoric hashes.
    """
    hashes = []
    num_peaks = len(peaks)
    
    # Ensure peaks are sorted by time order
    peaks = sorted(peaks, key=lambda x: x[0])
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        
        # Look ahead at consecutive peaks to evaluate the target zone pairing window
        matched_count = 0
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            delta_t = t2 - t1
            
            # Constraints defining the limits of our forward Target Zone window
            if delta_t < 1:  # Skip synchronous or near-instant vertical frames
                continue
            if delta_t > target_zone_lookahead: # Exceeded maximum search window
                break
            if abs(f2 - f1) > target_zone_width: # Restrict matching to vertical bands nearby
                continue
                
            # Create standard robust fingerprint tuple key: (f1, f2, delta_t)
            hash_key = (int(f1), int(f2), int(delta_t))
            hashes.append((hash_key, int(t1)))
            
            matched_count += 1
            if matched_count >= fan_out: # Limit pairs per anchor point to optimize memory space
                break
    return hashes