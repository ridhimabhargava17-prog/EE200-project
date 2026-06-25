import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np

# Import custom inner project backend modules safely
from src.audio_processing import load_and_preprocess_audio, compute_spectrogram
from src.fingerprinting import get_2d_peaks, generate_hashes
from src.database import load_database, index_directory, DB_FILE_PATH
from src.matching import match_query

# Configure Page Theme & Layout
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")

# Custom UI Dark Theme Injection
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #c9d1d9; }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    
    .header-badge {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border-left: 5px solid #00fff2;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        border: 1px solid #30363d;
    }
    
    .track-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    .track-title { color: #00fff2; font-weight: bold; font-size: 1.1rem; }
    .track-meta { color: #8b949e; font-size: 0.9rem; }
    
    .stButton>button {
        background-color: #00fff2 !important; color: #0b0e14 !important;
        font-weight: bold !important; border-radius: 6px !important;
        border: none !important; padding: 0.5rem 1.5rem !important;
    }
    .stButton>button:hover { background-color: #00cccc !important; }
    
    .step-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #8b949e !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Top Header Banner Panel
st.markdown("""
    <div class="header-badge">
        <h1 style='margin:0;'>📊 EE200: Audio Fingerprinting</h1>
        <p style='margin:5px 0 0 0; color:#8b949e;'>SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO</p>
    </div>
""", unsafe_allow_html=True)

# Load Database
db = load_database() if os.path.exists(DB_FILE_PATH) else {}

# Navigation Tabs
tab_lib, tab_id, tab_batch = st.tabs(["🔹 LIBRARY", "🎯 IDENTIFY", "📋 BATCH"])

# --- TAB 1: LIBRARY MANAGEMENT ---
with tab_lib:
    st.subheader("Database Library Management")
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("Build / Reindex Master Database"):
            with st.spinner("Analyzing target directory audio files..."):
                db = index_directory("data/database_songs")
            st.success("Database tracks indexing completed successfully!")
            st.rerun()
    with col_info:
        st.info(f"**Database Status:** Online | **Unique Fingerprint Keys Indexed:** {len(db)}")
        
    st.markdown("---")
    st.write("### Currently Indexed Library Tracks")
    song_dir = "data/database_songs"
    if os.path.exists(song_dir):
        supported_exts = ('.mp3', '.wav', '.flac', '.m4a')
        audio_files = [f for f in os.listdir(song_dir) if f.lower().endswith(supported_exts)]
        if audio_files:
            cols = st.columns(4)
            for idx, filename in enumerate(audio_files):
                with cols[idx % 4]:
                    st.markdown(f'<div class="track-card"><div class="track-title">{os.path.splitext(filename)[0]}</div><div class="track-meta">Format: {os.path.splitext(filename)[1].upper()}</div></div>', unsafe_allow_html=True)

# --- TAB 2: IDENTIFY SINGLE CLIP ---
with tab_id:
    st.subheader("Identify a clip")
    uploaded_file = st.file_uploader("Upload a snippet...", type=["mp3", "wav", "m4a", "flac"], key="single_id")
    
    if uploaded_file is not None:
        temp_path = "data/temp_query.wav"
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Identify", key="btn_run_id"):
            with st.spinner("Processing acoustic features..."):
                y_q, sr_q = load_and_preprocess_audio(temp_path)
                spec_q = compute_spectrogram(y_q, sr_q)
                peaks_q = get_2d_peaks(spec_q)
                hashes_q = generate_hashes(peaks_q)
                matches = match_query(hashes_q, db)
                
            if matches:
                best_song, details = matches[0]
                
                # --- STEP 1: CONSTELLATION EXTRACTION MAPS ---
                st.markdown('<div class="step-header">Step 1 • Acoustic Feature Extraction</div>', unsafe_allow_html=True)
                st.write(f"From that rich image, only the **{len(peaks_q)} most prominent peaks** were kept. Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.")
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_facecolor('#0b0e14')
                    ax.set_facecolor('#161b22')
                    ax.imshow(spec_q, origin='lower', aspect='auto', cmap='magma', extent=[0, len(y_q)/sr_q, 0, sr_q/2])
                    ax.set_title("Raw Spectrogram Magnitude", color='white', fontsize=10)
                    ax.tick_params(colors='#8b949e', labelsize=8)
                    ax.set_ylim(0, 4000)
                    st.pyplot(fig)
                    
                with col_s2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_facecolor('#0b0e14')
                    ax.set_facecolor('#161b22')
                    if len(peaks_q) > 0:
                        p_times = [p[0] * (256/sr_q) for p in peaks_q]
                        p_freqs = [p[1] * (sr_q/1024) for p in peaks_q]
                        ax.scatter(p_times, p_freqs, color='#00fff2', s=8, facecolors='none', alpha=0.8)
                    ax.set_title(f"Constellation Map ({len(peaks_q)} peaks)", color='white', fontsize=10)
                    ax.tick_params(colors='#8b949e', labelsize=8)
                    ax.set_xlim(0, len(y_q)/sr_q)
                    ax.set_ylim(0, 4000)
                    st.pyplot(fig)

                # --- STEP 2: WHERE IN THE SONG GLOBAL SCATTER MAP ---
                st.markdown('<div class="step-header">Step 2 • Database Search</div>', unsafe_allow_html=True)
                st.markdown(f"### Where in the song?")
                st.write(f"The **{len(hashes_q)} fingerprint hashes** were looked up against every indexed track. Below is the full reconstructed fingerprint of *{os.path.splitext(best_song)[0]}*. The highlighted window shows exactly where the query clip sits inside the master timeline.")
                
                # Fetch all database anchor times for the winning track to build the global scatter
                all_db_time_frames = []
                all_db_freq_bins = []
                for h_key, structural_instances in db.items():
                    for s_id, t_base in structural_instances:
                        if s_id == best_song:
                            all_db_time_frames.append(t_base)
                            all_db_freq_bins.append(h_key[0]) # anchor frequency bin
                
                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor('#0b0e14')
                ax.set_facecolor('#161b22')
                ax.scatter(all_db_time_frames, all_db_freq_bins, color='#00fff2', s=2, alpha=0.3, label="Database Hashes")
                
                # Draw the bounding box window showing where the query landed
                best_offset = details['offset']
                query_duration_frames = spec_q.shape[1]
                start_box = best_offset
                end_box = best_offset + query_duration_frames
                
                ax.axvspan(start_box, end_box, color='#00fff2', alpha=0.15, edgecolor='#00fff2', linestyle='--', label='Query Clip Match Window')
                ax.set_title(f"Full Fingerprint Map of {os.path.splitext(best_song)[0]} (Query Window Highlighted)", color='white', fontsize=11)
                ax.set_xlabel("Time (frames)", color='white', fontsize=9)
                ax.set_ylabel("Frequency bin", color='white', fontsize=9)
                ax.tick_params(colors='#8b949e', labelsize=8)
                st.pyplot(fig)

                # --- STEP 3: THE ALIGNMENT SPIKE ---
                st.markdown('<div class="step-header">Step 3 • The Proof</div>', unsafe_allow_html=True)
                st.markdown("### The alignment spike")
                st.write("Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge spectacularly on a single offset.")
                
                fig, ax = plt.subplots(figsize=(12, 3.5))
                fig.patch.set_facecolor('#0b0e14')
                ax.set_facecolor('#161b22')
                
                counts, bins, patches = ax.hist(details['all_offsets'], bins=100, color='#1f2937', edgecolor='#30363d', alpha=0.7)
                # Color the maximum peak bar vibrant orange to reflect the course video layout style
                max_bin_idx = np.argmax(counts)
                for i, patch in enumerate(patches):
                    if i == max_bin_idx:
                        patch.set_facecolor('#ff9f43')
                        patch.set_edgecolor('#ff9f43')
                        patch.set_alpha(1.0)
                        
                ax.annotate(f"{int(details['score'])} hashes\nalign here", 
                            xy=(bins[max_bin_idx], counts[max_bin_idx]), 
                            xytext=(bins[max_bin_idx] + (max(bins)*0.05), counts[max_bin_idx] * 0.7),
                            color='#ff9f43', weight='bold', fontsize=10,
                            arrowprops=dict(arrowstyle="->", color='#ff9f43', lw=1.5))
                
                ax.text(max(bins)*0.75, max(counts)*0.05, "chance matches\n(noise floor)", color='#8b949e', fontsize=9, ha='center')
                ax.set_title("Time Offset Alignment Peak Analysis Histogram", color='white', fontsize=11)
                ax.set_xlabel("Time offset (database frame — query frame)", color='white', fontsize=9)
                ax.set_ylabel("# hashes", color='white', fontsize=9)
                ax.tick_params(colors='#8b949e', labelsize=8)
                st.pyplot(fig)
                
            else:
                st.error("No valid matching track could be determined from the active database records.")

# --- TAB 3: BATCH IDENTIFICATION ---
with tab_batch:
    st.subheader("Identify many clips at once")
    uploaded_files = st.file_uploader("Upload a set of query clips...", type=["mp3", "wav", "m4a", "flac"], accept_multiple_files=True, key="batch_id")
    if uploaded_files:
        if st.button("Run batch", key="btn_run_batch"):
            results_list = []
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                b_path = f"data/batch_{file.name}"
                with open(b_path, "wb") as f:
                    f.write(file.getbuffer())
                y_b, sr_b = load_and_preprocess_audio(b_path)
                spec_b = compute_spectrogram(y_b, sr_b)
                peaks_b = get_2d_peaks(spec_b)
                hashes_b = generate_hashes(peaks_b)
                matches_b = match_query(hashes_b, db)
                prediction = os.path.splitext(matches_b[0][0])[0] if matches_b else "none"
                results_list.append({"FILE": file.name, "PREDICTION": prediction})
                progress_bar.progress((idx + 1) / len(uploaded_files))
                os.remove(b_path)
            st.markdown("### Results")
            st.dataframe(results_list, use_container_width=True)