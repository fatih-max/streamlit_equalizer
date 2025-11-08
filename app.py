import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from scipy import signal

st.title("🎚️ Software-Defined Audio Mixer dan Equalizer 👌")

# --- Sidebar UI ---
st.sidebar.header("🎛️ Kontrol Mixer")

st.sidebar.subheader("Channel 1")
vol1 = st.sidebar.slider("Volume (dB)", -60.0, 6.0, 0.0, key="vol1")
balance1 = st.sidebar.slider("Balance (L ⟷ R)", -1.0, 1.0, 0.0, key="bal1")

st.sidebar.subheader("Channel 2")
vol2 = st.sidebar.slider("Volume (dB)", -60.0, 6.0, 0.0, key="vol2")
balance2 = st.sidebar.slider("Balance (L ⟷ R)", -1.0, 1.0, 0.0, key="bal2")

st.sidebar.header("🎼 Master EQ (Sesuai Requirement)")
st.sidebar.caption("LPF/BPF/HPF Crossover @ 250Hz & 5kHz")
eq_bass = st.sidebar.slider("Bass Gain (dB)", -12.0, 12.0, 0.0, key="eq_bass")
eq_mid = st.sidebar.slider("Mid Gain (dB)", -12.0, 12.0, 0.0, key="eq_mid")
eq_treble = st.sidebar.slider("Treble Gain (dB)", -12.0, 12.0, 0.0, key="eq_treble")

st.sidebar.subheader("🎵 Upload Audio")
file1 = st.sidebar.file_uploader("Channel 1 (.wav)", type=["wav"], key="file1")
file2 = st.sidebar.file_uploader("Channel 2 (.wav)", type=["wav"], key="file2")

process = st.sidebar.button("🔊 Proses Mixing")

# --- Fungsi bantu (Plot diganti ke Plotly) ---
# (Fungsi plot_waveform tidak diubah)
def plot_waveform(data, sr, title):
    fig = go.Figure()
    duration = len(data) / sr
    time = np.linspace(0, duration, len(data))

    max_points = 10000
    step = max(1, len(time) // max_points)
    time_plot = time[::step]

    if data.ndim == 1:
        y_plot = data[::step]
        fig.add_trace(go.Scatter(x=time_plot, y=y_plot, name='Mono',
                                 line=dict(color='dodgerblue', width=1),
                                 hovertemplate='Time: %{x:.4f}s<br>Amplitude: %{y:.6f}<extra></extra>'))
    else:
        left_plot = data[:, 0][::step]
        right_plot = data[:, 1][::step]
        fig.add_trace(go.Scatter(x=time_plot, y=left_plot, name='Left',
                                 line=dict(color='blue', width=1),
                                 hovertemplate='Time: %{x:.4f}s<br>Left: %{y:.6f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=time_plot, y=right_plot, name='Right',
                                 line=dict(color='orange', width=1),
                                 hovertemplate='Time: %{x:.4f}s<br>Right: %{y:.6f}<extra></extra>'))

    fig.update_layout(
        title=title,
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='zoom',
        hovermode='x unified'
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)

# (Fungsi plot_spectrum tidak diubah)
def plot_spectrum(data, sr, title):
    """
    Plot spektrum interaktif dengan Slider Peak Search.
    Versi ini diperbaiki agar hanya merender plot satu kali.
    """
    fig = go.Figure()

    if data.ndim == 1:
        data = np.stack([data, data], axis=1)

    # --- FFT dan Logika dB ---
    magnitude_db_left = np.full(1, -100.0)
    magnitude_db_right = np.full(1, -100.0)
    freqs = np.array([0])

    for ch_idx, ch_name, color in zip([0, 1], ['Left', 'Right'], ['blue', 'orange']):
        sig = data[:, ch_idx] - np.mean(data[:, ch_idx])
        fft = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        
        magnitude = np.abs(fft)
        ref_max = np.max(magnitude)

        if ref_max < 1e-12:
            magnitude_db = np.full_like(magnitude, -100.0)
        else:
            min_mag = ref_max * (10**(-100 / 20.0)) 
            magnitude_clipped = np.maximum(magnitude, min_mag)
            magnitude_db = 20 * np.log10(magnitude_clipped / ref_max)
        
        if ch_idx == 0:
            magnitude_db_left = magnitude_db
        else:
            magnitude_db_right = magnitude_db

        # Tambahkan trace ke figur utama
        fig.add_trace(go.Scatter(
            x=freqs,
            y=magnitude_db,
            name=f'{ch_name} Channel',
            mode='lines',
            line=dict(color=color, width=1),
            hoverinfo='none' # Sesuai kode asli
        ))

    # --- Logika Peak Search ---
    sig_combined = np.mean(data, axis=1) - np.mean(data, axis=1)
    fft_combined = np.fft.rfft(sig_combined)
    freqs_combined = np.fft.rfftfreq(len(sig_combined), 1 / sr)
    mag_combined = np.abs(fft_combined)
    ref_max_combined = np.max(mag_combined)
    
    magnitude_db_combined = np.full_like(mag_combined, -100.0)
    if ref_max_combined > 1e-12:
        min_mag_combined = ref_max_combined * (10**(-100 / 20.0))
        mag_clipped_combined = np.maximum(mag_combined, min_mag_combined)
        magnitude_db_combined = 20 * np.log10(mag_clipped_combined / ref_max_combined)

    # Cari semua puncak yang signifikan (di atas -90dB)
    peaks, _ = signal.find_peaks(magnitude_db_combined, height=-90, distance=5)

    # --- UI Slider dan Penanda Puncak (Ditempatkan SEBELUM plot) ---
    target_freq = 0
    display_freq = 0
    display_level = -100.0
    
    if len(peaks) == 0:
        st.warning("Tidak ada puncak signifikan yang terdeteksi.")
    else:
        peak_freqs = freqs_combined[peaks]
        peak_levels = magnitude_db_combined[peaks]
        
        min_f = max(20, int(freqs_combined.min()))
        max_f = int(freqs_combined.max())
        
        # 1. Buat Slider Kursor
        target_freq = st.slider(
            "Geser Kursor Puncak (Hz)", 
            min_f, 
            max_f, 
            int(peak_freqs[np.argmax(peak_levels)]), # Default ke puncak tertinggi
            key=f"slider_{title}"
        )
        
        # 2. Cari puncak terdekat dari slider
        closest_peak_idx = np.argmin(np.abs(peak_freqs - target_freq))
        display_freq = peak_freqs[closest_peak_idx]
        display_level = peak_levels[closest_peak_idx]

        # 3. Tampilkan metrik (Sesuai UI asli)
        st.metric(
            f"Puncak Terdekat (dari {target_freq} Hz)", 
            f"{display_freq:.1f} Hz", 
            f"{display_level:.1f} dBFS"
        )
        
        # 4. Tambahkan penanda ke plot
        # Tambahkan Kursor Slider
        fig.add_vline(x=target_freq, line_dash="dash", line_color="grey", annotation_text="Kursor")
        
        # Tambahkan Penanda Puncak
        fig.add_trace(go.Scatter(
            x=[display_freq],
            y=[display_level],
            mode='markers+text',
            text=[f"{display_freq:.1f} Hz"],
            textposition="top center",
            name='Puncak Terdekat',
            marker=dict(color='red', size=10, symbol='x'),
            hoverinfo='none'
        ))

    # --- Layout dan Tampilkan Plot (Hanya satu kali) ---
    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Level (dBFS)",
        yaxis_range=[-100, 5], 
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
        hovermode=False, # Sesuai UI asli
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Sesuai UI asli
    )
    fig.update_xaxes(type="log", rangeslider=dict(visible=True))

    # Tampilkan plot (hanya satu kali render)
    st.plotly_chart(fig, use_container_width=True)

# (Fungsi apply_balance tidak diubah)
def apply_balance(stereo, bal):
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)
    if bal > 0:
        left_gain, right_gain = 1 - bal, 1
    else:
        left_gain, right_gain = 1, 1 + bal
    return np.stack([stereo[:, 0] * left_gain, stereo[:, 1] * right_gain], axis=1)

# --- FUNGSI design_filter DIHAPUS ---
# (Fungsi ini tidak lagi digunakan karena kita
#  menggunakan LPF/BPF/HPF Crossover)

# --- Preview sebelum mixing ---
if file1:
    file1.seek(0)
    data1, sr1 = sf.read(file1)
    st.subheader("🎧 Preview Channel 1")
    st.audio(file1)
    plot_waveform(data1, sr1, "Waveform Channel 1")
    plot_spectrum(data1, sr1, "Spektrum Channel 1")

if file2:
    file2.seek(0)
    data2, sr2 = sf.read(file2)
    st.subheader("🎧 Preview Channel 2")
    st.audio(file2)
    plot_waveform(data2, sr2, "Waveform Channel 2")
    plot_spectrum(data2, sr2, "Spektrum Channel 2")

# --- Proses mixing ---
if process:
    if not file1 or not file2:
        st.error("⚠️ Mohon upload dua file audio terlebih dahulu!")
    else:
        file1.seek(0)
        file2.seek(0)
        data1, sr1 = sf.read(file1)
        data2, sr2 = sf.read(file2)

        if sr1 != sr2:
            st.error("⚠️ Sample rate kedua file harus sama!")
        else:
            min_len = min(len(data1), len(data2))
            data1, data2 = data1[:min_len], data2[:min_len]

            gain1, gain2 = 10 ** (vol1 / 20), 10 ** (vol2 / 20)
            data1, data2 = data1 * gain1, data2 * gain2

            data1 = apply_balance(data1, balance1)
            data2 = apply_balance(data2, balance2)

            mixed = data1 + data2

            # --- IMPLEMENTASI EQ BARU (SESUAI REQUIREMENT) ---
            
            # 1. Tentukan frekuensi cutoff sesuai requirement
            low_cutoff = 250  # Batas LPF
            high_cutoff = 5000 # Batas HPF
            
            # 2. Desain filter LPF, BPF, dan HPF (Orde 2 / 12dB per oktaf)
            #    (Menggunakan filter Butterworth)
            b_lpf, a_lpf = signal.butter(2, low_cutoff, btype='lowpass', fs=sr1)
            b_bpf, a_bpf = signal.butter(2, [low_cutoff, high_cutoff], btype='bandpass', fs=sr1)
            b_hpf, a_hpf = signal.butter(2, high_cutoff, btype='highpass', fs=sr1)

            # 3. Terapkan filter secara PARALEL
            signal_low = signal.lfilter(b_lpf, a_lpf, mixed, axis=0)
            signal_mid = signal.lfilter(b_bpf, a_bpf, mixed, axis=0)
            signal_high = signal.lfilter(b_hpf, a_hpf, mixed, axis=0)

            # 4. Ambil nilai gain dari slider sidebar
            gain_bass = 10**(eq_bass / 20)
            gain_mid = 10**(eq_mid / 20)
            gain_treble = 10**(eq_treble / 20)

            # 5. Terapkan gain ke setiap jalur sinyal dan gabungkan kembali
            eq_output = (signal_low * gain_bass) + \
                        (signal_mid * gain_mid) + \
                        (signal_high * gain_treble)

            # --- AKHIR BLOK EQ BARU ---

            max_abs = np.max(np.abs(eq_output))
            # Normalisasi jika terjadi clipping
            final_output = eq_output / max_abs if max_abs > 1.0 else eq_output

            st.subheader("🎶 Hasil Mixing & EQ")
            sf.write("mixed_output.wav", final_output, sr1)
            st.audio("mixed_output.wav", format="audio/wav")

            plot_waveform(final_output, sr1, "Waveform Output (After Mixing & EQ)")
            plot_spectrum(final_output, sr1, "Spektrum Output (After Mixing & EQ)")

            st.success("✅ Proses selesai! File dapat diunduh di bawah ini.")
            with open("mixed_output.wav", "rb") as f:
                st.download_button("⬇️ Download Hasil", f, file_name="mixed_output.wav")

# --- Generator Audio ---
st.markdown("---")
st.header("🎵 Generate Audio (.wav)")

wave_type = st.selectbox("Pilih bentuk gelombang:", ["Sine", "Square", "Triangle", "Sawtooth", "Noise"])
freq = st.number_input("Frekuensi (Hz)", 100, 5000, 440)
duration = st.number_input("Durasi (detik)", 0.1, 10.0, 2.0)
sr_gen = st.number_input("Sample Rate", 8000, 48000, 44100, key="sr_gen")

if st.button("⚙️ Generate"):
    t = np.linspace(0, duration, int(sr_gen * duration), endpoint=False)
    if wave_type == "Sine":
        y = 0.5 * np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square":
        y = 0.5 * signal.square(2 * np.pi * freq * t)
    elif wave_type == "Triangle":
        y = 0.5 * signal.sawtooth(2 * np.pi * freq * t, 0.5)
    elif wave_type == "Sawtooth":
        y = 0.5 * signal.sawtooth(2 * np.pi * freq * t)
    else: # Noise
        y = 0.5 * np.random.uniform(-1, 1, size=len(t))

    sf.write("generated.wav", y, sr_gen)
    st.success("✅ Audio berhasil dibuat!")
    st.audio("generated.wav", format="audio/wav")
    plot_waveform(y, sr_gen, f"Waveform {wave_type} ({freq} Hz)")
    plot_spectrum(y, sr_gen, f"Spektrum {wave_type}")

    with open("generated.wav", "rb") as f:
        st.download_button("⬇️ Download Generated Audio", f, file_name=f"{wave_type.lower()}_{int(freq)}Hz.wav")

# --- Watermark ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-style:italic;'>Copyright © 2025 2_D4_Telekomunikasi_A_kelompok_1_PDSK All rights reserved.</p>",
    unsafe_allow_html=True
)
