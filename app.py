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
def plot_waveform(data, sr, title):
    fig = go.Figure()
    duration = len(data) / sr
    time = np.linspace(0, duration, len(data))

    max_points = 10000
    step = max(1, len(time) // max_points)
    time_plot = time[::step]

    # Pastikan data tidak kosong
    if data.size == 0:
        st.warning(f"Tidak ada data untuk di-plot di {title}")
        return

    if data.ndim == 1:
        y_plot = data[::step]
        fig.add_trace(go.Scatter(x=time_plot, y=y_plot, name='Mono',
                                 line=dict(color='dodgerblue', width=1),
                                 hovertemplate='Time: %{x:.4f}s<br>Amplitude: %{y:.6f}<extra></extra>'))
    else:
        left_plot = data[::step, 0]
        right_plot = data[::step, 1]
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
        hovermode='x unified',
        yaxis_range=[-1.1, 1.1] # Beri sedikit ruang untuk melihat clipping
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)

def plot_spectrum(data, sr, title):
    fig = go.Figure()

    if data.ndim == 1:
        # Jika mono, buat jadi 2 channel duplikat untuk plotting
        data = np.stack([data, data], axis=1)
    
    # Pastikan data tidak kosong
    if data.size == 0:
        st.warning(f"Tidak ada data untuk di-plot di {title}")
        return

    # --- FFT dan Logika dB ---
    freqs = np.array([0])
    
    # Cari referensi maksimum dari kedua channel agar skalanya sama
    ref_max_global = 1e-12 # Hindari log(0)
    fft_results = []
    
    for ch_idx in [0, 1]:
        sig = data[:, ch_idx] - np.mean(data[:, ch_idx])
        if len(sig) == 0: continue
            
        fft = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        magnitude = np.abs(fft)
        ref_max_channel = np.max(magnitude)
        
        if ref_max_channel > ref_max_global:
            ref_max_global = ref_max_channel
            
        fft_results.append((magnitude, freqs))

    # Plot menggunakan ref_max_global
    for (magnitude, freqs), ch_name, color in zip(fft_results, ['Left', 'Right'], ['blue', 'orange']):
        min_mag = ref_max_global * (10**(-100 / 20.0)) 
        magnitude_clipped = np.maximum(magnitude, min_mag)
        magnitude_db = 20 * np.log10(magnitude_clipped / ref_max_global)

        fig.add_trace(go.Scatter(
            x=freqs,
            y=magnitude_db,
            name=f'{ch_name} Channel',
            mode='lines',
            line=dict(color=color, width=1),
            hovertemplate='Freq: %{x:.1f} Hz<br>Level: %{y:.1f} dBFS<extra></extra>' 
        ))

    # --- Layout dan Tampilkan Plot ---
    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Level (dBFS)",
        yaxis_range=[-100, 5], # Beri ruang untuk melihat gain
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(type="log", rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)


def apply_balance(stereo, bal):
    # Pastikan input adalah stereo
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)
    
    if bal > 0: # Pan ke Kanan
        left_gain, right_gain = 1 - bal, 1
    else: # Pan ke Kiri
        left_gain, right_gain = 1, 1 + bal
    return np.stack([stereo[:, 0] * left_gain, stereo[:, 1] * right_gain], axis=1)


# --- Preview sebelum mixing ---
if file1:
    file1.seek(0)
    with st.spinner("Membaca Channel 1..."):
        data1_prev, sr1_prev = sf.read(file1)
        st.subheader("🎧 Preview Channel 1")
        st.audio(file1)
        plot_waveform(data1_prev, sr1_prev, "Waveform Channel 1")
        plot_spectrum(data1_prev, sr1_prev, "Spektrum Channel 1")

if file2:
    file2.seek(0)
    with st.spinner("Membaca Channel 2..."):
        data2_prev, sr2_prev = sf.read(file2)
        st.subheader("🎧 Preview Channel 2")
        st.audio(file2)
        plot_waveform(data2_prev, sr2_prev, "Waveform Channel 2")
        plot_spectrum(data2_prev, sr2_prev, "Spektrum Channel 2")

# --- Proses mixing ---
if process:
    if not file1 or not file2:
        st.error("⚠️ Mohon upload dua file audio terlebih dahulu!")
    else:
        st.subheader("🎶 Hasil Mixing & EQ")
        
        # Selalu baca ulang file untuk proses
        file1.seek(0)
        file2.seek(0)
        data1, sr1 = sf.read(file1)
        data2, sr2 = sf.read(file2)

        if sr1 != sr2:
            st.error("⚠️ Sample rate kedua file harus sama!")
        else:
            with st.spinner("Memproses mixing dan EQ..."):
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]

                # Pastikan kedua sinyal stereo sebelum di-proses
                if data1.ndim == 1:
                    data1 = np.stack([data1, data1], axis=1)
                if data2.ndim == 1:
                    data2 = np.stack([data2, data2], axis=1)

                # --- PROSES CHANNEL (Volume & Balance) ---
                gain1 = 10 ** (vol1 / 20)
                gain2 = 10 ** (vol2 / 20)
                
                data1_processed = apply_balance(data1 * gain1, balance1)
                data2_processed = apply_balance(data2 * gain2, balance2)

                # --- PROSES MIXER ---
                mixed = data1_processed + data2_processed

                # --- IMPLEMENTASI EQ (SESUAI REQUIREMENT LPF/BPF/HPF) ---
                low_cutoff = 250
                high_cutoff = 5000
                
                b_lpf, a_lpf = signal.butter(2, low_cutoff, btype='lowpass', fs=sr1)
                b_bpf, a_bpf = signal.butter(2, [low_cutoff, high_cutoff], btype='bandpass', fs=sr1)
                b_hpf, a_hpf = signal.butter(2, high_cutoff, btype='highpass', fs=sr1)

                signal_low = signal.lfilter(b_lpf, a_lpf, mixed, axis=0)
                signal_mid = signal.lfilter(b_bpf, a_bpf, mixed, axis=0)
                signal_high = signal.lfilter(b_hpf, a_hpf, mixed, axis=0)

                gain_bass = 10**(eq_bass / 20)
                gain_mid = 10**(eq_mid / 20)
                gain_treble = 10**(eq_treble / 20)

                final_output = (signal_low * gain_bass) + \
                               (signal_mid * gain_mid) + \
                               (signal_high * gain_treble)

                # --- AKHIR BLOK EQ ---

                # --- PERBAIKAN: HAPUS NORMALISASI OTOMATIS ---
                # Baris lama: final_output = eq_output / max_abs if max_abs > 1.0 else eq_output
                # Baris baru tidak ada. Kita biarkan apa adanya.
                
                # --- FITUR BARU: DETEKSI CLIPPING (Untuk Kriteria Analisis) ---
                max_abs = np.max(np.abs(final_output))
                if max_abs > 1.0:
                    st.error(f"🔥 PERINGATAN: CLIPPING TERDETEKSI! (Level Puncak: {max_abs:.2f})")
                    st.caption("Suara mungkin akan terdengar pecah (distorsi). Turunkan volume channel atau gain EQ.")
                    # Untuk mencegah audio yang sangat merusak, kita bisa 'clip' (jepit) nilainya
                    final_output_clipped = np.clip(final_output, -1.0, 1.0)
                else:
                    st.success("✅ Proses selesai! Level audio aman.")
                    final_output_clipped = final_output

                # Simpan file yang sudah di-clip (jika perlu) untuk didengarkan
                sf.write("mixed_output.wav", final_output_clipped, sr1)
                st.audio("mixed_output.wav", format="audio/wav")

                # Tapi, plot waveform dan spektrum dari sinyal ASLI (sebelum di-clip)
                # Ini PENTING untuk kriteria Analisis!
                plot_waveform(final_output, sr1, "Waveform Output (Before Clipping)")
                plot_spectrum(final_output, sr1, "Spektrum Output (Showing True Gain)")
                
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
