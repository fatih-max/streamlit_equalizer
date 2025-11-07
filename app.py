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

st.sidebar.header("🎼 Master EQ")
eq_bass = st.sidebar.slider("Bass Gain (dB)", -12.0, 12.0, 0.0, key="eq_bass")
eq_mid = st.sidebar.slider("Mid Gain (dB)", -12.0, 12.0, 0.0, key="eq_mid")
eq_treble = st.sidebar.slider("Treble Gain (dB)", -12.0, 12.0, 0.0, key="eq_treble")

st.sidebar.subheader("🎵 Upload Audio")
file1 = st.sidebar.file_uploader("Channel 1 (.wav)", type=["wav"], key="file1")
file2 = st.sidebar.file_uploader("Channel 2 (.wav)", type=["wav"], key="file2")

process = st.sidebar.button("🔊 Proses Mixing")


# --- Fungsi bantu (Plot diganti ke Plotly) ---
def plot_waveform(data, sr, title):
    """Fungsi plot waveform interaktif menggunakan Plotly.
    Menambahkan rangeslider dan dragmode zoom agar mudah melakukan zoom.
    Untuk performa, data yang diplot bisa di-downsample saat terlalu panjang.
    """
    fig = go.Figure()
    duration = len(data) / sr
    time = np.linspace(0, duration, len(data))

    # Downsample tampilan jika terlalu banyak titik
    max_points = 10000
    step = max(1, len(time) // max_points)
    time_plot = time[::step]

    if data.ndim == 1:
        y_plot = data[::step]
        fig.add_trace(
            go.Scatter(
                x=time_plot,
                y=y_plot,
                name='Mono',
                line=dict(color='dodgerblue', width=1),
                hovertemplate='Time: %{x:.4f}s<br>Amplitude: %{y:.6f}<extra></extra>',
            )
        )
    else:
        left_plot = data[:, 0][::step]
        right_plot = data[:, 1][::step]
        fig.add_trace(
            go.Scatter(
                x=time_plot,
                y=left_plot,
                name='Left',
                line=dict(color='blue', width=1),
                hovertemplate='Time: %{x:.4f}s<br>Left: %{y:.6f}<extra></extra>',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_plot,
                y=right_plot,
                name='Right',
                line=dict(color='orange', width=1),
                hovertemplate='Time: %{x:.4f}s<br>Right: %{y:.6f}<extra></extra>',
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='zoom'  # default drag mode untuk zooming
    )
    # Tambahkan rangeslider untuk mempermudah navigasi / zoom pada sumbu waktu
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)


def plot_spectrum(data, sr, title):
    """Fungsi plot spektrum interaktif menggunakan Plotly (diubah ke dBFS).
    Menambahkan marker dan hovertemplate, serta tabel preview nilai frekuensi/puncak.
    """
    fig = go.Figure()

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Hapus DC offset kecil agar spektrum lebih rapi
    data = data - np.mean(data)

    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    magnitude = np.abs(fft)

    if np.max(magnitude) > 0:
        magnitude_normalized = magnitude / np.max(magnitude)
        magnitude_db = 20 * np.log10(magnitude_normalized + 1e-12)
    else:
        magnitude_db = np.full_like(magnitude, -200.0)

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=magnitude_db,
            name='Spectrum',
            mode='lines+markers',
            marker=dict(size=4, color='purple'),
            line=dict(color='purple'),
            hovertemplate='Freq: %{x:.1f} Hz<br>Level: %{y:.2f} dBFS<extra></extra>',
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Level (dBFS)",
        yaxis_type="linear",
        yaxis_range=[-100, 0],
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
    )
    # Tampilkan chart
    st.plotly_chart(fig, use_container_width=True)

    # --- Tampilkan preview nilai per "garis" spektrum (puncak-puncak utama) ---
    try:
        # Temukan puncak pada magnitudo (menggunakan nilai asli magnitude, bukan dB)
        prominence_threshold = np.max(magnitude) * 0.05 if np.max(magnitude) > 0 else 0.0
        peaks_idx, _ = signal.find_peaks(magnitude, prominence=prominence_threshold)
        if peaks_idx.size == 0:
            st.info("Tidak ditemukan puncak spektral yang menonjol untuk ditampilkan.")
            return

        # Ambil top N puncak berdasarkan magnitudo
        top_n = 20
        peaks_sorted = peaks_idx[np.argsort(magnitude[peaks_idx])][::-1][:top_n]
        peaks_list = []
        for p in peaks_sorted:
            peaks_list.append(
                {
                    "Frequency (Hz)": float(f"{freqs[p]:.1f}"),
                    "Level (dBFS)": float(f"{magnitude_db[p]:.2f}"),
                    "Magnitude": float(f"{magnitude[p]:.6e}")
                }
            )

        st.subheader("Preview Nilai Garis Spektrum (Puncak Utama)")
        st.table(peaks_list)
    except Exception as e:
        st.warning(f"Gagal menghitung preview spektrum: {e}")


def apply_balance(stereo, bal):
    """Fungsi apply_balance."""
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)

    if bal > 0:
        left_gain = 1 - bal
        right_gain = 1
    else:
        left_gain = 1
        right_gain = 1 + bal

    left = stereo[:, 0] * left_gain
    right = stereo[:, 1] * right_gain

    return np.stack([left, right], axis=1)


def design_filter(gain_db, cutoff, sr, filter_type, q=1.0):
    """Mendesain koefisien filter IIR (b, a) untuk EQ."""
    if gain_db == 0:
        return np.array([1]), np.array([1])

    A = 10**(gain_db / 40)
    w0 = 2 * np.pi * cutoff / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    if filter_type == 'low_shelf':
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    elif filter_type == 'peaking':
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif filter_type == 'high_shelf':
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:
        return np.array([1]), np.array([1])

    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


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

            gain1 = 10 ** (vol1 / 20)
            gain2 = 10 ** (vol2 / 20)

            data1 = data1 * gain1
            data2 = data2 * gain2

            data1 = apply_balance(data1, balance1)
            data2 = apply_balance(data2, balance2)

            mixed = data1 + data2

            # --- Proses EQ ---
            b_bass, a_bass = design_filter(eq_bass, 250, sr1, 'low_shelf')
            b_mid, a_mid = design_filter(eq_mid, 2000, sr1, 'peaking')
            b_treble, a_treble = design_filter(eq_treble, 5000, sr1, 'high_shelf')

            eq_output = signal.lfilter(b_bass, a_bass, mixed, axis=0)
            eq_output = signal.lfilter(b_mid, a_mid, eq_output, axis=0)
            eq_output = signal.lfilter(b_treble, a_treble, eq_output, axis=0)

            max_abs = np.max(np.abs(eq_output))
            if max_abs > 1.0:
                 final_output = eq_output / max_abs
            else:
                 final_output = eq_output

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
sr = st.number_input("Sample Rate", 8000, 48000, 44100)

if st.button("⚙️ Generate"):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if wave_type == "Sine":
        y = 0.5 * np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square":
        y = 0.5 * signal.square(2 * np.pi * freq * t)
    elif wave_type == "Triangle":
        y = 0.5 * signal.sawtooth(2 * np.pi * freq * t, 0.5)
    elif wave_type == "Sawtooth":
        y = 0.5 * signal.sawtooth(2 * np.pi * freq * t)
    else:
        y = 0.5 * np.random.uniform(-1, 1, size=len(t))

    sf.write("generated.wav", y, sr)
    st.success("✅ Audio berhasil dibuat!")
    st.audio("generated.wav", format="audio/wav")
    plot_waveform(y, sr, f"Waveform {wave_type} ({freq} Hz)")
    plot_spectrum(y, sr, f"Spektrum {wave_type}")

    with open("generated.wav", "rb") as f:
        st.download_button("⬇️ Download Generated Audio", f, file_name=f"{wave_type.lower()}_{int(freq)}Hz.wav")


# --- Copyright Section ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-style:italic;'>Copyright © 2025 2_D4_Telekomunikasi_A_kelompok_1_PDSK All rights reserved.</p>",
    unsafe_allow_html=True
)
