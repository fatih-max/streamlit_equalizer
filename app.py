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
eq1_freq = st.sidebar.slider("EQ Frequency (Hz)", 20, 20000, 1000, key="eq1_freq")
eq1_gain = st.sidebar.slider("EQ Gain (dB)", -12.0, 12.0, 0.0, key="eq1_gain")

st.sidebar.subheader("Channel 2")
vol2 = st.sidebar.slider("Volume (dB)", -60.0, 6.0, 0.0, key="vol2")
eq2_freq = st.sidebar.slider("EQ Frequency (Hz)", 20, 20000, 5000, key="eq2_freq")
eq2_gain = st.sidebar.slider("EQ Gain (dB)", -12.0, 12.0, 0.0, key="eq2_gain")

# --- File Upload ---
uploaded_file1 = st.sidebar.file_uploader("Upload Audio Channel 1 (WAV/FLAC)", type=["wav", "flac"], key="file1")
uploaded_file2 = st.sidebar.file_uploader("Upload Audio Channel 2 (WAV/FLAC)", type=["wav", "flac"], key="file2")


# --- Fungsi Filter EQ ---
def apply_eq(data, sr, freq, gain_db):
    b, a = signal.iirpeak(freq / (0.5 * sr), Q=2)
    eq = signal.lfilter(b, a, data)
    gain = 10 ** (gain_db / 20)
    return eq * gain


# --- Fungsi Plot Spektrum dengan Peak Search ---
def plot_spectrum(data, sr, title):
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data - np.mean(data)

    # FFT
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    magnitude = np.abs(fft)
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

    # Header dan tombol Peak Search
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{title}**")
    with col2:
        key_button = f"peak_search_{title}"
        if key_button not in st.session_state:
            st.session_state[key_button] = False
        if st.button("🔍 Peak Search", key=key_button + "_btn"):
            st.session_state[key_button] = not st.session_state[key_button]

    # Plot dasar
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitude_db,
        mode='lines',
        name='Spectrum',
        line=dict(color='purple', width=1),
        hovertemplate='Freq: %{x:.1f} Hz<br>Level: %{y:.2f} dBFS<extra></extra>'
    ))

    # Peak search aktif
    if st.session_state[key_button]:
        peak_idx = np.argmax(magnitude_db)
        peak_freq = freqs[peak_idx]
        peak_level = magnitude_db[peak_idx]

        fig.add_trace(go.Scatter(
            x=[peak_freq],
            y=[peak_level],
            mode='markers+text',
            text=[f"{peak_freq:.1f} Hz"],
            textposition="top center",
            name='Peak',
            marker=dict(color='red', size=10)
        ))

        st.success(f"🔺 Peak tertinggi: **{peak_freq:.2f} Hz** ({peak_level:.1f} dBFS)")

    # Layout
    fig.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Level (dBFS)",
        yaxis_range=[-100, 0],
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
        hovermode='x unified',
        template="plotly_white"
    )
    fig.update_xaxes(type="log", rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)


# --- Proses Audio ---
if uploaded_file1 is not None and uploaded_file2 is not None:
    data1, sr1 = sf.read(uploaded_file1)
    data2, sr2 = sf.read(uploaded_file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rate kedua file harus sama!")
    else:
        data1_eq = apply_eq(data1, sr1, eq1_freq, eq1_gain)
        data2_eq = apply_eq(data2, sr2, eq2_freq, eq2_gain)

        mix = (data1_eq * 10 ** (vol1 / 20)) + (data2_eq * 10 ** (vol2 / 20))
        mix = mix / np.max(np.abs(mix))

        st.audio(mix, sample_rate=sr1)

        st.divider()
        plot_spectrum(data1_eq, sr1, "Channel 1 Spectrum")
        plot_spectrum(data2_eq, sr2, "Channel 2 Spectrum")
        plot_spectrum(mix, sr1, "Mixed Output Spectrum")
else:
    st.info("📂 Upload dua file audio untuk memulai.")
