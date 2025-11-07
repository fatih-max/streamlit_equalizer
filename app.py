import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from scipy import signal

st.title("🎚️ Software-Defined Audio Mixer dan Equalizer 👌")

# === Sidebar ===
st.sidebar.header("🎛️ Kontrol Mixer")
st.sidebar.subheader("Channel 1")
vol1 = st.sidebar.slider("Volume (dB)", -60.0, 6.0, 0.0, key="vol1")

uploaded_file = st.file_uploader("Unggah file audio (WAV)", type=["wav"])
if uploaded_file:
    data, fs = sf.read(uploaded_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Aplikasi volume
    gain = 10 ** (vol1 / 20)
    data = data * gain

    # Tombol Peak Search
    peak_search = st.button("🔍 Peak Search")

    # Hitung spektrum
    freqs, Pxx = signal.welch(data, fs=fs, nperseg=1024)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=10*np.log10(Pxx), mode="lines", name="Spektrum"))

    if peak_search:
        peak_idx = np.argmax(Pxx)
        peak_freq = freqs[peak_idx]
        peak_power = 10 * np.log10(Pxx[peak_idx])
        fig.add_trace(go.Scatter(
            x=[peak_freq],
            y=[peak_power],
            mode="markers+text",
            name="Peak",
            text=[f"{peak_freq:.1f} Hz"],
            textposition="top center",
            marker=dict(color="red", size=10)
        ))
        st.success(f"🔺 Peak found at **{peak_freq:.2f} Hz**")

    fig.update_layout(
        title="Spektrum Audio (Welch PSD)",
        xaxis_title="Frekuensi (Hz)",
        yaxis_title="Daya (dB)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
