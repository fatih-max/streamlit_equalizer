def plot_spectrum(data, sr, title):
    """Plot spektrum interaktif + tombol Peak Search."""
    fig = go.Figure()

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data - np.mean(data)

    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    magnitude = np.abs(fft)
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

    # --- Tombol Peak Search ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{title}**")
    with col2:
        do_peak = st.button("🔍 Peak Search", key=title)

    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitude_db,
        name='Spectrum',
        mode='lines',
        line=dict(color='purple', width=1),
        hovertemplate='Freq: %{x:.1f} Hz<br>Level: %{y:.2f} dBFS<extra></extra>'
    ))

    # --- Jika tombol ditekan, tandai puncak ---
    if do_peak:
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
        st.success(f"🔺 Peak ditemukan di **{peak_freq:.2f} Hz** ({peak_level:.1f} dBFS)")

    fig.update_layout(
        title=None,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Level (dBFS)",
        yaxis_range=[-100, 0],
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
        hovermode='x unified'
    )

    fig.update_xaxes(type="log", rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)
