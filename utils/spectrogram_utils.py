import numpy as np
from scipy.signal import spectrogram
import plotly.graph_objects as go

def create_spectrogram(audio, sr, title=""):
    f, t, sxx = spectrogram(audio, sr, nperseg=512, noverlap=384)
    sxx_db = 10 * np.log10(sxx + 1e-10)

    fig = go.Figure(
        data=go.Heatmap(
            z=sxx_db,
            x=t,
            y=f,
            colorscale="Viridis",
            colorbar=dict(title="dB")
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig