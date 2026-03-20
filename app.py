import os
import numpy as np
from dash import Dash, html, dcc, Input, Output

from utils.audio_utils import (
    load_audio,
    resample_audio,
    normalize_audio,
    add_noise,
    audio_to_base64,
)
from utils.spectrogram_utils import create_spectrogram
from utils.aic_utils import AICEnhancer

# Create Dash app FIRST
app = Dash(__name__)
server = app.server

# Initialize SDK
enhancer = AICEnhancer()

# Load clean audio
clean_audio, sr = load_audio("data/clean.wav")
clean_audio = resample_audio(clean_audio, sr, enhancer.sample_rate)
clean_audio = normalize_audio(clean_audio)
sr = enhancer.sample_rate

# Precompute original displays
orig_spec = create_spectrogram(clean_audio, sr, "Original")
orig_audio = audio_to_base64(clean_audio, sr)

app.layout = html.Div(
    [
        html.H2("AIC Speech Enhancement Dashboard"),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Noise Level (dBFS)"),
                        dcc.Slider(
                            id="noise-slider",
                            min=-80,
                            max=0,
                            step=5,
                            value=-40,
                            marks={i: str(i) for i in range(-80, 1, 20)},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label("Enhancement Level"),
                        dcc.Slider(
                            id="enhance-slider",
                            min=0.0,
                            max=1.0,
                            step=0.1,
                            value=1.0,
                            marks={0: "0.0", 0.5: "0.5", 1: "1.0"},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    id="vad-output",
                    style={"fontWeight": "bold", "marginBottom": "20px"},
                ),
            ]
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H4("Original"),
                        dcc.Graph(figure=orig_spec),
                        html.Audio(src=orig_audio, controls=True, style={"width": "100%"}),
                    ],
                    style={"width": "32%"},
                ),
                html.Div(
                    [
                        html.H4("Noisy"),
                        dcc.Graph(id="noisy-spec"),
                        html.Audio(id="noisy-audio", controls=True, style={"width": "100%"}),
                    ],
                    style={"width": "32%"},
                ),
                html.Div(
                    [
                        html.H4("Enhanced"),
                        dcc.Graph(id="enhanced-spec"),
                        html.Audio(id="enhanced-audio", controls=True, style={"width": "100%"}),
                    ],
                    style={"width": "32%"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "16px",
                "flexWrap": "wrap",
            },
        ),
    ],
    style={"padding": "20px", "fontFamily": "Arial"},
)

@app.callback(
    Output("noisy-spec", "figure"),
    Output("enhanced-spec", "figure"),
    Output("noisy-audio", "src"),
    Output("enhanced-audio", "src"),
    Output("vad-output", "children"),
    Input("noise-slider", "value"),
    Input("enhance-slider", "value"),
)
def update_dashboard(noise_db, enhance_level):
    noisy = add_noise(clean_audio, noise_db)

    enhancer.set_enhancement_level(enhance_level)
    enhanced = enhancer.enhance(noisy)
    enhanced = np.clip(enhanced, -1.0, 1.0)

    vad_text = f"VAD Speech Detected: {enhancer.speech_detected()}"

    return (
        create_spectrogram(noisy, sr, "Noisy"),
        create_spectrogram(enhanced, sr, "Enhanced"),
        audio_to_base64(noisy, sr),
        audio_to_base64(enhanced, sr),
        vad_text,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
