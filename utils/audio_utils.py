import numpy as np
import soundfile as sf
import base64
import io
from scipy.signal import resample_poly
from math import gcd

def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)

def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)

def add_noise(audio, noise_dbfs):
    noise = np.random.randn(len(audio)).astype(np.float32)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-8)

    noise_amp = 10 ** (noise_dbfs / 20.0)
    noise = noise * noise_amp

    noisy = audio + noise
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)

def audio_to_base64(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"