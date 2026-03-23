import os
import numpy as np
from dotenv import load_dotenv
import aic_sdk as aic
#from aic import Model, AICModelType, AICParameter
load_dotenv()

class AICEnhancer:
    def __init__(self):
        self.api_key = os.getenv("AIC_SDK_LICENSE") or os.getenv("AIC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing AIC_SDK_LICENSE in .env")

        print("Loading model...")
       

        model_path = os.path.join("models", "rook_l_16khz_dtv5nvgu_v20.aicmodel")

        #model_path = "/Users/nayanajacob/Desktop/AIC_Dash_App/models/quail_vf_2_0_l_16khz_d42jls1e_v18.aicmodel"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = aic.Model.from_file(model_path)

        self.config = aic.ProcessorConfig.optimal(
            self.model,
            num_channels=1
        )

        self.processor = aic.Processor(
            self.model,
            self.api_key,
            self.config
        )

        self.frame_size = self.config.num_frames
        self.sample_rate = self.model.get_optimal_sample_rate()

        self.context = self.processor.get_processor_context()
        self.vad = self.processor.get_vad_context()

    def set_enhancement_level(self, level: float):
        level = float(np.clip(level, 0.0, 1.0))
        self.context.set_parameter(aic.ProcessorParameter.EnhancementLevel, level)

    def set_bypass(self, enabled: bool):
        self.context.set_parameter(aic.ProcessorParameter.Bypass, enabled)

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        original_length = len(audio)

        remainder = original_length % self.frame_size
        if remainder != 0:
            pad = self.frame_size - remainder
            audio = np.pad(audio, (0, pad))

        enhanced_chunks = []

        for i in range(0, len(audio), self.frame_size):
            chunk = audio[i:i + self.frame_size]
            chunk = chunk.reshape(1, -1)   # shape = (channels, frames)

            out = self.processor.process(chunk)
            enhanced_chunks.append(out.flatten())

        enhanced_audio = np.concatenate(enhanced_chunks)
        return enhanced_audio[:original_length]

    def speech_detected(self) -> bool:
        return self.vad.is_speech_detected()
