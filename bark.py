from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile

class BarkTTS:
    def __init__(self, model_path, voice_preset="v2/ru_speaker_5"):
        self.model_path = model_path
        self.voice_preset = voice_preset
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = BarkModel.from_pretrained(model_path)

    def text_to_speech(self, text, output_file="./data/final_audio.wav"):
        inputs = self.processor(text, voice_preset=self.voice_preset)
        audio_array = self.model.generate(**inputs).cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate

        scipy.io.wavfile.write(output_file, rate=sample_rate, data=audio_array)
        print(f"Audio saved to {output_file}")
