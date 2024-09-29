# pip install -U openai-whisper
# pip install diffusers
# pip install pydub

import whisper
import torch
from transformers import pipeline
from datasets import load_dataset
from diffusers import DiffusionPipeline

from pydub import AudioSegment

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast



model_translate = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer_translate = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

m4a_file = "/content/Recording_2.m4a"

audio_export = AudioSegment.from_file(m4a_file, format="m4a")

# Export as MP3
mp3_file = "/content/recording_random_spanish_audio_final.mp3"
audio_export.export(mp3_file, format="mp3")

print(f"Conversion complete. Saved as {mp3_file}")

model = whisper.load_model("small")
audio = whisper.load_audio(mp3_file)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large",
  chunk_length_s=30,
  device=device,
)
prediction = pipe(audio.copy(), batch_size=8)["text"]

tokenizer_translate.src_lang = "es_XX"
encoded_hi = tokenizer_translate(prediction, return_tensors="pt")
generated_tokens = model_translate.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer_translate.lang_code_to_id["en_XX"]
)
sentence_en = tokenizer_translate.batch_decode(generated_tokens, skip_special_tokens=True)

prompt = sentence_en

pipe = DiffusionPipeline.from_pretrained("THUDM/CogVideoX-5b")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

image = pipe(prompt).images[0]
video_path = export_to_video(video_frames)
