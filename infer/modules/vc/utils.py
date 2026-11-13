import os
import ffmpeg
import subprocess
import random
import gradio as gr

from infer.lib.utils import load_audio_infer, load_embedding

from pytube import YouTube

def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )

    
def load_hubert(config):
  //  device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder_model = "contentvec"
    self.hubert_model = load_embedding(embedder_model, embedder_model_custom=None)
    self.hubert_model = self.hubert_model.float().to(config.device)
    
    return hubert_model.eval()

async def download_and_split_audio(url, model):
    audio = YouTube(url).streams.get_audio_only()
    audio.download(filename="audio.wav", output_path="output/yt", skip_existing=False)
    command = f"demucs --two-stems=vocals -n {model} output/yt/audio.wav -o output"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    vocal = f"output/{model}/audio/vocals.wav"
    inst = f"output/{model}/audio/no_vocals.wav"
    gr.Info("Successfully download and split the audio!")
    return vocal, vocal, result.stdout

async def combine_audio(model, vocal):
    inst = f"output/{model}/audio/no_vocals.wav"
    os.makedirs(os.path.join("output", "combined"), exist_ok=True)
    random_number = random.randint(0, 1000000)
    output_path = os.path.join("output", "combined", f"combined_{random_number}.wav")
    while output_path in os.listdir(os.path.join("output", "combined")):
        random_number = random.randint(0, 1000000)
        output_path = os.path.join("output", "combined", f"combined_{random_number}.wav")
    command = f"ffmpeg -i {inst} -i {vocal} -filter_complex amix=inputs=2:duration=longest -vn {output_path}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    gr.Info("Successfully combine the audio!")
    return output_path, result.stdout
