import gradio as gr
import requests
import os
import tempfile
import subprocess
from pathlib import Path
import modal
import shutil

asr = modal.Cls.from_name("clipscript-asr-service", "ASR")

def process_audio(file_or_url: str):
    downloaded_path = None
    wav_path = None
    try:
        if "youtube.com" in file_or_url or "youtu.be" in file_or_url:
            print(f"Processing YouTube URL: {file_or_url}")
            input_path = download_youtube_audio(file_or_url)
            downloaded_path = input_path
        elif file_or_url.startswith(("http://", "https://")):
            print(f"Processing direct audio URL: {file_or_url}")
            input_path = download_audio_url(file_or_url)
            downloaded_path = input_path
        else:
            print(f"Processing uploaded file: {file_or_url}")
            input_path = file_or_url

        wav_path = convert_to_wav(input_path)

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        print("Sending audio to Modal for transcription...")
        result = asr().transcribe.remote(audio_bytes)

        if result.get("error"):
            return f"Error from ASR service: {result['error']}"

        return result["text"]

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {str(e)}"
    finally:
        if downloaded_path and os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

def download_youtube_audio(url: str) -> str:
    import yt_dlp

    temp_dir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(temp_dir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
            ],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded_files = list(Path(temp_dir).glob("*.wav"))
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp failed to create a WAV file.")

        source_path = downloaded_files[0]
        fd, dest_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        shutil.move(source_path, dest_path)

        return dest_path
    finally:
        shutil.rmtree(temp_dir)

def download_audio_url(url: str) -> str:
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return path

def convert_to_wav(input_path: str) -> str:
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd) 

    print(f"Converting {input_path} to {output_path}...")
    subprocess.run([
        'ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', '-y', output_path
    ], check=True, capture_output=True)
    return output_path

# Gradio Interface
with gr.Blocks(title="Audio Transcription", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Transcribe Audio with NVIDIA Parakeet on Modal")
    gr.Markdown("Upload an audio file, or provide a YouTube/direct URL. The audio will be processed on a remote GPU.")

    with gr.Tabs():
        with gr.TabItem("Upload Audio File"):
            file_input = gr.File(label="Upload any audio file", type="filepath")
            file_output = gr.Textbox(label="Transcription", lines=10)
            file_button = gr.Button("Transcribe File", variant="primary")
        
        with gr.TabItem("Transcribe from URL"):
            url_input = gr.Textbox(label="YouTube or Direct Audio URL", placeholder="https://www.youtube.com/watch?v=...")
            url_output = gr.Textbox(label="Transcription", lines=10)
            url_button = gr.Button("Transcribe from URL", variant="primary")

    file_button.click(fn=process_audio, inputs=file_input, outputs=file_output)
    url_button.click(fn=process_audio, inputs=url_input, outputs=url_output)

if __name__ == "__main__":
    demo.launch()