import gradio as gr
import requests
import os
import tempfile
from pathlib import Path
import subprocess
from asr import ASR


class AudioTranscriber:
    def __init__(self):
        self.asr = ASR()

    def convert_to_wav(self, input_path: str) -> str:
        output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
        try:
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "-y",
                output_path
            ], check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Audio conversion failed: {e}")