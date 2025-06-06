import modal

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

def download_model():
    try: 
        import nemo.collections.asr as nemo_asr # type: ignore
        nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    except ImportError:
        pass

asr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "nemo_toolkit[asr]",
        extra_options="-U",
        gpu="A10G",
    )
    .run_function(
        download_model,
        gpu="A10G",
    )
)

app = modal.App(name="clipscript-asr-service")

@app.cls(image=asr_image, gpu="A10G", concurrency_limit=20, container_idle_timeout=300)
class ASR:
    def __enter__(self):
        import nemo.collections.asr as nemo_asr # type: ignore
        print("loading model...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        print("model loaded.")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict[str, str]:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with open(tmp.name, "wb") as f:
                f.write(audio_bytes)

            print("transcribing...")
            output = self.model.transcribe([tmp.name])
        
        if not output or not hasattr(output[0], "text"):
            return {"text": "", "error": "Transcription failed."}

        print("transcription complete.")
        return {"text": output[0].text, "error": ""}