import modal
import uuid

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

def download_model():
    try: 
        import nemo.collections.asr as nemo_asr # type: ignore
        nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    except ImportError:
        pass

asr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "librosa",
        "omegaconf",
        "lightning",
        "cuda-python>=12.3",
        "git+https://github.com/NVIDIA/multi-storage-client.git",
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo@main",
        extra_options="-U",
        gpu="A10G",
    )
    .run_function(
        download_model,
        gpu="A10G",
    )
)

with asr_image.imports():
    import nemo.collections.asr as nemo_asr # type: ignore
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig # type: ignore
    from nemo.collections.asr.parts.utils.streaming_utils import BatchedFrameASRTDT # type: ignore
    from nemo.collections.asr.parts.utils.transcribe_utils import get_buffered_pred_feat_rnnt # type: ignore
    import math
    import torch # type: ignore
    from omegaconf import OmegaConf # type: ignore
    import librosa # type: ignore
    import os

app = modal.App(name="clipscript-asr-service")

# This must be the same volume object used in processing.py
upload_volume = modal.Volume.from_name(
    "clipscript-uploads", create_if_missing=True
)

@app.cls(
    image=asr_image,
    gpu="A10G",
    scaledown_window=600,
    volumes={"/data": upload_volume},  # Mount the shared volume
)
class ASR:
    @modal.enter()
    def startup(self):
        print("loading model...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        print("model loaded.")
        
        self.model.freeze()
        torch.set_grad_enabled(False)
        
        # Configure for buffered inference
        model_cfg = self.model._cfg
        OmegaConf.set_struct(model_cfg.preprocessor, False)
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0
        OmegaConf.set_struct(model_cfg.preprocessor, True)
        
        # Setup decoding for TDT model
        decoding_cfg = RNNTDecodingConfig()
        decoding_cfg.strategy = "greedy"  # TDT requires greedy
        decoding_cfg.preserve_alignments = True
        decoding_cfg.fused_batch_size = -1
        
        if hasattr(self.model, 'change_decoding_strategy'):
            self.model.change_decoding_strategy(decoding_cfg)
        
        # Calculate timing parameters
        self.feature_stride = model_cfg.preprocessor['window_stride']
        self.model_stride = 4  # TDT model stride
        self.model_stride_in_secs = self.feature_stride * self.model_stride
        
        # Buffered inference parameters
        self.chunk_len_in_secs = 15.0
        self.total_buffer_in_secs = 20.0
        self.batch_size = 64 
        self.max_steps_per_timestep = 15
        
        # Calculate chunk parameters
        self.tokens_per_chunk = math.ceil(self.chunk_len_in_secs / self.model_stride_in_secs)
        
        print("ASR setup complete with buffered inference support.")

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            # Fallback: estimate from file size (rough approximation)
            file_size = os.path.getsize(audio_path)
            # Rough estimate: 16kHz, 16-bit mono = ~32KB per second
            return file_size / 32000
    
    def _simple_transcribe(self, audio_path: str) -> str:
        print("Using simple transcription...")
        output = self.model.transcribe([audio_path])
        
        if not output or not hasattr(output[0], "text"):
            return ""
        
        return output[0].text
    
    def _buffered_transcribe(self, audio_path: str) -> str:
        print("Using buffered transcription...")
        
        # Setup TDT frame processor
        frame_asr = BatchedFrameASRTDT(
            asr_model=self.model,
            frame_len=self.chunk_len_in_secs,
            total_buffer=self.total_buffer_in_secs,
            batch_size=self.batch_size,
            max_steps_per_timestep=self.max_steps_per_timestep,
            stateful_decoding=False,
        )
        
        # Calculate delay for TDT
        mid_delay = math.ceil((self.chunk_len_in_secs + (self.total_buffer_in_secs - self.chunk_len_in_secs) / 2) / self.model_stride_in_secs)
        
        # Process with buffered inference
        hyps = get_buffered_pred_feat_rnnt(
            asr=frame_asr,
            tokens_per_chunk=self.tokens_per_chunk,
            delay=mid_delay,
            model_stride_in_secs=self.model_stride_in_secs,
            batch_size=self.batch_size,
            manifest=None,
            filepaths=[audio_path],
            accelerator='gpu',
        )
        
        # Extract transcription
        if hyps and len(hyps) > 0:
            return hyps[0].text
        
        return ""

    @modal.method()
    def transcribe(self, audio_filename: str = None, audio_bytes: bytes = None, use_buffered: bool | None = None) -> dict[str, str]:
        audio_path = None
        temp_audio_path = None
        try:
            if audio_filename:
                audio_path = f"/data/{audio_filename}"
            elif audio_bytes:
                # When bytes are passed, they must be written to a file for librosa/nemo to read.
                temp_audio_path = f"/tmp/input_{uuid.uuid4()}.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)
                audio_path = temp_audio_path
            else:
                raise ValueError("Either 'audio_filename' or 'audio_bytes' must be provided.")
            
            if not os.path.exists(audio_path):
                return {"text": "", "error": f"Audio file not found at path: {audio_path}"}
            
            # Determine transcription method
            if use_buffered is None:
                duration = self._get_audio_duration(audio_path)
                use_buffered = duration > 1800.0  # 30 minutes
                print(f"Audio duration: {duration:.1f}s, using {'buffered' if use_buffered else 'simple'} transcription")
            
            if use_buffered:
                text = self._buffered_transcribe(audio_path)
            else:
                text = self._simple_transcribe(audio_path)
            
            print("transcription complete.")
            return {"text": text, "error": ""}
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "error": str(e)}
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    @modal.method()
    def transcribe_simple(self, audio_filename: str = None, audio_bytes: bytes = None) -> dict[str, str]:
        """Force simple transcription (for compatibility)"""
        return self.transcribe(audio_filename=audio_filename, audio_bytes=audio_bytes, use_buffered=False)
    
    @modal.method()
    def transcribe_buffered(self, audio_filename: str = None, audio_bytes: bytes = None) -> dict[str, str]:
        """Force buffered transcription"""
        return self.transcribe(audio_filename=audio_filename, audio_bytes=audio_bytes, use_buffered=True)