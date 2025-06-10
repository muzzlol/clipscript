import modal
import uuid

sandbox_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
)

app = modal.App(
    "clipscript-processing-service",
)

asr_handle = modal.Cls.from_name("clipscript-asr-service", "ASR")

# A persistent, named volume to stage file uploads from the Gradio app.
upload_volume = modal.Volume.from_name(
    "clipscript-uploads", create_if_missing=True
)

@app.function(
    image=sandbox_image,
    volumes={"/data": upload_volume},
    cpu=2.0,
    memory=4096,
    timeout=7200,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
def process_media(url: str = None, upload_id: str = None):
    """
    Securely processes media from a URL or a file from the upload Volume using a Sandbox.

    This function orchestrates a Sandbox to perform the download and conversion,
    then passes the resulting audio bytes to the ASR service.
    """
    output_filename = f"processed-{uuid.uuid4()}.wav"
    output_wav_path_in_sandbox = f"/tmp/{output_filename}"
    audio_bytes = None

    sb = None
    try:
        volumes = {"/data": upload_volume} if upload_id else {}
        
        sb = modal.Sandbox.create(
            image=sandbox_image,
            volumes=volumes,
        )
        
        cmd = []
        if url:
            print(f"Sandbox: Downloading and converting from non-YouTube URL: {url}")
            cmd = [
                'ffmpeg', '-i', url,
                '-ar', '16000', '-ac', '1', '-y', output_wav_path_in_sandbox
            ]
        elif upload_id:
            print(f"Sandbox: Converting uploaded file: {upload_id}")
            # Input path is on the mounted volume
            uploaded_file_path_in_sandbox = f"/data/{upload_id}"
            cmd = [
                'ffmpeg', '-i', uploaded_file_path_in_sandbox,
                '-ar', '16000', '-ac', '1', '-y', output_wav_path_in_sandbox
            ]
        else:
            raise ValueError("Either 'url' or 'upload_id' must be provided.")

        print("Sandbox: Executing FFMPEG...")
        p = sb.exec(*cmd)
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"ffmpeg execution failed: {stderr}")

        print("Sandbox: Process complete. Reading WAV data from sandbox's filesystem.")
        
        # Read the file directly from the sandbox's filesystem.
        with sb.open(output_wav_path_in_sandbox, "rb") as f:
            audio_bytes = f.read()

    except Exception as e:
        print(f"Error during sandbox processing: {e}")
        raise
    finally:
        if sb:
            print("Terminating sandbox.")
            sb.terminate()

    if not audio_bytes:
        raise RuntimeError("Processing failed to produce audio data.")

    # If we processed a user upload, we can now clean up the original file.
    if upload_id:
        try:
            print(f"Cleaning up original upload {upload_id} from volume.")
            upload_volume.remove_file(upload_id)
            upload_volume.commit()
        except Exception as e:
            # This is not a critical error, so we just warn.
            print(f"Warning: Failed to clean up {upload_id} from volume: {e}")

    print("Sending audio bytes to ASR service.")
    
    # Retry ASR service call with exponential backoff
    max_asr_retries = 3
    result = None
    for attempt in range(max_asr_retries):
        try:
            # Pass the audio bytes directly to the ASR service
            result = asr_handle.transcribe.remote(audio_bytes=audio_bytes)
            break
        except Exception as e:
            if attempt == max_asr_retries - 1:
                raise e
            wait_time = 2 ** attempt
            print(f"ASR service attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            import time
            time.sleep(wait_time)

    return result 