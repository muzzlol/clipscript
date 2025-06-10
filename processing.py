import modal
import os
import uuid

sandbox_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("yt-dlp")
)

app = modal.App(
    "clipscript-processing-service",
    secrets=[modal.Secret.from_name("youtube-cookies")]
)

asr_handle = modal.Cls.from_name("clipscript-asr-service", "ASR")

# A persistent, named volume to stage file uploads from the Gradio app.
upload_volume = modal.Volume.from_name(
    "clipscript-uploads", create_if_missing=True
)

@app.function(
    image=sandbox_image,
    secrets=[modal.Secret.from_name("youtube-cookies")],
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
    Securely processes media from a URL or a file from the upload Volume.

    This function orchestrates a Sandbox to perform the download and conversion,
    then passes the result to the ASR service.
    """
    output_filename = f"processed-{uuid.uuid4()}.wav"
    # The sandbox mounts the volume at /data, so we write the output there.
    output_wav_path_in_sandbox = f"/data/{output_filename}"
    cookie_path_in_sandbox = "/tmp/cookies.txt"

    sb = None
    try:
        sb = modal.Sandbox.create(
            image=sandbox_image,
            volumes={"/data": upload_volume},
        )
        
        # Check if the secret env var is present in this function's environment.
        cookie_data = os.environ.get("YOUTUBE_COOKIES")
        use_cookies = False
        if cookie_data:
            print("Found youtube-cookies secret, writing it to the sandbox.")
            # Use the sandbox's filesystem API to write the cookie data to a file inside it.
            with sb.open(cookie_path_in_sandbox, "w") as f:
                f.write(cookie_data)
            use_cookies = True

        if url:
            print(f"Sandbox: Downloading and converting from URL: {url}")
            # Use yt-dlp to extract audio and convert to WAV in one command.
            cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--output", output_wav_path_in_sandbox,
            ]
            
            if use_cookies:
                cmd.extend(["--cookies", cookie_path_in_sandbox])

            cmd.extend(["--", url])
            
            p = sb.exec(*cmd)
            p.wait()

            if p.returncode != 0:
                stderr = p.stderr.read()
                raise RuntimeError(f"yt-dlp execution failed: {stderr}")

        elif upload_id:
            print(f"Sandbox: Converting uploaded file: {upload_id}")
            # The uploaded file is at /data/{upload_id} inside the sandbox.
            uploaded_file_path = f"/data/{upload_id}"
            cmd = [
                'ffmpeg', '-i', uploaded_file_path,
                '-ar', '16000', '-ac', '1', '-y', output_wav_path_in_sandbox
            ]
            p = sb.exec(*cmd)
            p.wait()

            if p.returncode != 0:
                stderr = p.stderr.read()
                raise RuntimeError(f"ffmpeg execution failed: {stderr}")
        else:
            raise ValueError("Either 'url' or 'upload_id' must be provided.")

        print("Sandbox: Process complete. WAV data written to shared volume.")

    finally:
        if sb:
            print("Terminating sandbox.")
            sb.terminate()

    print(f"Sending filename '{output_filename}' to ASR service.")
    
    # Retry ASR service call with exponential backoff
    max_asr_retries = 3
    for attempt in range(max_asr_retries):
        try:
            # Pass the filename (string) NOT the audio bytes.
            # Note: it's asr_handle.transcribe, not asr_handle().transcribe
            result = asr_handle.transcribe.remote(output_filename)
            break
        except Exception as e:
            if attempt == max_asr_retries - 1:
                raise e
            wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"ASR service attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            import time
            time.sleep(wait_time)

    # Clean up the processed WAV file as well
    try:
        print(f"Cleaning up {output_filename} from volume.")
        upload_volume.remove_file(output_filename)
        upload_volume.commit()
    except Exception as e:
        print(f"Warning: Failed to clean up {output_filename} from volume: {e}")

    return result 