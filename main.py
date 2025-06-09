import gradio as gr
import requests
import os
import tempfile
import subprocess
from pathlib import Path
import modal
import shutil
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

asr = modal.Cls.from_name("clipscript-asr-service", "ASR")
llm = "deepseek/deepseek-r1-0528:free"
api_key = os.environ.get("OPENROUTER_API_KEY")

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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def handle_transcription(file, url):
    if file is not None:
        return process_audio(file)
    elif url and url.strip():
        return process_audio(url)
    else:
        return "Error: Please upload a file or enter a URL."

def add_transcript_to_chat(transcript: str):
    if transcript.startswith("Error"):
        return []
    return [{"role": "user", "content": transcript}]

def user_chat(user_message: str, history: list):
    return "", history + [{"role": "user", "content": user_message}]

def _stream_chat_response(history: list, system_prompt: str):
    if not history:
        # Don't do anything if there's no history (e.g. transcript error)
        return

    messages = [{"role": "system", "content": system_prompt}] + history
    
    stream = client.chat.completions.create(
        model=llm,
        messages=messages,
        stream=True
    )

    history.append({"role": "assistant", "content": ""})
    response_content = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += content
            history[-1]["content"] = response_content
            yield history

def generate_blog_post(history: list):
    system_prompt = "You are a helpful assistant that converts audio transcripts into blog posts. The user will provide a transcript, and you should turn it into a well-structured and engaging blog post."
    yield from _stream_chat_response(history, system_prompt)
    
def bot_chat(history: list):
    system_prompt = "You are a helpful assistant that helps refine a blog post created from an audio transcript. The user will provide instructions for changes."
    yield from _stream_chat_response(history, system_prompt)


# Gradio Interface
with gr.Blocks(title="ClipScript", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Turn your videos into blogs")

    gr.Markdown("Upload an audio file, or provide a YouTube/direct URL.")
    file_input = gr.File(label="Upload any audio file", type="filepath")
    gr.HTML("<div style='text-align: center;'><h2><strong>OR</strong></h2></div>")
    url_input = gr.Textbox(label="YouTube or Direct Audio URL", placeholder="https://www.youtube.com/watch?v=...")
    transcribe_button = gr.Button("Blogify", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=20, interactive=False)
        chatbot = gr.Chatbot(label="Blog Post", type="messages", height=500, bubble_full_width=False)

    with gr.Row():
        chat_input = gr.Textbox(label="Your message", placeholder="Refine the blog post or ask for changes...", container=False, scale=8)
        clear_button = gr.ClearButton([chat_input, chatbot], scale=1)

    # Event handlers to disable/enable inputs based on usage
    def on_file_upload(file):
        if file is not None:
            return gr.update(interactive=False) 
        else:
            return gr.update(interactive=True)  

    def on_url_change(url):
        if url and url.strip():
            return gr.update(interactive=False) 
        else:
            return gr.update(interactive=True)  

    file_input.change(fn=on_file_upload, inputs=file_input, outputs=url_input)
    url_input.change(fn=on_url_change, inputs=url_input, outputs=file_input)

    # Chained events for blog generation
    (
        transcribe_button.click(
            fn=handle_transcription,
            inputs=[file_input, url_input],
            outputs=transcript_output,
        )
        .then(
            fn=add_transcript_to_chat,
            inputs=transcript_output,
            outputs=chatbot,
            queue=False,
        )
        .then(fn=generate_blog_post, inputs=chatbot, outputs=chatbot)
    )

    # Event handler for follow-up chat
    chat_input.submit(
        fn=user_chat,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
        queue=False,
    ).then(fn=bot_chat, inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()