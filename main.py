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
import re

load_dotenv()

asr = modal.Cls.from_name("clipscript-asr-service", "ASR")
llm = "deepseek/deepseek-r1-0528:free"
api_key = os.environ.get("OPENROUTER_API_KEY")

def extract_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_thumbnail_url(video_id: str) -> str:
    """Get the high quality thumbnail URL for a YouTube video."""
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

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
    gr.Info("Starting transcription... This might take a moment.")
    if file is not None:
        return process_audio(file)
    elif url and url.strip():
        return process_audio(url)
    else:
        gr.Warning("Please upload a file or enter a URL.")
        return "Error: Please upload a file or enter a URL."

def add_transcript_to_chat(transcript: str):
    if transcript.startswith("Error"):
        gr.Error("Transcription failed. Please check the logs.")
        return []
    gr.Info("Transcript ready! Generating blog post...")
    # Return empty list for display but store transcript for LLM processing
    return []

def user_chat(user_message: str, history: list):
    return "", history + [{"role": "user", "content": user_message}]

def _stream_chat_response(history: list, system_prompt: str, transcript: str = None):
    if not history and not transcript:
        # Don't do anything if there's no history and no transcript
        return

    # Include transcript as first user message if provided, but don't display it
    messages = [{"role": "system", "content": system_prompt}]
    if transcript:
        messages.append({"role": "user", "content": transcript})
    messages.extend(history)
    
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

def generate_blog_post(history: list, transcript: str):
    system_prompt = """You are an expert blog writer and editor. Your task is to transform a raw video transcription into a well-structured, engaging, and publish-ready blog post in Markdown format.
Core Mandate: Erase the Video Origin
This is a critical function. The reader must not know the content came from a video.
Eliminate all video-specific language: Remove phrases like "in this video," "thanks for watching," "as you can see here," "welcome to the channel," etc.
Scrub all platform calls-to-action: No "like and subscribe," "hit the bell icon," or "comment below."
Remove sponsor messages and ads: Completely omit any sponsor mentions.
Rephrase visual references: Convert "look at this screen" to a description of the information itself (e.g., "The data reveals that...").
Content & Formatting Rules:
Title: Create a compelling, SEO-friendly H1 title.
Structure: Use ## for main headings and ### for subheadings to create a logical flow.
Readability: Use short paragraphs, bulleted/numbered lists, and bolding for key terms.
Refine Prose: Convert conversational speech into clean, professional writing.
Remove all filler words (um, uh, like, you know).
Fix grammar and consolidate rambling sentences.
Flow: Start with a strong introduction and end with a concise summary or conclusion.
Your output must be a complete, polished article in Markdown."""
    yield from _stream_chat_response(history, system_prompt, transcript)
    
def bot_chat(history: list):
    system_prompt = "You are a helpful assistant that helps refine a blog post created from an audio transcript. The user will provide instructions for changes and you will return only the updated blog post."
    yield from _stream_chat_response(history, system_prompt)

def update_thumbnail_display(url: str):
    """Update the thumbnail display when YouTube URL is entered."""
    if not url or not url.strip():
        return gr.update(visible=False, value=None)
    
    video_id = extract_youtube_video_id(url)
    if video_id:
        thumbnail_url = get_youtube_thumbnail_url(video_id)
        return gr.update(visible=True, value=thumbnail_url)
    else:
        return gr.update(visible=False, value=None)

# Gradio Interface
theme = gr.themes.Ocean()

with gr.Blocks(title="ClipScript", theme=theme) as demo:
    gr.Markdown("# Turn your videos into blogs")

    gr.Markdown("Upload an audio file, or provide a YouTube/direct URL.")
    with gr.Row():
        file_input = gr.File(label="Upload any audio file", type="filepath", scale=3, height=300)
        
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="YouTube or Direct Audio URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            # YouTube thumbnail display
            thumbnail_display = gr.Image(
                label="Video Thumbnail",
                visible=False,
                height=200,
                show_download_button=False,
                interactive=False
            )

    transcribe_button = gr.Button("Blogify", variant="primary")

    gr.Markdown("---")

    with gr.Accordion("View Transcript", open=False):
        transcript_output = gr.Textbox(label="Transcript", lines=15, interactive=False)

    chatbot = gr.Chatbot(
        label="Blog Post", type="messages", height=500
    )
    chat_input = gr.Textbox(
        label="Your message",
        placeholder="Refine the blog post or ask for changes...",
        container=False,
    )
    clear_button = gr.ClearButton([chat_input, chatbot])


    # Event handlers to disable/enable inputs based on usage
    def on_file_upload(file):
        if file is not None:
            return gr.update(interactive=False), gr.update(visible=False, value=None)
        else:
            return gr.update(interactive=True), gr.update(visible=False, value=None)

    def on_url_change(url):
        if url and url.strip():
            thumbnail_update = update_thumbnail_display(url)
            return gr.update(interactive=False), thumbnail_update
        else:
            return gr.update(interactive=True), gr.update(visible=False, value=None)

    file_input.change(fn=on_file_upload, inputs=file_input, outputs=[url_input, thumbnail_display])
    url_input.change(fn=on_url_change, inputs=url_input, outputs=[file_input, thumbnail_display])

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
        .then(fn=generate_blog_post, inputs=[chatbot, transcript_output], outputs=chatbot)
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