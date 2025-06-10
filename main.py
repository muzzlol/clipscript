from functools import wraps
import logging
import gradio as gr
import os
import modal
from openai import OpenAI
from dotenv import load_dotenv
import re
import time
import uuid
import yt_dlp
import tempfile
import shutil
from pathlib import Path

load_dotenv()


process_media_remotely = modal.Function.from_name("clipscript-processing-service", "process_media")
asr_handle = modal.Cls.from_name("clipscript-asr-service", "ASR")
upload_volume = modal.Volume.from_name("clipscript-uploads", create_if_missing=True)


llm = "deepseek/deepseek-r1-0528:free"
api_key = os.environ.get("OPENROUTER_API_KEY")


def retry_on_rate_limit(max_retries: int = 5, base_delay: float = 2.0):
    """Decorator for exponential backoff on rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for 429 status code in different ways
                    status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                    if status_code == 429 or '429' in str(e) or 'rate limit' in str(e).lower():
                        logging.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
            raise Exception("Max retries exceeded due to rate limits or other persistent errors.")
        return wrapper
    return decorator


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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def download_and_convert_youtube_audio(url: str) -> str:
    """
    Downloads audio from a YouTube URL and converts it to a 16kHz mono WAV file.
    Uses a temporary directory for all intermediate files, ensuring cleanup.
    Returns the path to the final temporary WAV file.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        output_tmpl = os.path.join(temp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_tmpl,
            "postprocessors": [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': {
                'extractaudio': ['-ar', '16000', '-ac', '1']
            },
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded .wav file
        downloaded_files = list(Path(temp_dir).glob("*.wav"))
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp failed to create a WAV file. The video might be protected or unavailable.")

        # Move the final file to a new temporary location so we can clean up the directory
        source_path = downloaded_files[0]
        fd, dest_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        shutil.move(source_path, dest_path)
        
        return dest_path
    finally:
        shutil.rmtree(temp_dir)

def handle_transcription(file, url):
    if not file and not (url and url.strip()):
        gr.Warning("Please upload a file or enter a URL.")
        return "Error: Please upload a file or enter a URL."

    gr.Info("Starting secure transcription... This might take a moment.")
    
    try:
        result = None
        if url and url.strip():
            video_id = extract_youtube_video_id(url)
            if video_id:
                converted_wav_path = None
                try:
                    print(f"Detected YouTube URL. Processing locally: {url}")
                    converted_wav_path = download_and_convert_youtube_audio(url)
                    
                    # Read audio bytes and call ASR service
                    with open(converted_wav_path, "rb") as f:
                        audio_bytes = f.read()

                    print("Sending audio bytes to ASR service.")
                    result = asr_handle().transcribe.remote(audio_bytes=audio_bytes)
                finally:
                    # Clean up the final temp file
                    if converted_wav_path and os.path.exists(converted_wav_path):
                        os.remove(converted_wav_path)

            else:
                # Process other URLs remotely and securely.
                print(f"Sending URL to Modal for processing: {url}")
                result = process_media_remotely.remote(url=url)
        elif file is not None:
            # For file uploads:
            # 1. Generate a unique ID for the file.
            upload_id = f"upload-{uuid.uuid4()}"
            print(f"Uploading file to Modal volume with ID: {upload_id}")
            
            # 2. Upload the local file to the remote volume 
            with upload_volume.batch_upload() as batch:
                batch.put_file(file, upload_id)
            
            # 3. Trigger remote processing by passing the upload ID.
            print(f"Sending upload ID to Modal for processing: {upload_id}")
            result = process_media_remotely.remote(upload_id=upload_id)

        if result.get("error"):
            return f"Error from ASR service: {result['error']}"

        return result["text"]

    except Exception as e:
        print(f"An error occurred: {e}")
        # It's good practice to remove the local temp file if it exists
        if file and os.path.exists(file):
            os.remove(file)
        return f"Error: {str(e)}"
    finally:
        # Gradio's gr.File widget creates a temporary file. We should clean it up.
        if file and os.path.exists(file):
            os.remove(file)

def add_transcript_to_chat(transcript: str):
    if transcript.startswith("Error"):
        gr.Error("Transcription failed. Please check the logs.")
        return []
    gr.Info("Transcript ready! Generating blog post...")
    # Return empty list for display but store transcript for LLM processing
    return []

def user_chat(user_message: str, history: list):
    return "", history + [{"role": "user", "content": user_message}]

@retry_on_rate_limit(max_retries=3, base_delay=1.0)
def _stream_chat_response(history: list, system_prompt: str, transcript: str = None):
    if not history and not transcript:
        # Don't do anything if there's no history and no transcript
        return

    if transcript.startswith("Error"):
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

def generate_blog_post(history: list, transcript: str, context: str):
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
    
    # Combine transcript with additional context if provided
    full_transcript = transcript
    if context and context.strip():
        full_transcript = f"{transcript}\n\n--- Additional Context ---\n{context.strip()}\n\nThis is some additional context relevant to the transcription above."
    
    yield from _stream_chat_response(history, system_prompt, full_transcript)
    
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
    gr.Markdown("# 🎬➡️📝 ClipScript: Video-to-Blog Transformer", elem_classes="hero-title")

    gr.Markdown("### Upload an audio file, or provide a YouTube/direct URL *of any size*.")
    with gr.Row():
        # Column 1: File input, URL input, and thumbnail
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload any audio file", type="filepath", height=200, file_types=["audio", ".webm", ".mp3", ".mp4", ".m4a", ".ogg", ".wav"])
            
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="YouTube(Recommended) or Direct Audio URL",
                        placeholder="youtube.com/watch?v=... OR xyz.com/audio.mp3",
                        scale=2,
                        elem_classes="ellipsis-text"
                    )
            
                # YouTube thumbnail display
                thumbnail_display = gr.Image(
                    label="Thumbnail",
                    visible=False,
                    height=100,
                    show_download_button=False,
                    interactive=False,
                    scale=2
                )
        
        # Column 2: Transcript view
        with gr.Column(scale=2):
            transcript_output = gr.Textbox(label="Transcription POWERED by Modal Labs", lines=12, interactive=True, show_copy_button=True)

    transcribe_button = gr.Button("Blogify", variant="primary")

    gr.Markdown("---")

    # Add Context section
    context_input = gr.Textbox(
        label="Additional Context",
        placeholder="Enter any additional context, code, articles, or any references that relate to the video content...",
        lines=5,
        interactive=True
    )

    chatbot = gr.Chatbot(
        label="Blog Post", type="messages", height=500, show_copy_all_button=True, show_copy_button=True, show_share_button=True
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
            fn=lambda: gr.update(value=None, interactive=True),
            outputs=file_input,
            queue=False,
        )
        .then(
            fn=add_transcript_to_chat,
            inputs=transcript_output,
            outputs=chatbot,
            queue=False,
        )
        .then(fn=generate_blog_post, inputs=[chatbot, transcript_output, context_input], outputs=chatbot)
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