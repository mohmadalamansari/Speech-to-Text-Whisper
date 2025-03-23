import gradio as gr
from transformers import pipeline

# Load Whisper model with language setting
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Function to transcribe audio
def transcribe_audio(file_path):
    if not file_path:
        return "Error: No file uploaded."

    # Use correct input format and force English transcription
    result = stt_pipeline(file_path, generate_kwargs={"language": "en"})

    return result.get("text", "Transcription failed.")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Speech-to-Text Whisper UI")
    gr.Markdown("Upload an MP3 file to convert it into text using OpenAI's Whisper.")

    audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload MP3")
    transcribe_button = gr.Button("Transcribe")
    text_output = gr.Textbox(label="Transcribed Text", interactive=True)

    # Button click action
    transcribe_button.click(fn=transcribe_audio, inputs=audio_input, outputs=text_output)

# Launch UI
demo.launch(share=True)
