# backend/server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
from moviepy.editor import VideoFileClip
from vosk import Model, KaldiRecognizer
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import re
import logging
from werkzeug.utils import secure_filename
import wave
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_dir = os.path.join(script_dir, '..', 'Server', 'chroma_db')
video_download_dir = os.path.join(script_dir, '..', 'Server', 'video')
vosk_model_path = os.path.join(script_dir, 'models', 'vosk-model-small-en-us-0.15')

# Ensure directories exist
os.makedirs(chroma_db_dir, exist_ok=True)
os.makedirs(video_download_dir, exist_ok=True)

# Initialize ChromaDB
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

# Initialize Vosk model
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"Vosk model not found at {vosk_model_path}. Please download and extract the model.")

vosk_model = Model(vosk_model_path)

# FUNCTIONS

def expand_youtube_url(short_url):
    """
    Expands a shortened YouTube URL to its full form.
    """
    match = re.search(r'youtu\.be/([^?]+)', short_url)
    if match:
        video_id = match.group(1)
        full_url = f'https://www.youtube.com/watch?v={video_id}'
        return full_url
    else:
        return short_url  # Return the original URL if it's not a short link

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video_clip.close()
        return True
    except Exception as e:
        app.logger.error(f"Error extracting audio: {e}", exc_info=True)
        return False

def transcribe_audio_vosk(audio_path):
    """
    Transcribes audio to text using Vosk.
    """
    try:
        wf = wave.open(audio_path, "rb")
        
        # Check if the audio file is in the correct format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
            app.logger.error("Audio file must be WAV format mono PCM.")
            wf.close()
            return None

        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)

        transcription = ""

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                result_json = json.loads(result)
                transcription += result_json.get("text", "") + " "

        # Get the final bits of audio
        final_result = rec.FinalResult()
        final_result_json = json.loads(final_result)
        transcription += final_result_json.get("text", "")

        wf.close()
        return transcription.strip()
    except Exception as e:
        app.logger.error(f"Error transcribing audio with Vosk: {e}", exc_info=True)
        return None

def transcribe_audio(audio_path):
    """
    Transcribes audio to text using Vosk.
    """
    return transcribe_audio_vosk(audio_path)

def store_transcription(video_title, transcription):
    """
    Stores the transcribed text in ChromaDB with embeddings.
    """
    try:
        embedding = embeddings.embed_query(transcription)
        vectorstore.add(
            documents=[transcription],
            embeddings=[embedding],
            metadatas=[{"video_title": video_title}],
            ids=[video_title]  # Ensure unique IDs or generate unique IDs
        )
        return True
    except Exception as e:
        app.logger.error(f"Error storing transcription: {e}", exc_info=True)
        return False

############################################################################################
# ENDPOINTS
############################################################################################

@app.route('/download', methods=['POST'])
def download_video():
    """
    Endpoint to download a YouTube video, extract its audio, transcribe the audio to text,
    and store the transcription in ChromaDB.
    """
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400

        # Expand shortened YouTube URLs
        video_url = expand_youtube_url(video_url)

        ydl_opts = {
            'format': 'best[ext=mp4]',  # Download the best available mp4 format
            'outtmpl': os.path.join(video_download_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,  # Suppress yt-dlp output
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

            # Extract video info to get the title and filename
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', None)
            video_filename = ydl.prepare_filename(info_dict)
            video_path = os.path.join(video_download_dir, os.path.basename(video_filename))

        # Extract audio from the downloaded video
        audio_filename = f"{video_title}.wav"
        audio_path = os.path.join(video_download_dir, audio_filename)
        success = extract_audio(video_path, audio_path)
        if not success:
            return jsonify({'error': 'Failed to extract audio'}), 500

        # Transcribe the extracted audio to text
        transcription = transcribe_audio(audio_path)
        if not transcription:
            return jsonify({'error': 'Failed to transcribe audio'}), 500

        # Store the transcription in ChromaDB
        stored = store_transcription(video_title, transcription)
        if not stored:
            return jsonify({'error': 'Failed to store transcription'}), 500

        # Optionally, remove audio and video files after processing to save space
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except Exception as e:
            app.logger.error(f"Error removing files: {e}", exc_info=True)

        return jsonify({'message': 'Video downloaded, processed, and transcription stored successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error downloading video: {e}", exc_info=True)
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

############################################################################################
# RUN SERVER
############################################################################################

if __name__ == '__main__':
    app.run(debug=True)
