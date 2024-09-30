from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
from moviepy.editor import VideoFileClip
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import re
import logging
from werkzeug.utils import secure_filename
import uuid
import speech_recognition as sr
from PIL import Image
import base64
import json
import ollama
import chromadb
import numpy as np
import cv2
from supabase import create_client, Client
from datetime import datetime
import asyncio
from transcribe_anything.api import transcribe
from transformers import pipeline
import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
buffer_dir = os.path.join(script_dir, 'buffer')
chroma_db_dir = os.path.join(buffer_dir, 'chroma_db')
video_download_dir = os.path.join(buffer_dir, 'video')
transcription_dir = os.path.join(buffer_dir, 'transcriptions')
frames_dir = os.path.join(buffer_dir, 'frames')

# Ensure directories exist
os.makedirs(chroma_db_dir, exist_ok=True)
os.makedirs(video_download_dir, exist_ok=True)
os.makedirs(transcription_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# Initialize ChromaDB
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

# Initialize Chroma DB client
chroma_client = chromadb.Client(Settings(persist_directory=chroma_db_dir))

# Create a collection in Chroma DB if it doesn't exist
collection_name = "image_descriptions"
try:
    collection = chroma_client.create_collection(collection_name)
    logging.debug(f"Created collection '{collection_name}' in ChromaDB.")
except Exception as e:
    collection = chroma_client.get_collection(collection_name)
    logging.debug(f"Collection '{collection_name}' already exists in ChromaDB.")

# Initialize Supabase client
supabase_url = 'https://dlmwlgnyehclzrryxepq.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsbXdsZ255ZWhjbHpycnl4ZXBxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzQ0OTQ0OCwiZXhwIjoyMDQzMDI1NDQ4fQ.b4plF-vw8ZJ5g-E84LWwUMF5OEzE-NBuG-9sI4sw8BE'
supabase: Client = create_client(supabase_url, supabase_key)

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
        logging.debug(f"Extracted audio to {audio_path}")
        return True
    except Exception as e:
        app.logger.error(f"Error extracting audio: {e}", exc_info=True)
        return False


def chunk_text(text, chunk_size=1000):
    """
    Chunks the text into smaller parts to fit within the context window.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def store_transcription(video_title, transcription):
    """
    Stores the transcribed text in ChromaDB with embeddings.
    """
    try:
        chunks = chunk_text(transcription)
        for i, chunk in enumerate(chunks):
            embedding = embeddings.embed_query(chunk)
            vectorstore.add_texts(
                texts=[chunk],
                metadatas=[{"video_title": video_title, "chunk_index": i}],
                ids=[f"{video_title}_{i}"]  # Ensure unique IDs or generate unique IDs
            )
        logging.debug("Stored transcription in ChromaDB")
        return True
    except Exception as e:
        app.logger.error(f"Error storing transcription: {e}", exc_info=True)
        return False

def read_captions(caption_path):
    """
    Reads captions from a file and returns the text.
    """
    try:
        with open(caption_path, 'r', encoding='utf-8') as file:
            captions = file.read()
        return captions
    except FileNotFoundError:
        app.logger.warning(f"Caption file not found: {caption_path}")
        return ""
    except Exception as e:
        app.logger.error(f"Error reading captions: {e}", exc_info=True)
        return ""

def save_transcription(video_uuid, transcription):
    """
    Saves the transcription to a temporary file.
    """
    print("Saving transcription")
    transcription_path = os.path.join(transcription_dir, f"{video_uuid}.txt")
    try:
        with open(transcription_path, 'w', encoding='utf-8') as file:
            file.write(transcription)
        logging.debug(f"Saved transcription to {transcription_path}")
    except Exception as e:
        app.logger.error(f"Error saving transcription: {e}", exc_info=True)

def video_to_frames(video_path, video_frames_dir):
    """
    Extracts frames from a video, resizes them, and saves them as JPEG images in the output folder.
    """
    try:
        logging.debug("Processing video...")
        clip = VideoFileClip(video_path)
        os.makedirs(video_frames_dir, exist_ok=True)
        frame_count = 0
        for frame in clip.iter_frames(fps=1):
            logging.debug(f"Processing frame {frame_count}")
            img = Image.fromarray(frame)
            logging.debug(f"Converted frame to Image object: {img}")
            try:
                resized_img = img.resize((300, 300))
                logging.debug(f"Resized frame: {resized_img}")
            except Exception as e:
                logging.error(f"Error resizing frame {frame_count}: {e}", exc_info=True)
                continue
            frame_filename = os.path.join(video_frames_dir, f'frame_{frame_count:04d}.jpg')
            try:
                resized_img.save(frame_filename, 'JPEG')
                logging.debug(f"Saved frame {frame_count} to {frame_filename}")
            except Exception as e:
                logging.error(f"Error saving frame {frame_count} to {frame_filename}: {e}", exc_info=True)
                continue
            frame_count += 1
        clip.close()
        logging.debug(f"Extracted {frame_count} frames.")
        return True
    except Exception as e:
        logging.error(f"Error extracting frames: {e}", exc_info=True)
        return False

def calculate_histogram(image):
    """
    Calculates the histogram of an image.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(hist1, hist2):
    """
    Compares two histograms using the correlation method.
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def score_text(text):
    """
    Scores the given text using the LLM model.
    """
    try:
        prompt = f"""
        Please provide a JSON response with the following structure:
        {{
            "description": "What is the content of this text?",
            "score": "On a scale of 1-10, rate if the contents of the image are radical or extremist in nature. Check for symbols, flags, banners, or any visual elements associated with extremist groups, radical ideologies, or extremist religious preaching, or contains religous topics or cultural toxicity, includes gun violence or nudity/partial nudity. Be very strict with the score as I want to protect people from such content."
        }}
        Example:
        {{
            "description": "A flag with a symbol commonly associated with an extremist group.",
            "score": 10
        }}
        Text:
        {text}
        """

        response = ollama.generate(
            model='llama3',
            prompt=prompt,
            stream=False,
            format='json'
        )

        app.logger.debug(f"Raw response from Ollama API: {response}")

        if not response or not response.get('response'):
            return None

        try:
            response_data = json.loads(response.get('response'))

            if isinstance(response_data, list):
                if len(response_data) == 0:
                    return None
                response_data = response_data[0]

            description = response_data.get('description')
            score = response_data.get('score')

            if description is None or score is None:
                return None

            return {
                'description': description,
                'score': score
            }
        except json.JSONDecodeError as e:
            app.logger.error(f"JSONDecodeError: {e}")
            return None

    except Exception as e:
        app.logger.error(f"Error in score_text: {e}", exc_info=True)
        return None

def filter_similar_frames_histogram(video_frames_dir, window_size=3, threshold=0.95):
    """
    Filters out similar frames based on histogram comparison using a sliding window approach.
    """
    try:
        frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        frames_to_keep = []
        for i in range(len(frame_files)):
            frame_path = os.path.join(video_frames_dir, frame_files[i])
            current_frame = cv2.imread(frame_path)
            current_hist = calculate_histogram(current_frame)
            if i < window_size:
                frames_to_keep.append(frame_files[i])
            else:
                similar = False
                for j in range(1, window_size + 1):
                    previous_frame_path = os.path.join(video_frames_dir, frame_files[i - j])
                    previous_frame = cv2.imread(previous_frame_path)
                    previous_hist = calculate_histogram(previous_frame)
                    similarity_score = compare_histograms(current_hist, previous_hist)
                    logging.debug(f"Similarity score for frame {frame_files[i]} and frame {frame_files[i - j]}: {similarity_score}")
                    if similarity_score > threshold:
                        similar = True
                        break
                if not similar:
                    frames_to_keep.append(frame_files[i])
        for frame_file in frame_files:
            if frame_file not in frames_to_keep:
                frame_path = os.path.join(video_frames_dir, frame_file)
                try:
                    os.remove(frame_path)
                    logging.debug(f"Removed similar frame: {frame_file}")
                except FileNotFoundError:
                    app.logger.warning(f"Frame file not found during removal: {frame_path}")
                except Exception as e:
                    logging.error(f"Error removing frame {frame_file}: {e}", exc_info=True)
        logging.debug("Filtered similar frames.")
        return True
    except Exception as e:
        logging.error(f"Error filtering similar frames: {e}", exc_info=True)
        return False

def process_frames_with_model(frames_dir):
    """
    Processes each frame through a multimodal model like LLava using Ollama.
    Returns a list of dictionaries containing description and score for each frame.
    """
    frame_responses = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        print('Processing frame:', frame_file)
        try:
            with open(frame_path, 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                prompt = """
                Please provide a JSON response with the following structure:
                {
                    "description": "What is in this picture?",
                    "score": "On a scale of 1-10, rate if the contents of the image are radical or extremist in nature. Check for symbols, flags, banners, or any visual elements associated with extremist groups, radical ideologies, or extremist religious preaching, or contains religous topics or cultural toxicity, includes gun violence or nudity/partial nudity. Be very strict with the score as I want to protect people from such content."
                }
                Example:
                {
                    "description": "A flag with a symbol commonly associated with an extremist group.",
                    "score": 10
                }
                """
                response = ollama.generate(
                    model='llava',
                    prompt=prompt,
                    stream=False,
                    images=[image_base64],
                    format='json'
                )
                if response:
                    print(response)
                    response_data = json.loads(response.get('response'))
                    print(response_data)
                    frame_responses.append({"description": response_data.get('description'), "score": response_data.get('score')})
                    print(frame_responses)
                app.logger.debug(f"Processed frame: {frame_path}")
        except Exception as e:
            app.logger.error(f"Error processing frame {frame_file}: {e}", exc_info=True)
    return frame_responses

def get_relevant_data_from_chromadb(video_uuid, frame_index):
    """
    Fetches relevant data from ChromaDB based on the video UUID and frame index.
    """
    try:
        results = collection.query(
            query_embeddings=embeddings.embed_query(f"video_uuid:{video_uuid} frame_index:{frame_index}"),
            n_results=1
        )
        if results['documents']:
            return results['documents'][0]
    except Exception as e:
        app.logger.error(f"Error fetching data from ChromaDB: {e}", exc_info=True)
    return None

def summarize_with_tiny_llama(compiled_descriptions, video_uuid):
    """
    Summarizes the compiled descriptions using Ollama's Tiny LLaMA model and includes relevant data from ChromaDB.
    """
    relevant_data = []
    for i, description in enumerate(compiled_descriptions.split(" ")):
        data = get_relevant_data_from_chromadb(video_uuid, i)
        if data:
            relevant_data.append(data['description'])

    prompt = f"""
    Please provide a concise summary of the following descriptions and consider the relevant data from the database:

    Descriptions:
    {compiled_descriptions}

    Relevant Data:
    {' '.join(relevant_data)}

    Summary:
    """

    response = ollama.generate(
        model='llama3',
        prompt=prompt,
        stream=False,
        format='json'
    )

    if response and 'response' in response:
        summary = response.get('response', 'No summary available')
        return summary
    else:
        app.logger.error(f"No response from the model for summary generation: {response}")
        return 'No response from the model'

def store_frame_data(video_uuid, frame_responses):
    """
    Stores frame descriptions and scores in ChromaDB with embeddings.
    """
    try:
        for i, frame in enumerate(frame_responses):
            description = frame['description']
            score = frame['score']
            if not description:
                continue
            embedding = embeddings.embed_query(description)
            vectorstore.add_texts(
                texts=[description],
                embeddings=[embedding],
                metadatas=[{"video_uuid": video_uuid, "frame_index": i, "score": score}],
                ids=[f"{video_uuid}_{i}"]
            )
        logging.debug("Stored frame data in ChromaDB")
        return True
    except Exception as e:
        app.logger.error(f"Error storing frame data: {e}", exc_info=True)
        return False

def insert_into_supabase(report_id, created_at, summary, frame_timeline, transcription, data_type, score, description):
    """
    Inserts data into the Supabase table.
    """
    try:
        data = {
            "id": str(uuid.uuid4()),  # Generate a unique ID for the 'id' column
            "report_id": report_id,
            "created_at": created_at,
            "summary": summary,
            "frame_timeline": frame_timeline,
            "transcription": transcription,
            "type": data_type,
            "score": score,
            "summary": description
        }
        response = supabase.table('gp_reports').insert(data).execute()
        if response.status_code == 201:
            logging.debug("Data inserted into Supabase successfully")
            return True
        else:
            app.logger.error(f"Failed to insert data into Supabase: {response.text}")
            return False
    except Exception as e:
        app.logger.error(f"Error inserting data into Supabase: {e}", exc_info=True)
        return False

def insert_into_supabase_source(report_id, created_at, source_url, source_text, source_type):
    """
    Inserts data into the Supabase gp_source table.
    """
    try:
        data = {
            "id": str(uuid.uuid4()),  # Generate a unique ID for the 'id' column
            "report_id": report_id,
            "created_at": created_at,
            "source_url": source_url,
            "source_text": source_text,
            "source_type": source_type
        }

        response = supabase.table('gp_source').insert(data).execute()

        if response.status_code == 201:
            logging.debug("Data inserted into gp_source table successfully")
            return True
        else:
            logging.error(f"Failed to insert data into gp_source: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Error inserting data into gp_source: {e}", exc_info=True)
        return False


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def store_radical_data(video_uuid, description, score):
    """
    Stores radical data in ChromaDB with embeddings.
    """
    try:
        embedding = embeddings.embed_query(description)
        vectorstore.add_texts(
            texts=[description],
            embeddings=[embedding],
            metadatas=[{"video_uuid": video_uuid, "score": score}],
            ids=[f"{video_uuid}_radical"]
        )
        logging.debug("Stored radical data in ChromaDB")
        return True
    except Exception as e:
        app.logger.error(f"Error storing radical data: {e}", exc_info=True)
        return False

def retrieve_radical_data(video_uuid):
    """
    Retrieves radical data from ChromaDB based on the video UUID.
    """
    try:
        results = collection.query(
            query_embeddings=embeddings.embed_query(f"video_uuid:{video_uuid}"),
            n_results=10
        )
        if results['documents']:
            return results['documents']
    except Exception as e:
        app.logger.error(f"Error retrieving radical data from ChromaDB: {e}", exc_info=True)
    return []

# ENDPOINTS

@app.route('/download', methods=['POST'])
def download_video():
    """
    Endpoint to download a YouTube video, extract its audio, transcribe the audio to text,
    extract frames from the video, process frames, summarize descriptions, and store data in ChromaDB.
    """
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        caption_lang = data.get('caption_lang', 'en')
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400

        video_url = expand_youtube_url(video_url)

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(video_download_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'subtitlesformat': 'srt',
            'writesubtitles': True,
            'subtitleslangs': [caption_lang],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                app.logger.error(f"Error downloading video: {e}", exc_info=True)
                return jsonify({'error': 'Failed to download video'}), 500

            try:
                info_dict = ydl.extract_info(video_url, download=False)
                video_title = info_dict.get('title', None)
                video_filename = ydl.prepare_filename(info_dict)
            except Exception as e:
                app.logger.error(f"Error extracting video info: {e}", exc_info=True)
                return jsonify({'error': 'Failed to extract video info'}), 500

            video_uuid = str(uuid.uuid4())
            video_path = os.path.join(video_download_dir, f"{video_uuid}.mp4")
            audio_path = os.path.join(video_download_dir, f"{video_uuid}.wav")
            caption_path = os.path.join(video_download_dir, f"{video_uuid}.{caption_lang}.srt")

            try:
                os.rename(video_filename, video_path)
                logging.debug(f"Renamed video file to {video_path}")
            except FileNotFoundError:
                app.logger.error(f"Downloaded video file not found: {video_filename}")
                return jsonify({'error': 'Downloaded video file not found'}), 500
            except Exception as e:
                app.logger.error(f"Error renaming video file: {e}", exc_info=True)
                return jsonify({'error': 'Error renaming video file'}), 500

        success = extract_audio(video_path, audio_path)
        if not success:
            return jsonify({'error': 'Failed to extract audio'}), 500

        print(pipe)
        transcription = query(audio_path)
        print(transcription)

        save_transcription(video_uuid, transcription)

        video_frames_dir = os.path.join(frames_dir, video_uuid)

        success = video_to_frames(video_path, video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to extract frames'}), 500

        success = filter_similar_frames_histogram(video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to filter similar frames'}), 500

        frame_responses = process_frames_with_model(video_frames_dir)

        descriptions = [frame['description'] for frame in frame_responses if frame['description']]
        compiled_descriptions = " ".join(descriptions)
        app.logger.debug(f"Compiled Descriptions: {compiled_descriptions}")
        app.logger.debug(f"Type of compiled_descriptions: {type(compiled_descriptions)}")

        transcription_str = str(transcription)
        app.logger.debug(f"Transcription: {transcription_str}")
        app.logger.debug(f"Type of transcription: {type(transcription_str)}")

        compiled_descriptions = compiled_descriptions + " " + transcription_str  # Corrected line
        app.logger.debug(f"Final Compiled Descriptions: {compiled_descriptions}")

        # Score the compiled descriptions
        score_response = score_text(compiled_descriptions)
        if not score_response:
            return jsonify({'error': 'Failed to score text'}), 500

        score = score_response.get('score')
        description = score_response.get('description')

        summary = summarize_with_tiny_llama(compiled_descriptions + transcription_str, video_uuid)
        app.logger.debug(f"Summary: {summary}")

        if not summary:
            app.logger.error("Summary generation failed.")
            return jsonify({'error': 'Failed to generate summary'}), 500

        frame_store_success = store_frame_data(video_uuid, frame_responses)

        for frame in frame_responses:
            if frame['score'] > 5:
                store_radical_data(video_uuid, frame['description'], frame['score'])

        for frame_file in os.listdir(video_frames_dir):
            frame_path = os.path.join(video_frames_dir, frame_file)
            try:
                os.remove(frame_path)
                logging.debug(f"Removed frame file: {frame_path}")
            except FileNotFoundError:
                app.logger.warning(f"Frame file not found during removal: {frame_path}")
            except Exception as e:
                app.logger.error(f"Error removing frame file {frame_path}: {e}", exc_info=True)

        for file_path in [video_path, audio_path, caption_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed file: {file_path}")
                except Exception as e:
                    app.logger.error(f"Error removing file {file_path}: {e}", exc_info=True)
            else:
                app.logger.warning(f"File not found during removal: {file_path}")

        response_data = {
            'message': 'Video downloaded, processed, and transcription stored successfully',
            'transcription': transcription,
            'summary': summary,
            'frame_descriptions': frame_responses,
            'type': "video",
            'score': score,
            'vid': video_uuid
        }

        frame_timeline = json.dumps(frame_responses)
        report_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        data_type = "video"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        insert_into_supabase(report_id, created_at, summary, frame_timeline, transcription, data_type, score, description)

        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error in /download: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/text', methods=['POST'])
def process_text():
    """
    Endpoint to process text input with an LLM to detect radical content and store the results in ChromaDB.
    """
    video_uuid = str(uuid.uuid4())
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        chunks = chunk_text(text)
        compiled_descriptions = " ".join(chunks)

        prompt = f"""
        Please provide a JSON response with the following structure:
        {{
            "description": "What is the content of this text?",
            "score": "On a scale of 1-10, rate if the contents of the image are radical or extremist in nature. Check for symbols, flags, banners, or any visual elements associated with extremist groups, radical ideologies, or extremist religious preaching, or contains religous topics or cultural toxicity, includes gun violence or nudity/partial nudity. Be very strict with the score as I want to protect people from such content."
        }}
        Example:
        {{
            "description": "A flag with a symbol commonly associated with an extremist group.",
            "score": 10
        }}
        Text:
        {compiled_descriptions}
        """

        response = ollama.generate(
            model='llama3',
            prompt=prompt,
            stream=False,
            format='json'
        )

        app.logger.debug(f"Raw response from Ollama API: {response}")

        if not response or not response.get('response'):
            return jsonify({'error': 'No response from the model'}), 500

        try:
            response_data = json.loads(response.get('response'))

            if isinstance(response_data, list):
                if len(response_data) == 0:
                    return jsonify({'error': 'Empty response from the model'}), 500
                response_data = response_data[0]

            description = response_data.get('description')
            score = response_data.get('score')

            if description is None or score is None:
                return jsonify({'error': 'Invalid response from the model'}), 500

            embedding = embeddings.embed_query(compiled_descriptions)
            metadata = {"description": description, "score": score}

            if not isinstance(metadata, dict):
                return jsonify({'error': 'Invalid metadata structure'}), 500

            vectorstore.add_texts(
                texts=[compiled_descriptions],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            logging.debug("Stored text data in ChromaDB")

            if score > 5:
                store_radical_data(str(uuid.uuid4()), description, score)

            response_data = {
                'description': description,
                'score': score,
                'vid': video_uuid
            }

            print(response_data)

            # Insert data into Supabase
            report_id = video_uuid
            created_at = datetime.utcnow().isoformat()
            data_type = "text"
            og_text = text  # Ensure the original text is assigned to og_text
            insert_into_supabase_source(report_id, created_at, None, og_text, data_type)
            insert_into_supabase(report_id, created_at, description, None, None, data_type, score, description)

            return jsonify(response_data), 200

        except json.JSONDecodeError as e:
            app.logger.error(f"JSONDecodeError: {e}")
            return jsonify({'error': 'Invalid JSON response from the model'}), 500

    except Exception as e:
        app.logger.error(f"Error in /text: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/image', methods=['POST'])
def process_image():
    """
    Endpoint to process an image input with an LLM to detect radical content and store the results in ChromaDB.
    """
    video_uuid = str(uuid.uuid4())
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(buffer_dir, filename)
        file.save(file_path)

        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        prompt = """
        Please provide a JSON response with the following structure:
        {
            "description": "What is in this picture?",
            "score": "On a scale of 1-10, rate if the contents of the image are radical or extremist in nature. Check for symbols, flags, banners, or any visual elements associated with extremist groups, radical ideologies, or extremist religious preaching, or contains religous topics or cultural toxicity, weapons, lingerie etc, includes gun violence or nudity/partial nudity. Be very strict with the score as I want to protect people from such content."
        }
        Example:
        {
            "description": "A flag with a symbol commonly associated with an extremist group.",
            "score": 10
        }
        """

        response = ollama.generate(
            model='llava',
            prompt=prompt,
            stream=False,
            images=[image_base64],
            format='json'
        )

        if not response or not response.get('response'):
            return jsonify({'error': 'No response from the model'}), 500

        try:
            response_data = json.loads(response.get('response'))

            if isinstance(response_data, list):
                if len(response_data) == 0:
                    return jsonify({'error': 'Empty response from the model'}), 500
                response_data = response_data[0]

            description = response_data.get('description')
            score = response_data.get('score')

            if description is None or score is None:
                return jsonify({'error': 'Invalid response from the model'}), 500

            embedding = embeddings.embed_query(description)
            metadata = {"description": description, "score": score}

            if not isinstance(metadata, dict):
                return jsonify({'error': 'Invalid metadata structure'}), 500

            vectorstore.add_texts(
                texts=[description],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            logging.debug("Stored image data in ChromaDB")

            if score > 5:
                store_radical_data(str(uuid.uuid4()), description, score)

            response_data = {
                'description': description,
                'score': score,
                'vid': video_uuid
            }

            print(response_data)

            # Insert data into Supabase
            report_id = video_uuid
            created_at = datetime.utcnow().isoformat()
            data_type = "image"
            # insert_into_supabase_source(report_id, created_at, description, None, None, data_type, score, description)
            insert_into_supabase(report_id, created_at, description, None, None, data_type, score, description)

            return jsonify(response_data), 200

        except json.JSONDecodeError as e:
            app.logger.error(f"JSONDecodeError: {e}")
            return jsonify({'error': 'Invalid JSON response from the model'}), 500

    except Exception as e:
        app.logger.error(f"Error in /image: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/video', methods=['POST'])
def process_video():
    """
    Endpoint to process a local video input, extract frames, process frames, summarize descriptions, and store data in ChromaDB.
    """
    video_uuid = str(uuid.uuid4())
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(video_download_dir, filename)
        file.save(video_path)

        audio_path = os.path.join(video_download_dir, f"{video_uuid}.wav")

        success = extract_audio(video_path, audio_path)
        if not success:
            return jsonify({'error': 'Failed to extract audio'}), 500

        transcription = pipe(audio_path)

        save_transcription(video_uuid, transcription)

        video_frames_dir = os.path.join(frames_dir, video_uuid)

        success = video_to_frames(video_path, video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to extract frames'}), 500

        success = filter_similar_frames_histogram(video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to filter similar frames'}), 500

        frame_responses = process_frames_with_model(video_frames_dir)

        descriptions = [frame['description'] for frame in frame_responses if frame['description']]
        compiled_descriptions = " ".join(descriptions)
        app.logger.debug(f"Compiled Descriptions: {compiled_descriptions}")

        # Score the compiled descriptions
        score_response = score_text(compiled_descriptions)
        if not score_response:
            return jsonify({'error': 'Failed to score text'}), 500

        score = score_response.get('score')
        description = score_response.get('description')

        summary = summarize_with_tiny_llama(compiled_descriptions, video_uuid)
        app.logger.debug(f"Summary: {summary}")

        store_success = store_transcription(video_uuid, transcription)

        frame_store_success = store_frame_data(video_uuid, frame_responses)
        if not frame_store_success:
            return jsonify({'error': 'Failed to store frame data'}), 500

        for frame in frame_responses:
            # Convert score to float before comparison to handle any type issues
            frame_score = float(frame['score']) if isinstance(frame['score'], (str, float)) else frame['score']
            if frame_score > 5:
                store_radical_data(video_uuid, frame['description'], frame_score)

        for frame_file in os.listdir(video_frames_dir):
            frame_path = os.path.join(video_frames_dir, frame_file)
            try:
                os.remove(frame_path)
                logging.debug(f"Removed frame file: {frame_path}")
            except FileNotFoundError:
                app.logger.warning(f"Frame file not found during removal: {frame_path}")
            except Exception as e:
                app.logger.error(f"Error removing frame file {frame_path}: {e}", exc_info=True)

        for file_path in [video_path, audio_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed file: {file_path}")
                except Exception as e:
                    app.logger.error(f"Error removing file {file_path}: {e}", exc_info=True)
            else:
                app.logger.warning(f"File not found during removal: {file_path}")

        response_data = {
            'message': 'Video processed and transcription stored successfully',
            'transcription': transcription,
            'summary': summary,
            'frame_descriptions': frame_responses,
            'type': "video",
            'score': score,
            'vid': video_uuid
        }

        frame_timeline = json.dumps(frame_responses)
        report_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        data_type = "video"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        insert_into_supabase(report_id, created_at, summary, frame_timeline, transcription, data_type, score, description)

        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error in /video: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
