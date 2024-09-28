from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
from moviepy.editor import VideoFileClip
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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

def transcribe_audio_speech_recognition(audio_path):
    """
    Transcribes audio to text using SpeechRecognition.
    """
    print("Transcribing audio...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            print("Transcription:", transcription)
            return transcription
        except sr.UnknownValueError:
            app.logger.error("Google Speech Recognition could not understand the audio")
            return ""
        except sr.RequestError as e:
            app.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

def store_transcription(video_title, transcription):
    """
    Stores the transcribed text in ChromaDB with embeddings.
    """
    try:
        embedding = embeddings.embed_query(transcription)
        # Use add_texts instead of add
        vectorstore.add_texts(
            texts=[transcription],
            metadatas=[{"video_title": video_title}],
            ids=[video_title]  # Ensure unique IDs or generate unique IDs
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
        # Load the video file
        clip = VideoFileClip(video_path)

        # Ensure output directory exists
        os.makedirs(video_frames_dir, exist_ok=True)

        # Iterate through each frame and save it as a JPEG image
        frame_count = 0
        for frame in clip.iter_frames(fps=1):  # Capture 1 frame per second
            logging.debug(f"Processing frame {frame_count}")

            # Convert the frame to an Image object
            img = Image.fromarray(frame)
            logging.debug(f"Converted frame to Image object: {img}")

            # Resize the frame to 300x300
            try:
                resized_img = img.resize((300, 300))
                logging.debug(f"Resized frame: {resized_img}")
            except Exception as e:
                logging.error(f"Error resizing frame {frame_count}: {e}", exc_info=True)
                continue

            # Save the frame as a JPEG image
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

        # Remove frames that are not in frames_to_keep
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

                # Convert image data to base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                # Define prompt for the model
                prompt = """
                Please provide a JSON response with the following structure:

                {
                    "description": "What is in this picture?",
                    "score": "On a scale of 1-10, rate if the contents of the image are radical or extremist in nature. Check for symbols, flags, banners, or any visual elements associated with extremist groups, radical ideologies, or extremist religious preaching. Be very strict with the score as I want to protect people from such content."
                }

                Example:

                {
                    "description": "A flag with a symbol commonly associated with an extremist group.",
                    "score": 10
                }
                """

                # Call the Ollama API
                response = ollama.generate(
                    model='llava',  # Ensure 'llava' model is available
                    prompt=prompt,
                    stream=False,
                    images=[image_base64],
                    format='json'
                )

                # Parse the response as JSON
                if(response):
                    print(response)

                    response_data = json.loads(response.get('response'))
                    print(response_data)

                    frame_responses.append({"description": response_data.get('description'), "score": response_data.get('score')})
                    print(frame_responses)

                app.logger.debug(f"Processed frame: {frame_path}")

        except Exception as e:
            app.logger.error(f"Error processing frame {frame_file}: {e}", exc_info=True)

    return frame_responses

def summarize_with_tiny_llama(compiled_descriptions):
    """
    Summarizes the compiled descriptions using Ollama's Tiny LLaMA model.
    """
    print(compiled_descriptions)
    prompt = f"Please provide a concise summary of the following descriptions:\n\n{compiled_descriptions}\n\nSummary:"

    response = ollama.generate(
        model='tinyllama',  # Ensure 'tiny_llama' model is available
        prompt=prompt,
        stream=False
    )

    if response:
        print(response)
        # Access the 'response' key to get the summary text
        return response.get('response', 'No summary available')

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
                continue  # Skip frames with empty descriptions
            embedding = embeddings.embed_query(description)
            vectorstore.add_texts(
                texts=[description],
                embeddings=[embedding],
                metadatas=[{"video_uuid": video_uuid, "frame_index": i, "score": score}],
                ids=[f"{video_uuid}_{i}"]  # Ensure unique IDs
            )
        logging.debug("Stored frame data in ChromaDB")
        return True
    except Exception as e:
        app.logger.error(f"Error storing frame data: {e}", exc_info=True)
        return False

############################################################################################
# ENDPOINTS
############################################################################################

@app.route('/download', methods=['POST'])
def download_video():
    """
    Endpoint to download a YouTube video, extract its audio, transcribe the audio to text,
    extract frames from the video, process frames, summarize descriptions, and store data in ChromaDB.
    """
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        caption_lang = data.get('caption_lang', 'en')  # Default to English
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
            'subtitlesformat': 'srt',  # Download subtitles in SRT format
            'writesubtitles': True,  # Write subtitles to file
            'subtitleslangs': [caption_lang],  # Specify the language for subtitles
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                app.logger.error(f"Error downloading video: {e}", exc_info=True)
                return jsonify({'error': 'Failed to download video'}), 500

            # Extract video info to get the title and filename
            try:
                info_dict = ydl.extract_info(video_url, download=False)
                video_title = info_dict.get('title', None)
                video_filename = ydl.prepare_filename(info_dict)
            except Exception as e:
                app.logger.error(f"Error extracting video info: {e}", exc_info=True)
                return jsonify({'error': 'Failed to extract video info'}), 500

            # Generate a unique UUID for the video and audio
            video_uuid = str(uuid.uuid4())
            video_path = os.path.join(video_download_dir, f"{video_uuid}.mp4")
            audio_path = os.path.join(video_download_dir, f"{video_uuid}.wav")
            caption_path = os.path.join(video_download_dir, f"{video_uuid}.{caption_lang}.srt")

            # Rename the downloaded video file
            try:
                os.rename(video_filename, video_path)
                logging.debug(f"Renamed video file to {video_path}")
            except FileNotFoundError:
                app.logger.error(f"Downloaded video file not found: {video_filename}")
                return jsonify({'error': 'Downloaded video file not found'}), 500
            except Exception as e:
                app.logger.error(f"Error renaming video file: {e}", exc_info=True)
                return jsonify({'error': 'Error renaming video file'}), 500

        # Extract audio from the downloaded video
        success = extract_audio(video_path, audio_path)
        if not success:
            return jsonify({'error': 'Failed to extract audio'}), 500

        # Transcribe the extracted audio to text
        transcription = transcribe_audio_speech_recognition(audio_path)
        if transcription is None:
            transcription = ""

        # Save the transcription temporarily
        save_transcription(video_uuid, transcription)

        # Create a directory for the extracted frames using the video's UUID
        video_frames_dir = os.path.join(frames_dir, video_uuid)

        # Extract frames from the video and save them in the created directory
        success = video_to_frames(video_path, video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to extract frames'}), 500

        # Filter out similar frames using histogram comparison with a sliding window approach
        success = filter_similar_frames_histogram(video_frames_dir)
        if not success:
            return jsonify({'error': 'Failed to filter similar frames'}), 500

        # Process frames with the model
        frame_responses = process_frames_with_model(video_frames_dir)

        # Compile descriptions for summarization
        descriptions = [frame['description'] for frame in frame_responses if frame['description']]
        compiled_descriptions = " ".join(descriptions)
        app.logger.debug(f"Compiled Descriptions: {compiled_descriptions}")

        # Summarize the compiled descriptions using Tiny LLaMA
        summary = summarize_with_tiny_llama(compiled_descriptions)
        app.logger.debug(f"Summary: {summary}")

        # Store the transcription in ChromaDB
        store_success = store_transcription(video_title, transcription)
        if not store_success:
            return jsonify({'error': 'Failed to store transcription'}), 500

        # Store frame data in ChromaDB
        frame_store_success = store_frame_data(video_uuid, frame_responses)
        if not frame_store_success:
            return jsonify({'error': 'Failed to store frame data'}), 500

        # Delete all processed frames
        for frame_file in os.listdir(video_frames_dir):
            frame_path = os.path.join(video_frames_dir, frame_file)
            try:
                os.remove(frame_path)
                logging.debug(f"Removed frame file: {frame_path}")
            except FileNotFoundError:
                app.logger.warning(f"Frame file not found during removal: {frame_path}")
            except Exception as e:
                app.logger.error(f"Error removing frame file {frame_path}: {e}", exc_info=True)

        # Optionally, remove audio and video files after processing to save space
        for file_path in [video_path, audio_path, caption_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed file: {file_path}")
                except Exception as e:
                    app.logger.error(f"Error removing file {file_path}: {e}", exc_info=True)
            else:
                app.logger.warning(f"File not found during removal: {file_path}")

        # Prepare the response with frame descriptions and summary
        return jsonify({
            'message': 'Video downloaded, processed, and transcription stored successfully',
            'transcription': transcription,
            'summary': summary,
            'frame_descriptions': frame_responses  # Array of {description, score} for each frame
        })

    except Exception as e:
        app.logger.error(f"Error in /download: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)