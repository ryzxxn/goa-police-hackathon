import logging
import os
import shutil
import subprocess
from transcribe_anything.api import transcribe


def transcribe_audio_whisper(audio_path):
    """
    Transcribes audio to text using Whisper AI via transcribe-anything.
    """
    try:
        print("audio path : ", audio_path)
        # result = subprocess.run(
        #     [
        #         "transcribe-anything",
        #         audio_path,
        #     ]
        # )
        result = transcribe(
            url_or_file=audio_path,
            output_dir="output_dir",
        )
        print("result :", result)
        transcript_path = "output_dir"
        transcription = ""
        with open(os.path.join(transcript_path, "out.txt")) as F:
            transcription = F.read()

        # THIS DELETES THE FOLDER, COMMENT IT OUT IF YOU WANT TO KEEP IT
        shutil.rmtree(transcript_path)


        if result.returncode != 0:
            print(f"Error transcribing audio: {result.stderr}")
            return ""

        # transcription = result.stdout
        logging.debug(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


if __name__ == "__main__":
    audio_path = "Why I stopped choking the chicken (mp3cut.net).mp3"
    x = transcribe_audio_whisper(audio_path=audio_path)
    print("transcription : ", x)