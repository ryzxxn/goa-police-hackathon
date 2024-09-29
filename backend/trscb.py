import logging
import subprocess
from transcribe_anything.api import transcribe


def transcribe_audio_whisper(audio_path):
    """
    Transcribes audio to text using Whisper AI via transcribe-anything.
    """
    try:
        print("audio path : ", audio_path)
        # Run the transcribe-anything command
        # result = subprocess.run(
        #     ["transcribe-anything", audio_path],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        #     text=True,
        # )
        result = subprocess.run(
            [
                "transcribe-anything",
                "Why I stopped choking the chicken (mp3cut.net).mp3",
            ]
        )

        print("result :", result)

        if result.returncode != 0:
            print(f"Error transcribing audio: {result.stderr}")
            return ""

        transcription = result.stdout
        logging.debug(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}", exc_info=True)
        return ""


if __name__ == "__main__":
    transcribe_audio_whisper("Why I stopped choking the chicken.mp3")
