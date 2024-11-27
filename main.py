import requests
import librosa
import numpy as np
import base64
import wave
import os
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions
import soundfile as sf
import cv2  # Para processar o vídeo

API_URL = "http://127.0.0.1:60808/chat"


def run_vad(ori_audio, sr):
    """
    Applies Voice Activity Detection (VAD) to the original audio to extract only speech portions.
    
    Args:
        ori_audio (bytes): Original audio in bytes format.
        sr (int): Sample rate of the original audio.

    Returns:
        bytes: Processed audio with speech portions, in bytes format.
    """
    try:
        # Convert the audio from bytes to NumPy array
        audio = np.frombuffer(ori_audio, dtype=np.int16).astype(np.float32) / 32768.0
        sampling_rate = 16000  # Standard sample rate
        if sr != sampling_rate:
            # Resample the audio to the desired sample rate
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        # Apply VAD (Voice Activity Detection)
        vad_parameters = VadOptions()  # VAD parameters
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        processed_audio = collect_chunks(audio, speech_chunks)
        
        # Resample back to original sample rate if necessary
        if sr != sampling_rate:
            vad_audio = librosa.resample(processed_audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = processed_audio
        
        # Convert processed audio back to bytes
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        return vad_audio.tobytes()
    except Exception as e:
        print(f"Error applying VAD: {e}")
        return ori_audio


def ensure_audio_format(input_path, output_path, target_sr=16000):
    """
    Converts the audio file to the desired format, ensuring the correct sample rate and mono channel.
    
    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the converted audio file.
        target_sr (int): Desired sample rate (default is 16000).
    """
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, audio, target_sr, subtype='PCM_16')


def save_audio(audio_bytes, output_path, sr=24000):
    """
    Saves audio in WAV format based on the given audio bytes.

    Args:
        audio_bytes (bytes): Processed audio in bytes.
        output_path (str): Path to save the WAV audio file.
        sr (int): Sample rate (default is 24000).
    """
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16 bits per sample
        wf.setframerate(sr)  # Sample rate
        wf.writeframes(audio_bytes)


def encode_file_to_base64(file_path):
    """Encode a file to base64."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_last_frame_from_video(video_path):
    """
    Extracts the last frame of a video as an image.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        PIL.Image: The last frame of the video as an image.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = None, None
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break
    cap.release()

    if frame is not None:
        # Convert the frame to a format that can be used in base64 encoding
        from PIL import Image
        import io
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()
    else:
        return None


def prepare_and_send_audio_video(input_audio_path, input_video_path=None):
    """
    Prepares both audio and video for sending: converts, encodes in base64, and sends via HTTP POST.
    
    Args:
        input_audio_path (str): Path to the input audio file.
        input_video_path (str): Path to the input video file.
    """
    processed_audio_path = "processed_audio.wav"
    ensure_audio_format(input_audio_path, processed_audio_path)

    # Base64 encode the audio
    base64_audio = encode_file_to_base64(processed_audio_path)

    payload = {
        "audio": base64_audio,
    }

    if input_video_path:
        # Extract and encode the last frame of the video as base64
        last_frame_data = extract_last_frame_from_video(input_video_path)
        if last_frame_data:
            base64_video = base64.b64encode(last_frame_data).decode('utf-8')
            payload["image"] = base64_video

    output_audio_bytes = b""  # Variable to store output audio
    resp_text = ""  # Variable to store the response text

    # Send POST request with audio and video
    try:
        with requests.post(API_URL, json=payload) as response:
            buffer = b''  # Buffer to process response data
            for chunk in response.iter_content(chunk_size=2048):
                buffer += chunk
                while b'\r\n--frame\r\n' in buffer:
                    frame, buffer = buffer.split(b'\r\n--frame\r\n', 1)
                    if b'Content-Type: audio/wav' in frame:
                        audio_data = frame.split(b'\r\n\r\n', 1)[1]
                        output_audio_bytes += audio_data
                    elif b'Content-Type: text/plain' in frame:
                        text_data = frame.split(b'\r\n\r\n', 1)[1].decode()
                        resp_text += text_data

            print(resp_text)
            # Process the output audio
            output_audio_path = "output_audio.wav"
            #save_audio(output_audio_bytes, output_audio_path)
            print(f"Audio response saved to {output_audio_path}")

    except Exception as e:
        print(f"Error during request: {e}")


# Example usage:
input_audio_path = "tests/videos/dogs/audio.wav"
input_video_path = "tests/videos/dogs/dogs.mp4"
prepare_and_send_audio_video(input_audio_path, input_video_path)
