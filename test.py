import shutil
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions

import streamlit as st
import wave

import numpy as np
import base64
import io
from typing import List
import os
import tempfile
import librosa
from pydub import AudioSegment
from PIL import Image
from io import BytesIO
import base64
import requests
import librosa
import numpy as np
import wave
import os
import tempfile
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions
import soundfile as sf
import cv2 
from pydub import AudioSegment

from inference_vision import OmniVisionInference
from PIL import Image
import io

OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2

stream_stride = 4
max_tokens = 2048

omni = OmniVisionInference()

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

def extract_last_frame_from_video(video_path):
    """
    Extracts the last frame of a video as an image.

    Args:
        video_path (str): Path to the video file.

    Returns:
        PIL.Image or None: The last frame of the video as an image, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video or error
            break
        last_frame = frame  # Keep updating with the latest frame
    
    cap.release()

    if last_frame is not None:
        from PIL import Image
        import io
        image = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        image.save("last_frame.jpg")
        return image
    else:
        print("Error: No frames could be read from the video.")
        return None

def extract_first_frame_from_video(video_path, output_image_path=None):
    """
    Extracts the first frame of a video as an image and optionally saves it to disk.

    Args:
        video_path (str): Path to the video file.
        output_image_path (str, optional): Path to save the first frame as an image.

    Returns:
        PIL.Image or None: The first frame of the video as an image, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    ret, frame = cap.read()  # Read the first frame
    cap.release()

    if ret and frame is not None:
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save the first frame if an output path is provided
        if output_image_path:
            image.save("first_frame.jpg")
            print(f"First frame saved to {output_image_path}")

        return image
    else:
        print("Error: Could not read the first frame from the video.")
        return None

def encode_file_to_base64(file_path):
    """Encode a file to base64."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')    

def save_tmp_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        file_name = tmpfile.name
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=OUT_SAMPLE_WIDTH,
            frame_rate=OUT_RATE,
            channels=OUT_CHANNELS,
        )
        audio.export(file_name, format="wav")
        return file_name
    
def send_audio_and_video(input_audio_path, input_video_path = None):
    
    processed_audio_path = "processed_audio.wav"
    ensure_audio_format(input_audio_path, processed_audio_path)

    base64_audio = encode_file_to_base64(processed_audio_path)

    image_data_buf = None
    if input_video_path:
        frame = extract_first_frame_from_video(input_video_path)

        if frame:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            frame_data = buffer.getvalue()
            base64_video = base64.b64encode(frame_data).decode('utf-8')

        image_data_buf = base64_video if frame_data else None
        if image_data_buf:
            image_data_buf = image_data_buf.encode("utf-8")
        image_data_buf = base64.b64decode(image_data_buf)

    audio_path, img_path = None, None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f, \
            tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
        audio_f.write(base64.b64decode(base64_audio))
        audio_path = audio_f.name

        if image_data_buf:
            img_f.write(image_data_buf)
            img_path = img_f.name
        else:
            img_path = None

        if img_path is not None:
            resp_generator = omni.run_vision_AA_batch_stream(audio_f.name, img_f.name,
                                                            stream_stride, max_tokens,
                                                            save_path='./vision_qa_out_cache.wav')
        else:
            resp_generator = omni.run_AT_batch_stream(audio_f.name, stream_stride,
                                                    max_tokens,
                                                    save_path='./audio_qa_out_cache.wav')
        
        if resp_generator:
            response_texts = []
            frames = []

            for response in resp_generator:
                if isinstance(response, tuple):
                    f, text = response
                    response_texts.append(text)
                    frames.append(f)
                else:
                    response_texts.append(str(response))

            full_response = ''.join(response_texts) 
            print("Full Response:", full_response)

            frames_combined = b''.join(frames)
            out_file = save_tmp_audio(frames_combined)
            output_dir = './saved_audio_files'
            os.makedirs(output_dir, exist_ok=True)
            output_audio_path = os.path.join(output_dir, 'output_audio.wav')
            shutil.move(out_file, output_audio_path)

        else:
            print("No response received.")


send_audio_and_video("tests/videos/rio/audio.wav", "tests/videos/rio/rio.mp4")