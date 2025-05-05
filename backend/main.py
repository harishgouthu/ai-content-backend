# import os
# import shutil
# import uuid
# import subprocess
# import sys
# import logging
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional
# from moviepy.editor import VideoClip
# import numpy as np
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Directory setup
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# OUTPUT_DIR = os.path.join(BASE_DIR, "output")
#
# # Subdirectories for media storage
# IMAGE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "images")
# VIDEO_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "videos")
# VOICE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "voices")
# STATIC_VOICE_DIR = os.path.join(STATIC_DIR, "voices")
# STATIC_VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
# GENERATED_VIDEO_DIR = os.path.join(OUTPUT_DIR, "generated_videos")
#
# # Ensure directories exist
# for directory in [
#     IMAGE_UPLOAD_DIR, VIDEO_UPLOAD_DIR, VOICE_UPLOAD_DIR,
#     STATIC_VOICE_DIR, STATIC_VIDEO_DIR, GENERATED_VIDEO_DIR
# ]:
#     os.makedirs(directory, exist_ok=True)
#
# # Mount static files for serving generated media
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#
# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# def generate_cloned_voice(text: str, voice_sample_path: str, output_audio_path: str, language: str):
#     """
#     Generate cloned AI voice from input text and voice sample using TTS.
#     """
#     try:
#         escaped_text = text.replace('"', '\\"').replace('\n', ' ')
#         python_executable = sys.executable  # Use the current environment's Python
#
#         # Construct the command for TTS voice cloning
#         command = [
#             python_executable,
#             "-c",
#             f"from TTS.api import TTS; "
#             f"tts = TTS(model_name='tts_models/multilingual/multi-dataset/your_tts', progress_bar=False, gpu=False); "
#             f"tts.tts_to_file(text='{escaped_text}', speaker_wav=r'{voice_sample_path}', language='{language}', file_path=r'{output_audio_path}')"
#         ]
#
#         subprocess.run(command, check=True, text=True, capture_output=True)
#         logger.info(f"✅ AI voice cloned and saved at: {output_audio_path}")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ Error generating AI voice: {e.stderr}")
#         raise HTTPException(status_code=500, detail=f"AI voice generation failed: {e.stderr}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
#
# def create_placeholder_video(output_path):
#     """
#     Create a simple black screen video as a placeholder (30 seconds long).
#     """
#     try:
#         # Create a simple black screen video using moviepy (1 FPS, 640x480 resolution)
#         def make_frame(t):
#             return np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
#
#         video = VideoClip(make_frame, duration=30)  # 30 seconds video
#         video.write_videofile(output_path, fps=1)  # Save video at 1 FPS (to simulate black screen)
#         logger.info(f"✅ Placeholder video created at: {output_path}")
#     except Exception as e:
#         logger.error(f"❌ Error creating placeholder video: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error creating placeholder video: {str(e)}")
#
# def run_wav2lip(face_video_path: str, audio_path: str, output_video_path: str):
#     """
#     Run Wav2Lip to lip-sync the input video and generated audio.
#     """
#     try:
#         # For this function, you are still using the Python from wav2lip_env.
#         # If you prefer to use the current environment instead, replace with sys.executable.
#         wav2lip_env_python = os.path.join(BASE_DIR, "wav2lip_env", "Scripts", "python")  # Adjust for your OS
#
#         checkpoint_path = os.path.join(BASE_DIR, "Wav2Lip", "checkpoints", "wav2lip_gan.pth")
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
#
#         command = [
#             wav2lip_env_python,
#             os.path.join(BASE_DIR, "Wav2Lip", "inference.py"),
#             "--face", face_video_path,
#             "--audio", audio_path,
#             "--outfile", output_video_path,
#             "--checkpoint_path", checkpoint_path,
#         ]
#
#         subprocess.run(command, check=True)
#         logger.info(f"✅ Lip-synced video generated at: {output_video_path}")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ Wav2Lip error: {e.stderr}")
#         raise HTTPException(status_code=500, detail="Error during Wav2Lip lip-sync process.")
#     except FileNotFoundError as e:
#         logger.error(f"❌ Error: {str(e)}")
#         raise HTTPException(status_code=404, detail=str(e))
#
# @app.post("/generate-ai-lip-sync")
# async def generate_ai_lip_sync(
#     image: Optional[UploadFile] = File(None),
#     video: Optional[UploadFile] = File(None),
#     voice_sample: UploadFile = File(...),
#     text: str = Form(...),
#     language: str = Form("en")
# ):
#     """
#     Endpoint to generate an AI lip-synced video with a cloned voice.
#     """
#     try:
#         unique_id = str(uuid.uuid4())
#
#         # Save voice sample
#         voice_path = os.path.join(VOICE_UPLOAD_DIR, f"voice_{unique_id}.wav")
#         with open(voice_path, "wb") as buffer:
#             shutil.copyfileobj(voice_sample.file, buffer)
#         logger.info(f"✅ Voice sample saved at: {voice_path}")
#
#         # Save image or video; if none provided, create a placeholder video
#         if image:
#             input_media_path = os.path.join(IMAGE_UPLOAD_DIR, f"image_{unique_id}.jpg")
#             with open(input_media_path, "wb") as buffer:
#                 shutil.copyfileobj(image.file, buffer)
#             logger.info(f"✅ Image saved at: {input_media_path}")
#         elif video:
#             input_media_path = os.path.join(VIDEO_UPLOAD_DIR, f"video_{unique_id}.mp4")
#             with open(input_media_path, "wb") as buffer:
#                 shutil.copyfileobj(video.file, buffer)
#             logger.info(f"✅ Video saved at: {input_media_path}")
#         else:
#             input_media_path = os.path.join(VIDEO_UPLOAD_DIR, f"placeholder_{unique_id}.mp4")
#             create_placeholder_video(input_media_path)
#             logger.info(f"✅ Placeholder video created at: {input_media_path}")
#
#         # Paths for output
#         audio_output_path = os.path.join(STATIC_VOICE_DIR, f"cloned_voice_{unique_id}.wav")
#         lip_sync_output_path = os.path.join(STATIC_VIDEO_DIR, f"lip_synced_video_{unique_id}.mp4")
#
#         # Generate AI voice and lip-sync video
#         generate_cloned_voice(text, voice_path, audio_output_path, language)
#         run_wav2lip(input_media_path, audio_output_path, lip_sync_output_path)
#
#         # Return URL for the generated video
#         return JSONResponse(content={"message": "AI lip-sync video generated", "video_url": f"/static/videos/lip_synced_video_{unique_id}.mp4"})
#
#     except Exception as e:
#         logger.error(f"❌ Error generating AI lip-sync video: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error generating AI lip-sync video: {str(e)}")
#     finally:
#         # IMPORTANT: Consider if you want to clean up files immediately.
#         # If you remove the generated video, the /download endpoint might fail.
#         # Here, cleanup is commented out. Uncomment if cleanup is desired.
#         # for file_path in [voice_path, input_media_path, audio_output_path, lip_sync_output_path]:
#         #     if os.path.exists(file_path):
#         #         os.remove(file_path)
#         pass
#
# @app.get("/download/{filename}")
# async def download_file(filename: str):
#     """
#     Endpoint to download a generated video.
#     """
#     file_path = os.path.join(STATIC_VIDEO_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     else:
#         raise HTTPException(status_code=404, detail="File not found")
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
#
# import os
# import shutil
# import uuid
# import subprocess
# import sys
# import logging
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional
# from moviepy.editor import VideoClip
# import numpy as np
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Directory setup
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# OUTPUT_DIR = os.path.join(BASE_DIR, "output")
#
# IMAGE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "images")
# VIDEO_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "videos")
# VOICE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "voices")
# STATIC_VOICE_DIR = os.path.join(STATIC_DIR, "voices")
# STATIC_VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
#
# # Create directories if they do not exist
# for directory in [IMAGE_UPLOAD_DIR, VIDEO_UPLOAD_DIR, VOICE_UPLOAD_DIR, STATIC_VOICE_DIR, STATIC_VIDEO_DIR]:
#     os.makedirs(directory, exist_ok=True)
#
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#
# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def generate_cloned_voice(text: str, voice_sample_path: str, output_audio_path: str, language: str):
#     """ Generate AI cloned voice using TTS. """
#     try:
#         escaped_text = text.replace('"', '\\"').replace('\n', ' ')
#         python_executable = sys.executable
#
#         command = [
#             python_executable,
#             "-c",
#             f"from TTS.api import TTS; "
#             f"tts = TTS(model_name='tts_models/multilingual/multi-dataset/your_tts', progress_bar=False, gpu=False); "
#             f"tts.tts_to_file(text='{escaped_text}', speaker_wav=r'{voice_sample_path}', language='{language}', file_path=r'{output_audio_path}')"
#         ]
#
#         subprocess.run(command, check=True, text=True, capture_output=True)
#         logger.info(f"✅ AI voice cloned and saved at: {output_audio_path}")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ Error generating AI voice: {e.stderr}")
#         raise HTTPException(status_code=500, detail=f"AI voice generation failed: {e.stderr}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
#
#
# def run_wav2lip(face_video_path: str, audio_path: str, output_video_path: str):
#     """ Run Wav2Lip subprocess in wav2lip_env. """
#     try:
#         # Use wav2lip_env's Python interpreter on Windows
#         wav2lip_env_python = r"D:\ai-content-creator\backend\wav2lip_env\Scripts\python.exe"
#         checkpoint_path = os.path.join(BASE_DIR, "Wav2Lip", "checkpoints", "wav2lip_gan.pth")
#
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
#
#         inference_script_path = os.path.join(BASE_DIR, "Wav2Lip", "inference.py")
#
#         if not os.path.exists(inference_script_path):
#             raise FileNotFoundError(f"Inference script not found at {inference_script_path}")
#
#         command = [
#             wav2lip_env_python,
#             inference_script_path,
#             "--face", face_video_path,
#             "--audio", audio_path,
#             "--outfile", output_video_path,
#             "--checkpoint_path", checkpoint_path,
#         ]
#
#         process = subprocess.run(command, check=True, capture_output=True, text=True)
#         logger.info(f"✅ Lip-synced video generated at: {output_video_path}")
#         logger.info(f"Subprocess output: {process.stdout}")
#
#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ Wav2Lip error: {e.stderr}")
#         raise HTTPException(status_code=500, detail=f"Wav2Lip subprocess error: {e.stderr}")
#     except FileNotFoundError as e:
#         logger.error(f"❌ File not found: {str(e)}")
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         logger.error(f"❌ Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
#
#
# @app.post("/generate-ai-lip-sync")
# async def generate_ai_lip_sync(
#         image: Optional[UploadFile] = File(None),
#         video: Optional[UploadFile] = File(None),
#         voice_sample: UploadFile = File(...),
#         text: str = Form(...),
#         language: str = Form("en")
# ):
#     """ Endpoint to generate AI lip-synced video with a cloned voice. """
#     try:
#         unique_id = str(uuid.uuid4())
#         voice_path = os.path.join(VOICE_UPLOAD_DIR, f"voice_{unique_id}.wav")
#         with open(voice_path, "wb") as buffer:
#             shutil.copyfileobj(voice_sample.file, buffer)
#
#         if image:
#             input_media_path = os.path.join(IMAGE_UPLOAD_DIR, f"image_{unique_id}.jpg")
#             with open(input_media_path, "wb") as buffer:
#                 shutil.copyfileobj(image.file, buffer)
#         elif video:
#             input_media_path = os.path.join(VIDEO_UPLOAD_DIR, f"video_{unique_id}.mp4")
#             with open(input_media_path, "wb") as buffer:
#                 shutil.copyfileobj(video.file, buffer)
#         else:
#             raise HTTPException(status_code=400, detail="Image or video must be provided.")
#
#         audio_output_path = os.path.join(STATIC_VOICE_DIR, f"cloned_voice_{unique_id}.wav")
#         lip_sync_output_path = os.path.join(STATIC_VIDEO_DIR, f"lip_synced_video_{unique_id}.mp4")
#
#         generate_cloned_voice(text, voice_path, audio_output_path, language)
#         run_wav2lip(input_media_path, audio_output_path, lip_sync_output_path)
#
#         return JSONResponse(content={"message": "AI lip-sync video generated",
#                                      "video_url": f"/static/videos/lip_synced_video_{unique_id}.mp4"})
#
#     except Exception as e:
#         logger.error(f"❌ Error generating AI lip-sync video: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error generating AI lip-sync video: {str(e)}")
#
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import shutil
import uuid
import subprocess
import sys
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional


logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "images")
VIDEO_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "videos")
VOICE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "voices")
STATIC_VOICE_DIR = os.path.join(STATIC_DIR, "voices")
STATIC_VIDEO_DIR = os.path.join(STATIC_DIR, "videos")

# Create directories if they do not exist
for directory in [IMAGE_UPLOAD_DIR, VIDEO_UPLOAD_DIR, VOICE_UPLOAD_DIR, STATIC_VOICE_DIR, STATIC_VIDEO_DIR]:
    os.makedirs(directory, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file: UploadFile, destination_path: str):
    """ Helper function to save uploaded file to disk. """
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        logger.info(f"✅ File saved at: {destination_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

def generate_cloned_voice(text: str, voice_sample_path: str, output_audio_path: str, language: str):
    """ Generate AI cloned voice using TTS and save it to the output path. """
    try:
        escaped_text = text.replace('"', '\\"').replace('\n', ' ')
        python_executable = sys.executable

        command = [
            python_executable,
            "-c",
            f"from TTS.api import TTS; "
            f"tts = TTS(model_name='tts_models/multilingual/multi-dataset/your_tts', progress_bar=False, gpu=False); "
            f"tts.tts_to_file(text='{escaped_text}', speaker_wav=r'{voice_sample_path}', language='{language}', file_path=r'{output_audio_path}')"
        ]

        process = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info(f"✅ AI voice cloned and saved at: {output_audio_path}")
        logger.info(f"Subprocess output: {process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error generating AI voice: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"AI voice generation failed: {e.stderr}")





def run_wav2lip(face_video_path: str, audio_path: str, output_video_path: str):
    """ Run Wav2Lip subprocess in wav2lip_env to generate lip-synced video. """
    wav2lip_env_python = r"D:\ai-content-creator\backend\wav2lip_env\Scripts\python.exe"
    checkpoint_path = os.path.join(BASE_DIR, "Wav2Lip", "checkpoints", "wav2lip_gan.pth")
    inference_script_path = os.path.join(BASE_DIR, "Wav2Lip", "inference.py")

    # Python script to check video frames inside wav2lip_env (to isolate cv2)
    frame_check_script = """
import cv2
import sys
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if frame_count < 2:
    print(f"ERROR: Insufficient frames in input video ({frame_count} frames).")
    sys.exit(1)
else:
    print(f"✅ Input video has {frame_count} frames.")
"""

    # Check frames using cv2 within wav2lip_env
    try:
        frame_check_process = subprocess.run(
            [wav2lip_env_python, "-c", frame_check_script, face_video_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(frame_check_process.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Frame check failed: {e.stderr}")
        raise HTTPException(status_code=400, detail="Insufficient frames in input video. Please provide a longer video.")

    # Verify necessary files
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    if not os.path.exists(inference_script_path):
        raise FileNotFoundError(f"Inference script not found at {inference_script_path}")

    # Wav2Lip command
    command = [
        wav2lip_env_python,
        inference_script_path,
        "--face", face_video_path,
        "--audio", audio_path,
        "--outfile", output_video_path,
        "--checkpoint_path", checkpoint_path,
    ]

    logger.info(f"Running Wav2Lip command: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"✅ Wav2Lip subprocess output: {process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Wav2Lip subprocess failed with error: {e.stderr}")
        raise HTTPException(status_code=500, detail="Wav2Lip subprocess execution failed.")

    # Verify that the output video was generated
    logger.info(f"Expected output video path: {output_video_path}")
    if not os.path.exists(output_video_path):
        logger.error("❌ Lip-synced video not found at the expected path.")
        raise HTTPException(status_code=500, detail="Lip-synced video generation failed.")
    else:
        logger.info(f"✅ Lip-synced video generated successfully at {output_video_path}")

def validate_file_type(uploaded_file: UploadFile, allowed_mime_types: list):
    """ Validate the MIME type of the uploaded file. """
    if uploaded_file.content_type not in allowed_mime_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {uploaded_file.content_type}")

@app.post("/generate-ai-lip-sync")
async def generate_ai_lip_sync(
        image: Optional[UploadFile] = File(None),
        video: Optional[UploadFile] = File(None),
        voice_sample: UploadFile = File(...),
        text: str = Form(...),
        language: str = Form("en")
):
    """ Endpoint to generate AI lip-synced video with a cloned voice. """
    try:
        unique_id = str(uuid.uuid4())
        voice_path = os.path.join(VOICE_UPLOAD_DIR, f"voice_{unique_id}.wav")
        save_uploaded_file(voice_sample, voice_path)

        # Handle image or video input
        if image:
            validate_file_type(image, ["image/jpeg", "image/png"])
            input_media_path = os.path.join(IMAGE_UPLOAD_DIR, f"image_{unique_id}.jpg")
            save_uploaded_file(image, input_media_path)
        elif video:
            validate_file_type(video, ["video/mp4"])
            input_media_path = os.path.join(VIDEO_UPLOAD_DIR, f"video_{unique_id}.mp4")
            save_uploaded_file(video, input_media_path)
        else:
            raise HTTPException(status_code=400, detail="Image or video must be provided.")

        audio_output_path = os.path.join(STATIC_VOICE_DIR, f"cloned_voice_{unique_id}.wav")
        lip_sync_output_path = os.path.join(STATIC_VIDEO_DIR, f"lip_synced_video_{unique_id}.mp4")

        logger.info(f"🔍 Saving cloned voice at: {audio_output_path}")
        logger.info(f"🔍 Saving lip-synced video at: {lip_sync_output_path}")

        # Generate cloned voice and run Wav2Lip
        generate_cloned_voice(text, voice_path, audio_output_path, language)
        run_wav2lip(input_media_path, audio_output_path, lip_sync_output_path)

        return JSONResponse(content={"message": "AI lip-sync video generated",
                                     "video_url": f"/static/videos/lip_synced_video_{unique_id}.mp4"})

    except Exception as e:
        logger.error(f"❌ Error generating AI lip-sync video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating AI lip-sync video: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
