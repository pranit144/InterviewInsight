import os
import json
import logging
import tempfile
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import google.generativeai as genai
from groq import Groq
import pandas as pd
from datetime import datetime
import io
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from moviepy.editor import VideoFileClip
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Configure logger first before using it elsewhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure API keys with validation
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    raise ValueError("GROQ_API_KEY environment variable must be set")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable must be set")

# Initialize clients with proper configuration and error handling
try:
    # Initialize Groq client with basic configuration
    from groq._base_client import SyncHttpxClientWrapper
    import httpx

    # Create a simple httpx client
    http_client = SyncHttpxClientWrapper(
        base_url="https://api.groq.com/v1",
        timeout=httpx.Timeout(60.0)
    )

    # Initialize Groq client
    groq_client = Groq(
        api_key=GROQ_API_KEY,
        http_client=http_client
    )

    # Initialize Gemini client
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-1.5-flash"
    logger.info("API clients initialized successfully")
except Exception as e:
    logger.error(f"Error initializing API clients: {str(e)}")
    raise

# Emotion Detection Setup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
HAARCASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')

# Load models with optimized settings
try:
    # Configure TensorFlow for optimal CPU performance
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        try:
            # Limit memory growth to prevent OOM errors
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Not all devices support memory growth
            pass
            
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    # Load emotion model with optimized settings
    model = load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        raise Exception("Error: Haar Cascade file could not be loaded")
    
    logger.info("Successfully loaded model and face cascade")
except Exception as e:
    logger.error(f"Error loading model or face cascade: {str(e)}")
    model = None
    face_cascade = None

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Video processing configuration
VIDEO_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for video processing
MAX_VIDEO_DURATION = 120  # Maximum video duration in minutes
FRAME_SAMPLE_RATE = 5  # Process every 5th frame for long videos


def extract_json(text: str) -> Optional[str]:
    """Extract JSON from response text."""
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
        return None


def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extract audio from video file with optimized processing."""
    try:
        temp_audio_path = video_path.replace('.mp4', '.mp3')
        
        # Load video clip with optimized settings
        video_clip = VideoFileClip(
            video_path,
            audio_buffersize=200000,
            verbose=False,
            audio_fps=44100
        )
        
        if video_clip.audio is None:
            logger.warning("Video has no audio track")
            return None
        
        # Extract audio with optimized settings
        video_clip.audio.write_audiofile(
            temp_audio_path,
            buffersize=2000,
            verbose=False,
            logger=None
        )
        video_clip.close()

        logger.info(f"Successfully extracted audio to {temp_audio_path}")
        return temp_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None
    finally:
        # Ensure video clip is closed even if an exception occurs
        if 'video_clip' in locals() and video_clip is not None:
            try:
                video_clip.close()
            except:
                pass


def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio using Groq."""
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file not found at {audio_path}")
        return None
        
    try:
        # Transcribe audio
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )

        logger.info(f"Transcription successful: {transcription.text[:100]}...")
        return transcription.text

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None


def process_video_chunk(frame_chunk: List[np.ndarray], start_frame: int) -> Dict[str, Any]:
    """Process a chunk of video frames efficiently."""
    results = {
        'emotion_counts': {emotion: 0 for emotion in EMOTIONS},
        'faces_detected': 0,
        'frames_with_faces': 0,
        'frames_processed': 0
    }
    
    for frame_idx, frame in enumerate(frame_chunk):
        try:
            # Skip empty frames
            if frame is None or frame.size == 0:
                continue
                
            # Resize frame for faster processing if too large
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            results['frames_processed'] += 1
            if len(faces) > 0:
                results['frames_with_faces'] += 1
                results['faces_detected'] += len(faces)

                for (x, y, w, h) in faces:
                    # Add boundary checks
                    if y >= gray.shape[0] or x >= gray.shape[1] or y+h > gray.shape[0] or x+w > gray.shape[1]:
                        continue
                    
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
                    
                    if np.sum(roi) == 0:
                        continue

                    roi = roi.astype("float32") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    with tf.device('/CPU:0'):
                        preds = model.predict(roi, verbose=0)[0]
                    label = EMOTIONS[np.argmax(preds)]
                    results['emotion_counts'][label] += 1

        except Exception as e:
            logger.error(f"Error processing frame {start_frame + frame_idx}: {str(e)}")
            continue

    return results


def analyze_video_emotions(video_path: str) -> Dict[str, Any]:
    """Analyze emotions in a video with optimized processing for large files."""
    if model is None or face_cascade is None:
        logger.error("Model or face detector not properly loaded")
        return {
            'emotion_counts': {},
            'emotion_percentages': {},
            'total_faces': 0,
            'frames_processed': 0,
            'frames_with_faces': 0,
            'error': 'Models not properly loaded'
        }

    cap = None
    try:
        # Open video and get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if fps is 0
        duration = total_frames / max(fps, 1) / 60  # Duration in minutes, prevent division by zero

        # Check video duration
        if duration > MAX_VIDEO_DURATION:
            raise Exception(f"Video duration exceeds maximum limit of {MAX_VIDEO_DURATION} minutes")

        # Initialize results
        combined_results = {
            'emotion_counts': {emotion: 0 for emotion in EMOTIONS},
            'total_faces': 0,
            'frames_processed': 0,
            'frames_with_faces': 0,
            'processing_stats': {
                'total_video_frames': total_frames,
                'video_fps': fps,
                'video_duration_minutes': round(duration, 2)
            }
        }

        # Process video in chunks using ThreadPoolExecutor
        frame_buffer = []
        frame_count = 0
        chunk_size = 30  # Process 30 frames per chunk

        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as executor:
            future_to_chunk = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % FRAME_SAMPLE_RATE != 0:
                    continue

                frame_buffer.append(frame)

                if len(frame_buffer) >= chunk_size:
                    # Submit chunk for processing
                    future = executor.submit(
                        process_video_chunk,
                        frame_buffer.copy(),
                        frame_count - len(frame_buffer)
                    )
                    future_to_chunk[future] = len(frame_buffer)
                    frame_buffer = []

            # Process remaining frames
            if frame_buffer:
                future = executor.submit(
                    process_video_chunk,
                    frame_buffer,
                    frame_count - len(frame_buffer)
                )
                future_to_chunk[future] = len(frame_buffer)

            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    # Combine results
                    for emotion, count in chunk_results['emotion_counts'].items():
                        combined_results['emotion_counts'][emotion] += count
                    combined_results['total_faces'] += chunk_results['faces_detected']
                    combined_results['frames_processed'] += chunk_results['frames_processed']
                    combined_results['frames_with_faces'] += chunk_results['frames_with_faces']
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")

        # Calculate percentages
        total_emotions = sum(combined_results['emotion_counts'].values())
        combined_results['emotion_percentages'] = {
            emotion: round((count / max(total_emotions, 1) * 100), 2)
            for emotion, count in combined_results['emotion_counts'].items()
        }

        # Add processing statistics
        combined_results['processing_stats'].update({
            'frames_sampled': combined_results['frames_processed'],
            'sampling_rate': f'1/{FRAME_SAMPLE_RATE}',
            'processing_complete': True
        })

        return combined_results

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        return {
            'error': str(e),
            'emotion_counts': {emotion: 0 for emotion in EMOTIONS},
            'emotion_percentages': {emotion: 0 for emotion in EMOTIONS},
            'total_faces': 0,
            'frames_processed': 0,
            'frames_with_faces': 0,
            'processing_stats': {
                'error_occurred': True,
                'error_message': str(e)
            }
        }
    finally:
        if cap is not None:
            cap.release()


def analyze_interview(conversation_text: str, role_applied: Optional[str] = None, tech_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze technical interview transcript."""
    if not conversation_text or len(conversation_text.strip()) < 50:
        logger.warning("Transcript too short for meaningful analysis")
        return create_default_assessment()
        
    try:
        model = genai.GenerativeModel(MODEL_NAME)

        skills_context = ""
        if tech_skills and len(tech_skills) > 0:
            skills_context = f"Focus on evaluating these specific technical skills: {', '.join(tech_skills)}."

        role_context = ""
        if role_applied:
            role_context = f"The candidate is being interviewed for the role of {role_applied}."

        prompt = f"""
            Based on the following technical interview transcript, analyze the candidate's responses and provide a structured assessment in *valid JSON format*.

            {role_context}
            {skills_context}

            *JSON Format:*
            {{
                "candidate_assessment": {{
                    "technical_knowledge": {{
                        "score": 0,  // Score from 1-10
                        "strengths": [],
                        "areas_for_improvement": []
                    }},
                    "problem_solving": {{
                        "score": 0,  // Score from 1-10
                        "strengths": [],
                        "areas_for_improvement": []
                    }},
                    "communication": {{
                        "score": 0,  // Score from 1-10
                        "strengths": [],
                        "areas_for_improvement": []
                    }}
                }},
                "question_analysis": [
                    {{
                        "question": "",
                        "answer_quality": "",  // Excellent, Good, Average, Poor
                        "feedback": ""
                    }}
                ],
                "overall_recommendation": "",  // Hire, Strong Consider, Consider, Do Not Recommend
                "overall_feedback": ""
            }}

            *Interview Transcript:*
            {conversation_text}

            *Output Strictly JSON. Do NOT add explanations or extra text.*
        """

        # Set timeout and retry parameters
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        # Try to generate response with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                raw_response = response.text
                logger.info(f"Raw Gemini Response: {raw_response[:100]}...")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"All {max_retries} attempts failed")
                    return create_default_assessment()

        json_text = extract_json(raw_response)
        if json_text:
            try:
                assessment = json.loads(json_text)
                # Ensure the response has all required fields
                required_fields = {
                    'candidate_assessment': {
                        'technical_knowledge': ['score', 'strengths', 'areas_for_improvement'],
                        'problem_solving': ['score', 'strengths', 'areas_for_improvement'],
                        'communication': ['score', 'strengths', 'areas_for_improvement']
                    },
                    'question_analysis': ['question', 'answer_quality', 'feedback'],
                    'overall_recommendation': None,
                    'overall_feedback': None
                }

                # Validate and set defaults if needed
                if 'candidate_assessment' not in assessment:
                    assessment['candidate_assessment'] = {}
                
                for category in ['technical_knowledge', 'problem_solving', 'communication']:
                    if category not in assessment['candidate_assessment']:
                        assessment['candidate_assessment'][category] = {
                            'score': 5,
                            'strengths': ['Not enough information to assess.'],
                            'areas_for_improvement': ['Not enough information to assess.']
                        }
                    else:
                        cat_data = assessment['candidate_assessment'][category]
                        for field in required_fields['candidate_assessment'][category]:
                            if field not in cat_data:
                                if field == 'score':
                                    cat_data[field] = 5
                                else:
                                    cat_data[field] = ['Not enough information to assess.']

                if 'question_analysis' not in assessment or not assessment['question_analysis']:
                    assessment['question_analysis'] = [{
                        'question': 'General Interview',
                        'answer_quality': 'Average',
                        'feedback': 'Not enough specific questions to analyze.'
                    }]
                else:
                    for qa in assessment['question_analysis']:
                        for field in required_fields['question_analysis']:
                            if field not in qa:
                                qa[field] = 'Not available'

                if 'overall_recommendation' not in assessment or not assessment['overall_recommendation']:
                    assessment['overall_recommendation'] = 'Consider'

                if 'overall_feedback' not in assessment or not assessment['overall_feedback']:
                    assessment['overall_feedback'] = 'Not enough information to provide detailed feedback.'

                return assessment
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                return create_default_assessment()
        else:
            logger.error("No valid JSON found in response")
            return create_default_assessment()

    except Exception as e:
        logger.error(f"Interview analysis error: {str(e)}")
        return create_default_assessment()


def create_default_assessment() -> Dict[str, Any]:
    """Create a default assessment when analysis fails."""
    return {
        "candidate_assessment": {
            "technical_knowledge": {
                "score": 5,
                "strengths": ["Unable to assess strengths from the provided transcript."],
                "areas_for_improvement": ["Unable to assess areas for improvement from the provided transcript."]
            },
            "problem_solving": {
                "score": 5,
                "strengths": ["Unable to assess strengths from the provided transcript."],
                "areas_for_improvement": ["Unable to assess areas for improvement from the provided transcript."]
            },
            "communication": {
                "score": 5,
                "strengths": ["Unable to assess strengths from the provided transcript."],
                "areas_for_improvement": ["Unable to assess areas for improvement from the provided transcript."]
            }
        },
        "question_analysis": [{
            "question": "General Interview",
            "answer_quality": "Average",
            "feedback": "Unable to assess specific questions from the transcript."
        }],
        "overall_recommendation": "Consider",
        "overall_feedback": "Unable to provide a detailed assessment based on the provided transcript."
    }


def process_video_and_audio_parallel(video_path: str, role_applied: str = None, tech_skills: list = None) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Process video and audio in parallel with optimized handling."""
    audio_path = None
    emotion_results = None
    transcript = None
    interview_assessment = None
    
    try:
        with ThreadPoolExecutor(max_workers=min(3, os.cpu_count() or 2)) as executor:
            # Submit emotions analysis task
            emotion_future = executor.submit(analyze_video_emotions, video_path)
            
            # Submit audio extraction task
            audio_future = executor.submit(extract_audio_from_video, video_path)
            
            # Wait for audio extraction to complete with timeout
            try:
                audio_path = audio_future.result(timeout=120)  # 2 minutes timeout
            except concurrent.futures.TimeoutError:
                logger.error("Audio extraction timeout exceeded")
                audio_path = None
            
            # Continue with transcription if audio was extracted
            transcript_future = None
            if audio_path:
                transcript_future = executor.submit(transcribe_audio, audio_path)
            
            # Wait for emotion analysis with timeout
            try:
                emotion_results = emotion_future.result(timeout=300)  # 5 minutes timeout
            except concurrent.futures.TimeoutError:
                logger.error("Emotion analysis timeout exceeded")
                emotion_results = {
                    'error': 'Processing timeout exceeded',
                    'emotion_counts': {emotion: 0 for emotion in EMOTIONS},
                    'emotion_percentages': {emotion: 0 for emotion in EMOTIONS},
                    'total_faces': 0,
                    'frames_processed': 0,
                    'frames_with_faces': 0
                }
            
            # Wait for transcription with timeout
            if transcript_future:
                try:
                    transcript = transcript_future.result(timeout=300)  # 5 minutes timeout
                except concurrent.futures.TimeoutError:
                    logger.error("Transcription timeout exceeded")
                    transcript = "Transcription failed due to timeout."
            else:
                transcript = "Audio extraction failed, no transcription available."
            
            # Analyze interview content if transcript is available
            if transcript and len(transcript) > 50:
                interview_assessment = analyze_interview(transcript, role_applied, tech_skills)
            else:
                interview_assessment = create_default_assessment()
            
            # Clean up audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up audio file: {str(e)}")
            
            return emotion_results, transcript, interview_assessment
            
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        
        # Create default results if any component failed
        if not emotion_results:
            emotion_results = {
                'error': str(e),
                'emotion_counts': {emotion: 0 for emotion in EMOTIONS},
                'emotion_percentages': {emotion: 0 for emotion in EMOTIONS},
                'total_faces': 0,
                'frames_processed': 0,
                'frames_with_faces': 0
            }
        
        if not transcript:
            transcript = f"Error processing audio: {str(e)}"
            
        if not interview_assessment:
            interview_assessment = create_default_assessment()
            
        # Clean up audio file if it exists
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
                
        return emotion_results, transcript, interview_assessment


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is running."""
    return jsonify({"status": "ok", "message": "Server is running"}), 200


@app.route("/analyze_interview", methods=["POST", "OPTIONS"])
def analyze_interview_route():
    """Main route for comprehensive interview analysis."""
    # Add CORS headers for preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400'  # 24 hours
        }
        return ('', 204, headers)
        
    try:
        logger.info("Received analyze_interview request")
        
        # Check for required file
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "Video file is required"}), 400

        video_file = request.files['video']
        if not video_file:
            logger.error("Empty video file")
            return jsonify({"error": "Empty video file"}), 400

        # Get additional form data
        role_applied = request.form.get('role_applied', '')
        tech_skills = request.form.get('tech_skills', '')
        candidate_name = request.form.get('candidate_name', 'Candidate')
        tech_skills_list = [skill.strip() for skill in tech_skills.split(',')] if tech_skills else []

        # Create temporary video file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_temp:
                video_file.save(video_temp.name)
                video_temp_path = video_temp.name
                logger.info(f"Video saved to temporary file: {video_temp_path}")
        except Exception as e:
            logger.error(f"Error saving video file: {str(e)}")
            return jsonify({"error": f"Failed to save video file: {str(e)}"}), 500

        # Process video and audio in parallel
        try:
            emotion_analysis, transcript, interview_assessment = process_video_and_audio_parallel(
                video_temp_path, role_applied, tech_skills_list
            )
        except Exception as e:
            logger.error(f"Error during parallel processing: {str(e)}")
            return jsonify({"error": str(e)}), 500

        # Combine results
        combined_results = {
            "candidate_assessment": interview_assessment["candidate_assessment"],
            "question_analysis": interview_assessment["question_analysis"],
            "overall_recommendation": interview_assessment["overall_recommendation"],
            "overall_feedback": interview_assessment["overall_feedback"],
            "transcription": transcript,
            "candidate_name": candidate_name,
            "role_applied": role_applied,
            "interview_date": datetime.now().strftime('%Y-%m-%d'),
            "emotion_analysis": emotion_analysis
        }

        logger.info("Combined results created successfully")
        logger.debug(f"Response data: {json.dumps(combined_results, indent=2)}")

        # Clean up temporary video file
        try:
            os.unlink(video_temp_path)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

        # Add CORS headers to response
        response = jsonify(combined_results)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        logger.error(f"Error in analyze_interview_route: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/download_assessment', methods=['POST', 'OPTIONS'])
def download_assessment():
    """Download comprehensive assessment report."""
    # Add CORS headers for preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400'  # 24 hours
        }
        return ('', 204, headers)
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create Excel writer object
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#CCCCCC',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'border': 1
            })
            
            # Summary Sheet
            summary_data = {
                'Metric': [
                    'Technical Knowledge',
                    'Problem Solving',
                    'Communication',
                    'Overall Recommendation',
                    'Total Faces Detected'
                ],
                'Score/Rating': [
                    f"{data['candidate_assessment']['technical_knowledge']['score']}/10",
                    f"{data['candidate_assessment']['problem_solving']['score']}/10",
                    f"{data['candidate_assessment']['communication']['score']}/10",
                    data['overall_recommendation'],
                    data['emotion_analysis'].get('total_faces', 0)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format Summary sheet
            summary_sheet = writer.sheets['Summary']
            summary_sheet.set_column('A:A', 25)
            summary_sheet.set_column('B:B', 20)
            
            # Apply formats to Summary sheet
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
            
            for row_num in range(len(summary_df)):
                for col_num in range(len(summary_df.columns)):
                    summary_sheet.write(row_num + 1, col_num, summary_df.iloc[row_num, col_num], cell_format)
            
            # Technical Assessment Sheet
            tech_data = []
            
            # Add technical knowledge
            tech_data.append(['Technical Knowledge', f"{data['candidate_assessment']['technical_knowledge']['score']}/10", ''])
            tech_data.append(['Strengths', '', ''])
            for strength in data['candidate_assessment']['technical_knowledge']['strengths']:
                tech_data.append(['', '', strength])
            
            tech_data.append(['Areas for Improvement', '', ''])
            for area in data['candidate_assessment']['technical_knowledge']['areas_for_improvement']:
                tech_data.append(['', '', area])
            
            # Add problem solving
            tech_data.append(['Problem Solving', f"{data['candidate_assessment']['problem_solving']['score']}/10", ''])
            tech_data.append(['Strengths', '', ''])
            for strength in data['candidate_assessment']['problem_solving']['strengths']:
                tech_data.append(['', '', strength])
            
            tech_data.append(['Areas for Improvement', '', ''])
            for area in data['candidate_assessment']['problem_solving']['areas_for_improvement']:
                tech_data.append(['', '', area])
            
            # Add communication
            tech_data.append(['Communication', f"{data['candidate_assessment']['communication']['score']}/10", ''])
            tech_data.append(['Strengths', '', ''])
            for strength in data['candidate_assessment']['communication']['strengths']:
                tech_data.append(['', '', strength])
            
            tech_data.append(['Areas for Improvement', '', ''])
            for area in data['candidate_assessment']['communication']['areas_for_improvement']:
                tech_data.append(['', '', area])
            
            # Create Technical Assessment dataframe
            tech_df = pd.DataFrame(tech_data, columns=['Category', 'Score', 'Details'])
            tech_df.to_excel(writer, sheet_name='Technical Assessment', index=False)
            
            # Format Technical Assessment sheet
            tech_sheet = writer.sheets['Technical Assessment']
            tech_sheet.set_column('A:A', 25)
            tech_sheet.set_column('B:B', 15)
            tech_sheet.set_column('C:C', 60)
            
            # Apply formats to Technical Assessment sheet
            for col_num, value in enumerate(tech_df.columns.values):
                tech_sheet.write(0, col_num, value, header_format)
            
            # Question Analysis Sheet
            question_data = []
            for qa in data['question_analysis']:
                question_data.append([
                    qa['question'],
                    qa['answer_quality'],
                    qa['feedback']
                ])
            
            question_df = pd.DataFrame(question_data, columns=['Question', 'Answer Quality', 'Feedback'])
            question_df.to_excel(writer, sheet_name='Question Analysis', index=False)
            
            # Format Question Analysis sheet
            qa_sheet = writer.sheets['Question Analysis']
            qa_sheet.set_column('A:A', 40)
            qa_sheet.set_column('B:B', 15)
            qa_sheet.set_column('C:C', 60)
            
            # Apply formats to Question Analysis sheet
            for col_num, value in enumerate(question_df.columns.values):
                qa_sheet.write(0, col_num, value, header_format)
            
            # Emotion Analysis Sheet
            if 'emotion_analysis' in data and 'emotion_percentages' in data['emotion_analysis']:
                emotion_data = {
                    'Emotion': list(data['emotion_analysis']['emotion_percentages'].keys()),
                    'Percentage': list(data['emotion_analysis']['emotion_percentages'].values()),
                    'Count': [data['emotion_analysis']['emotion_counts'].get(emotion, 0) 
                             for emotion in data['emotion_analysis']['emotion_percentages'].keys()]
                }
                
                emotion_df = pd.DataFrame(emotion_data)
                emotion_df.to_excel(writer, sheet_name='Emotion Analysis', index=False)
                
                # Format Emotion Analysis sheet
                emotion_sheet = writer.sheets['Emotion Analysis']
                emotion_sheet.set_column('A:A', 15)
                emotion_sheet.set_column('B:B', 15)
                emotion_sheet.set_column('C:C', 15)
                
                # Apply formats to Emotion Analysis sheet
                for col_num, value in enumerate(emotion_df.columns.values):
                    emotion_sheet.write(0, col_num, value, header_format)
                
                # Add a chart
                chart = workbook.add_chart({'type': 'pie'})
                chart.add_series({
                    'name': 'Emotions',
                    'categories': ['Emotion Analysis', 1, 0, len(emotion_df), 0],
                    'values': ['Emotion Analysis', 1, 1, len(emotion_df), 1],
                    'data_labels': {'percentage': True}
                })
                
                chart.set_title({'name': 'Emotion Distribution'})
                chart.set_style(10)
                emotion_sheet.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Transcript Sheet
            if 'transcription' in data:
                transcript_data = {'Transcript': [data['transcription']]}
                transcript_df = pd.DataFrame(transcript_data)
                transcript_df.to_excel(writer, sheet_name='Transcript', index=False)
                
                # Format Transcript sheet
                transcript_sheet = writer.sheets['Transcript']
                transcript_sheet.set_column('A:A', 100)
                
                # Apply formats to Transcript sheet
                transcript_sheet.write(0, 0, 'Transcript', header_format)
            
            # Overall Feedback Sheet
            overall_data = {'Overall Feedback': [data['overall_feedback']]}
            overall_df = pd.DataFrame(overall_data)
            overall_df.to_excel(writer, sheet_name='Overall Feedback', index=False)
            
            # Format Overall Feedback sheet
            overall_sheet = writer.sheets['Overall Feedback']
            overall_sheet.set_column('A:A', 100)
            
            # Apply formats to Overall Feedback sheet
            overall_sheet.write(0, 0, 'Overall Feedback', header_format)

        # Prepare the output file for download
        output.seek(0)
        candidate_name = data.get('candidate_name', 'Candidate').replace(' ', '_')
        role_applied = data.get('role_applied', 'Role').replace(' ', '_')
        filename = f"{candidate_name}_{role_applied}_Assessment.xlsx"
        
        # Create response with appropriate headers
        response = send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"Error generating assessment report: {str(e)}")
        return jsonify({"error": f"Failed to generate assessment report: {str(e)}"}), 500


if __name__ == "__main__":
    # Setup Flask app with proper settings for production
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)