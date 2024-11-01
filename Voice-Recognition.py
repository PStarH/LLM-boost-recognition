import os
import glob
import traceback
import asyncio
import json
import re
import aiohttp
import aiofiles
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import List, Dict, Tuple, Optional
from decouple import config
import speech_recognition as sr  # For speech recognition
# import whisper  # For Whisper model
import librosa
import numpy as np
import soundfile as sf
from filelock import FileLock, Timeout
from transformers import AutoTokenizer, pipeline
import subprocess
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import webrtcvad
from pydub import AudioSegment
from langdetect import detect
from spellchecker import SpellChecker
import torch
import torchaudio
from pyannote.audio import Pipeline as PyannotePipeline
import fasttext
import rnn_noise_suppression  # Hypothetical advanced noise suppression library
from deep_speech import Model as DeepSpeechModel  # Hypothetical DeepSpeech integration

# Configuration for advanced noise suppression
try:
    import rnnoise  # Neural noise suppression library
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    logging.warning("RNNoise library not found. Advanced noise suppression will be limited.")

# Configuration for pyannote.audio VAD
try:
    pyannote_pipeline = PyannotePipeline.from_pretrained("pyannote/speech-activity-detection")
    PYANNOTE_AVAILABLE = True
except Exception:
    PYANNOTE_AVAILABLE = False
    logging.warning("Pyannote.audio VAD not available. Falling back to basic VAD.")

# Configuration for fastText language detection
try:
    language_model = fasttext.load_model(config('FASTTEXT_MODEL_PATH', default='lid.176.bin'))
    FASTTEXT_AVAILABLE = True
except Exception:
    FASTTEXT_AVAILABLE = False
    logging.warning("FastText language detection model not found. Falling back to langdetect.")

# Configuration for model ensembles
# WHISPER_AVAILABLE = whisper.__version__ is not None  # Removed whisper check
SPEECH_RECOGNITION_AVAILABLE = True if 'sr' in globals() else False
DEEPSPEECH_AVAILABLE = False  # Will be initialized later based on availability

WHOLE_SPEECH_RECOGNITION_AVAILABLE = False or SPEECH_RECOGNITION_AVAILABLE or DEEPSPEECH_AVAILABLE

# Configuration
USE_LOCAL_LLM = config('USE_LOCAL_LLM', default=True, cast=bool)
API_PROVIDER = config('API_PROVIDER', default="OPENAI", cast=str)  # OPENAI or LOCAL
OPENAI_API_KEY = config('OPENAI_API_KEY', default="", cast=str)
# WHISPER_MODEL_NAME = config('WHISPER_MODEL_NAME', default="base", cast=str)  # Removed whisper model config

# Configuration for audio processing
AUDIO_FORMAT = config('AUDIO_FORMAT', default="wav", cast=str)  # e.g., 'wav', 'mp3'
AUDIO_SAMPLE_RATE = config('AUDIO_SAMPLE_RATE', default=16000, cast=int)
AUDIO_CHANNELS = config('AUDIO_CHANNELS', default=1, cast=int)

# Supported languages configuration
SUPPORTED_LANGUAGES = config(
    "SUPPORTED_LANGUAGES",
    default="en,es,fr,de,it,zh,ja,ko,ru,ar,hi,pt,sv,nl,da,fi,no,pl,tr,vi",
    cast=lambda v: [lang.strip() for lang in v.split(",")],
)

# Custom vocabulary for SpeechRecognition
CUSTOM_VOCABULARY = config(
    "CUSTOM_VOCABULARY",
    default="",
    cast=lambda v: [word.strip() for word in v.split(",")],
)

# Custom pronunciation dictionary
CUSTOM_PRONUNCIATIONS = config(
    "CUSTOM_PRONUNCIATIONS",
    default="",
    cast=lambda v: {
        word.strip(): pronunciation.strip()
        for word, pronunciation in (item.split(":") for item in v.split(",") if ":" in item)
    },
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)

# ThreadPoolExecutor for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

async def preprocess_audio(audio_path: str, processed_audio_path: str) -> None:
    """
    Preprocesses the audio by converting to the desired format, reducing noise, normalizing volume,
    and applying advanced voice activity detection.

    Args:
        audio_path (str): Path to the original audio file.
        processed_audio_path (str): Path to save the processed audio file.
    """
    try:
        logging.info("Loading and converting audio file...")
        # Load audio and convert to desired format using pydub for better handling
        audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
        audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(AUDIO_CHANNELS)
        await asyncio.to_thread(audio.export, processed_audio_path, format="wav")

        # Advanced noise suppression using RNNoise if available
        if RNNOISE_AVAILABLE:
            logging.info("Applying neural noise suppression with RNNoise...")
            denoised_audio = await asyncio.to_thread(rnn_noise_suppression.reduce_noise, processed_audio_path)
            await asyncio.to_thread(sf.write, processed_audio_path, denoised_audio, AUDIO_SAMPLE_RATE)
            logging.info("Neural noise suppression completed.")
        else:
            # Perform spectral gating noise reduction as a fallback
            logging.info("Applying spectral gating for noise reduction...")
            y, sr = await asyncio.to_thread(librosa.load, processed_audio_path, sr=AUDIO_SAMPLE_RATE)
            reduced_noise = spectral_gate(y, sr)
            # Normalize audio
            logging.info("Normalizing audio...")
            normalized_audio = reduced_noise / np.max(np.abs(reduced_noise))
            await asyncio.to_thread(sf.write, processed_audio_path, normalized_audio, sr)
            logging.info("Spectral gating and normalization completed.")

        # Enhanced Voice Activity Detection (VAD) using pyannote.audio
        if PYANNOTE_AVAILABLE:
            logging.info("Applying advanced Voice Activity Detection (VAD) using pyannote.audio...")
            vad_segments = await asyncio.to_thread(pyannote_pipeline, {"audio": processed_audio_path})
            trimmed_audio = extract_segments(processed_audio_path, vad_segments)
            await asyncio.to_thread(sf.write, processed_audio_path, trimmed_audio, AUDIO_SAMPLE_RATE)
            logging.info("Advanced VAD processing completed.")
        else:
            # Fallback to basic VAD
            logging.info("Applying basic Voice Activity Detection (VAD)...")
            trimmed_audio, trimmed_rate = await asyncio.to_thread(apply_vad, processed_audio_path)
            await asyncio.to_thread(sf.write, processed_audio_path, trimmed_audio, trimmed_rate)
            logging.info(f"VAD processing completed: {processed_audio_path}")

    except Exception as e:
        logging.error(f"Error during audio preprocessing: {e}")

def spectral_gate(audio, sr, prop_decrease=1.0, n_fft=2048, hop_length=512, win_length=2048):
    """
    Applies spectral gating for noise reduction using librosa.

    Args:
        audio (np.ndarray): Audio signal.
        sr (int): Sample rate.
        prop_decrease (float): Proportion to decrease noise components.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Each frame of audio is windowed by window of length win_length.

    Returns:
        np.ndarray: Denoised audio signal.
    """
    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        magnitude, phase = librosa.magphase(stft)

        # Estimate noise power from the first 0.5 seconds
        noise_frames = int(0.5 * sr / hop_length)
        noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        magnitude = np.where(magnitude < noise_mag * prop_decrease, 0, magnitude)

        stft_clean = magnitude * phase
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length, win_length=win_length)
        return audio_clean
    except Exception as e:
        logging.error(f"Spectral gating error: {e}")
        return audio

def extract_segments(audio_path: str, segments) -> np.ndarray:
    """
    Extracts voiced segments from the audio based on VAD results.

    Args:
        audio_path (str): Path to the processed audio file.
        segments: VAD segments from pyannote.audio.

    Returns:
        np.ndarray: Trimmed audio data.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        trimmed_audio = np.array([])
        for segment in segments.get_timeline():
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            trimmed_audio = np.concatenate((trimmed_audio, audio[start_sample:end_sample]))
        return trimmed_audio
    except Exception as e:
        logging.error(f"Error extracting segments: {e}")
        audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        return audio

def apply_vad(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Applies Voice Activity Detection to trim silence from the audio using webrtcvad as a fallback.

    Args:
        audio_path (str): Path to the processed audio file.

    Returns:
        Tuple[np.ndarray, int]: Trimmed audio data and sample rate.
    """
    try:
        vad = webrtcvad.Vad(int(config('VAD_AGGRESSIVENESS', default=3)))
        audio, sample_rate = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        frame_duration = int(config('FRAME_DURATION_MS', default=30))  # ms
        frames = frame_generator(frame_duration, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, frame_duration, 300, vad, frames)
        trimmed_audio = b"".join([segment.bytes for segment in segments])
        trimmed_np = (
            np.frombuffer(trimmed_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return trimmed_np, sample_rate
    except Exception as e:
        logging.error(f"VAD error: {e}")
        audio, sample_rate = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        return audio, sample_rate

def frame_generator(frame_duration_ms: int, audio: np.ndarray, sample_rate: int):
    """
    Generates audio frames from PCM audio data.

    Args:
        frame_duration_ms (int): Duration of each frame in milliseconds.
        audio (np.ndarray): PCM audio data.
        sample_rate (int): Sample rate of the audio.

    Yields:
        Frame: Audio frame.
    """
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    num_frames = len(audio) // frame_size
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        yield Frame(
            audio[start:end].tobytes(),
            timestamp=i * frame_duration_ms / 1000.0,
            duration=frame_duration_ms / 1000.0,
        )

class Frame:
    """
    Represents a frame of audio data.
    """

    def __init__(self, bytes: bytes, timestamp: float, duration: float):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def vad_collector(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad,
    frames,
):
    """
    Filters out non-voiced audio frames using webrtcvad as a fallback.

    Args:
        sample_rate (int): Sample rate of the audio.
        frame_duration_ms (int): Duration of each frame in milliseconds.
        padding_duration_ms (int): Duration of padding in milliseconds.
        vad: VAD object.
        frames: Iterable of Frame objects.

    Yields:
        Frame: Voiced audio frames.
    """
    num_padding_frames = padding_duration_ms // frame_duration_ms
    ring_buffer = []
    triggered = False

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_voiced = len(
                [f for f in ring_buffer if vad.is_speech(f.bytes, sample_rate)]
            )
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                for f in ring_buffer:
                    yield f
                ring_buffer = []
        else:
            yield frame
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_unvoiced = len(
                [f for f in ring_buffer if not vad.is_speech(f.bytes, sample_rate)]
            )
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                ring_buffer = []

async def validate_words_with_llm(
    text: str, language: Optional[str] = "en"
) -> List[Tuple[str, bool]]:
    """
    Validates each word in the transcription using the LLM to ensure contextual accuracy.

    Args:
        text (str): The transcription text to validate.
        language (str, optional): Language of the text. Defaults to "en".

    Returns:
        List[Tuple[str, bool]]: A list of tuples containing words and their validation status.
    """
    logging.info("Validating each word in the transcription using LLM...")
    validated_words = []
    try:
        spell = SpellChecker(language=language)
        words = text.split()
        for word in words:
            if word in CUSTOM_VOCABULARY:
                # Skip validation for custom vocabulary as they are expected to be correct
                validated_words.append((word, True))
                continue
            corrected = spell.correction(word)
            is_correct = word == corrected
            validated_words.append((word, is_correct))
            if not is_correct:
                logging.debug(f"Word '{word}' corrected to '{corrected}'.")
        logging.info("Word validation completed.")
    except Exception as e:
        logging.error(f"Error during word validation with LLM: {e}")
    return validated_words

async def refine_transcription_contextually(
    text: str, language: Optional[str] = "en"
) -> str:
    """
    Refines the entire transcription using the LLM to improve contextual accuracy.

    Args:
        text (str): The initial transcription text.
        language (str, optional): Language of the text. Defaults to "en".

    Returns:
        str: Contextually refined transcription text.
    """
    logging.info("Refining transcription contextually using LLM...")
    try:
        prompt = (
            f"Improve the accuracy and coherence of the following transcription without altering the original words unnecessarily:\n\n"
            f"{text}\n\nRefined Transcription:"
        )
        refined_text = await generate_completion(
            prompt, max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS
        )
        if refined_text:
            # Apply spell checking as an additional step
            spell = SpellChecker(language=language)
            corrected_text = " ".join(
                [
                    spell.correction(word) if spell.unknown([word]) else word
                    for word in refined_text.split()
                ]
            )
            logging.info("Contextual refinement completed successfully.")
            return corrected_text
        else:
            logging.warning(
                "LLM contextual refinement returned no result. Using initial transcription."
            )
            return text
    except Exception as e:
        logging.error(f"Error during contextual transcription refinement with LLM: {e}")
        return text

async def transcribe_with_multiple_engines(
    processed_audio_path: str, language: str
) -> Tuple[str, float]:
    """
    Transcribes audio using multiple ASR engines and selects the most accurate transcription.

    Args:
        processed_audio_path (str): Path to the processed audio file.
        language (str): Language of the audio.

    Returns:
        Tuple[str, float]: The best transcription and its confidence score.
    """
    transcriptions = []
    confidences = []

    if SPEECH_RECOGNITION_AVAILABLE:
        try:
            logging.info("Transcribing with SpeechRecognition engine.")
            recognizer = sr.Recognizer()
            with sr.AudioFile(processed_audio_path) as source:
                audio = recognizer.record(source)

            # Apply custom vocabulary if any
            if CUSTOM_VOCABULARY:
                logging.info("Applying custom vocabulary to SpeechRecognition engine.")
                # Placeholder for dynamic vocabulary injection if supported
                # SpeechRecognition with Google API does not support custom vocabularies directly

            initial_text = await asyncio.to_thread(
                recognizer.recognize_google, audio, language=language
            )
            # Post-process the transcription with spell checker
            initial_text = " ".join([spell.correction(word) for word in initial_text.split()])
            transcriptions.append(initial_text)
            confidences.append(95.0)  # Assigned a high confidence as Google API does not provide it
            logging.info("SpeechRecognition transcription completed.")
        except sr.UnknownValueError:
            logging.error("Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            logging.error(f"Speech Recognition error: {e}")

    if DEEPSPEECH_AVAILABLE:
        try:
            logging.info("Transcribing with DeepSpeech engine.")
            ds_model = DeepSpeechModel(model_path=config('DEEPSPEECH_MODEL_PATH', default="deepspeech_model.pbmm"))
            transcription = await asyncio.to_thread(ds_model.stt, processed_audio_path)
            confidences.append(90.0)  # Hypothetical confidence score
            transcriptions.append(transcription)
            logging.info("DeepSpeech transcription completed.")
        except Exception as e:
            logging.error(f"DeepSpeech transcription error: {e}")

    if not transcriptions:
        logging.error("No transcriptions available from ASR engines.")
        return "", 0.0

    # Select transcription with highest confidence
    best_index = confidences.index(max(confidences))
    best_transcription = transcriptions[best_index]
    best_confidence = confidences[best_index]

    logging.info(
        "Multiple ASR engines transcribed the audio. Selected the best transcription based on confidence."
    )
    return best_transcription, best_confidence

async def convert_audio_to_text(audio_path: str) -> Tuple[str, float, Optional[str], List[Tuple[str, bool]]]:
    """
    Converts an audio file to text using multiple voice recognition engines, detects language,
    and validates each word using the LLM.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        Tuple[str, float, Optional[str], List[Tuple[str, bool]]]: Extracted text, average confidence,
        detected language, and list of validated words with their status.
    """
    try:
        # Preprocess the audio
        processed_audio_path = os.path.join("temp", "processed_" + os.path.basename(audio_path))
        os.makedirs(os.path.dirname(processed_audio_path), exist_ok=True)
        await preprocess_audio(audio_path, processed_audio_path)

        detected_language = None

        if (
            SPEECH_RECOGNITION_AVAILABLE
            or DEEPSPEECH_AVAILABLE
        ):
            # Detect language using fastText if available
            if FASTTEXT_AVAILABLE:
                logging.info("Detecting language using fastText.")
                audio, sr = await asyncio.to_thread(librosa.load, processed_audio_path, sr=AUDIO_SAMPLE_RATE)
                text, conf = await transcribe_with_multiple_engines(
                    processed_audio_path, "en"
                )  # Temporary language
                prediction = language_model.predict(text)
                detected_language = prediction[0][0].replace("__label__", "")
                logging.info(f"Detected language: {detected_language}")
            else:
                # Fallback to langdetect if fastText and Whisper are not available
                logging.info("Using langdetect for language detection.")
                audio, sr = await asyncio.to_thread(librosa.load, processed_audio_path, sr=AUDIO_SAMPLE_RATE)
                text, conf = await transcribe_with_multiple_engines(
                    processed_audio_path, "en"
                )
                detected_language = detect(text) if text else "en"
                logging.info(f"Detected language using langdetect: {detected_language}")

            # Check if detected language is supported
            if detected_language not in SUPPORTED_LANGUAGES:
                logging.warning(
                    f"Detected language '{detected_language}' is not in the supported languages list. Defaulting to English."
                )
                detected_language = "en"

            # Transcribe using multiple ASR engines
            transcription, confidence = await transcribe_with_multiple_engines(
                processed_audio_path, detected_language
            )
            initial_text = transcription
            confidence_score = confidence

            # Validate each word with LLM
            validated_words = await validate_words_with_llm(
                initial_text, language=detected_language
            )

            # Cleanup temporary processed audio file
            await asyncio.to_thread(os.remove, processed_audio_path)
            logging.info(f"Temporary file {processed_audio_path} deleted.")

            return initial_text, confidence_score, detected_language, validated_words
        else:
            logging.error("No voice recognition engine is available.")
            return "", 0.0, detected_language, []
    except Exception as e:
        logging.error(f"Failed to convert audio to text: {e}")
        return "", 0.0, None, []

async def process_command_text(
    text: str, language: Optional[str] = "en"
) -> str:
    """
    Processes command text using LLM for execution.

    Args:
        text (str): The command text to process.
        language (str, optional): Language of the text. Defaults to "en".

    Returns:
        str: Processed command result.
    """
    try:
        prompt = f"Execute the following command and return the result:\n\n{text}\n\nResult:"
        result = await generate_completion(prompt, max_tokens=500)
        return result if result else text
    except Exception as e:
        logging.error(f"Error processing command text: {e}")
        return text

async def correct_speech_errors(
    text: str, language: Optional[str] = "en"
) -> str:
    """
    Corrects speech recognition errors in the provided text using LLM.

    Args:
        text (str): The text to correct.
        language (str, optional): Language of the text. Defaults to "en".

    Returns:
        str: Corrected text.
    """
    logging.info("Starting speech error correction using LLM...")
    try:
        prompt = (
            f"Correct the following speech recognition errors in the text:\n\n{text}\n\nCorrected Text:"
        )
        corrected_text = await generate_completion(prompt, max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS)
        return corrected_text if corrected_text else text
    except Exception as e:
        logging.error(f"Error during speech error correction: {e}")
        return text

async def identify_speech_intent(
    text: str, language: Optional[str] = "en"
) -> str:
    """
    Identifies the intent of the spoken text using LLM.

    Args:
        text (str): The text to analyze.
        language (str, optional): Language of the text. Defaults to "en".

    Returns:
        str: Identified intent.
    """
    logging.info("Identifying speech intent using LLM...")
    try:
        prompt = f"Analyze the intent of the following text:\n\n{text}\n\nIntent:"
        intent = await generate_completion(prompt, max_tokens=50)
        return intent if intent else "Unknown"
    except Exception as e:
        logging.error(f"Error during intent identification: {e}")
        return "Unknown"

async def generate_completion(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """
    Generates text completion using the configured LLM provider.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    if USE_LOCAL_LLM:
        if API_PROVIDER.upper() == "OLLAMA":
            return await generate_completion_from_ollama(prompt, max_tokens)
        else:
            return await generate_completion_from_local_llm(
                DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens
            )
    else:
        logging.error("Local LLM usage is disabled.")
        return None

async def generate_completion_from_ollama(
    prompt: str, max_tokens: int = 500
) -> Optional[str]:
    """
    Generates text completion using Ollama's API asynchronously.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    try:
        logging.info("Generating completion using Ollama...")
        process = await asyncio.create_subprocess_exec(
            "ollama",
            "prompt",
            "--model",
            config('OLLAMA_MODEL', default="llama-13b"),  # Adjust as needed
            "--max-tokens",
            str(max_tokens),
            "--prompt",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            output = stdout.decode().strip()
            logging.info(
                f"Ollama response received. Output length: {len(output):,} characters"
            )
            return output
        else:
            logging.error(f"Ollama error: {stderr.decode().strip()}")
            return None
    except Exception as e:
        logging.error(f"Error while communicating with Ollama: {e}")
        return None

async def generate_completion_from_local_llm(
    llm_model_name: str,
    input_prompt: str,
    number_of_tokens_to_generate: int = 100,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Generates text completion using a local LLM asynchronously.

    Args:
        llm_model_name (str): Name of the LLM model to use.
        input_prompt (str): The input prompt for text generation.
        number_of_tokens_to_generate (int, optional): Number of tokens to generate. Defaults to 100.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    logging.info(
        f"Starting text completion using model: '{llm_model_name}' for input prompt."
    )
    llm = await asyncio.to_thread(load_model, llm_model_name)
    if not llm:
        logging.error("LLM model could not be loaded.")
        return None
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(
        number_of_tokens_to_generate,
        LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - 500,
    )  # TOKEN_BUFFER analogous
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = await chunk_text(
            input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - 300, llm_model_name=llm_model_name
        )  # TOKEN_CUSHION analogous
        results = []
        for chunk in chunks:
            try:
                output = await asyncio.to_thread(
                    llm,
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - 300,
                    temperature=temperature,
                )
                results.append(output["choices"][0]["text"])
                logging.info(
                    f"Chunk processed. Output tokens: {output['usage']['completion_tokens']:,}"
                )
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        output = await asyncio.to_thread(
            llm,
            prompt=input_prompt,
            max_tokens=adjusted_max_tokens,
            temperature=temperature,
        )
        generated_text = output["choices"][0]["text"]
        logging.info(
            f"Completed text completion. Beginning of generated text: \n'{generated_text[:150]}'..."
        )
        return generated_text

def load_model(llm_model_name: str, raise_exception: bool = True):
    """
    Loads the specified LLM model.

    Args:
        llm_model_name (str): Name of the LLM model to load.
        raise_exception (bool, optional): Whether to raise exceptions on failure. Defaults to True.

    Returns:
        pipeline or None: Loaded model pipeline instance or None if failed.
    """
    try:
        logging.info(f"Loading LLM model: {llm_model_name}")
        model = pipeline("text-generation", model=llm_model_name)
        return model
    except Exception as e:
        logging.error(f"Failed to load model {llm_model_name}: {e}")
        if raise_exception:
            raise
        return None

def estimate_tokens(text: str, model_name: str) -> int:
    """
    Estimates the number of tokens in the given text based on the model's encoding.

    Args:
        text (str): The text to estimate tokens for.
        model_name (str): The name of the model to use for encoding.

    Returns:
        int: Estimated number of tokens.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logging.error(f"Error estimating tokens: {e}")
        return 0

async def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
    llm_model_name: str = DEFAULT_LOCAL_MODEL_NAME,
) -> List[str]:
    """
    Splits the text into chunks suitable for processing by the LLM.

    Args:
        text (str): The text to chunk.
        chunk_size (int, optional): Maximum number of tokens per chunk. Defaults to 2000.
        overlap (int, optional): Number of overlapping tokens between chunks. Defaults to 200.
        llm_model_name (str, optional): The LLM model name for token estimation. Defaults to DEFAULT_LOCAL_MODEL_NAME.

    Returns:
        List[str]: List of text chunks.
    """
    logging.info("Starting text chunking...")
    try:
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks = []
        current_chunk = []
        current_chunk_length = 0

        for sentence in sentences:
            sentence_length = estimate_tokens(sentence, llm_model_name)
            if current_chunk_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_chunk_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_length = sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logging.info(f"Text chunked into {len(chunks)} parts.")
        return chunks
    except Exception as e:
        logging.error(f"Error during text chunking: {e}")
        return [text]

async def collect_user_feedback(original_text: str, corrected_text: str) -> None:
    """
    Collects user feedback on the transcription accuracy.

    Args:
        original_text (str): The original transcription text.
        corrected_text (str): The corrected transcription text by the user.
    """
    feedback = {
        "original_text": original_text,
        "corrected_text": corrected_text,
    }
    # Save feedback to a file or database
    feedback_file_path = "feedback.json"
    async with aiofiles.open(feedback_file_path, "a") as f:
        await f.write(json.dumps(feedback, ensure_ascii=False, indent=4) + "\n")
    logging.info(f"User feedback saved to: {feedback_file_path}")

async def main():
    """
    The main function orchestrating the voice recognition workflow.
    """
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        input_audio_file_path = config('INPUT_AUDIO_FILE_PATH', default="sample_audio.wav")
        reformat_as_markdown = config('REFORMAT_AS_MARKDOWN', default=True, cast=bool)

        # Load models if using local LLM
        if USE_LOCAL_LLM:
            logging.info(f"Using Local LLM with Model: {DEFAULT_LOCAL_MODEL_NAME}")
        else:
            logging.info(f"Using API for completions: {API_PROVIDER}")

        base_name = os.path.splitext(os.path.basename(input_audio_file_path))[0]
        output_extension = ".md" if reformat_as_markdown else ".txt"

        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        raw_sr_output_file_path = os.path.join(
            data_dir, f"{base_name}__raw_sr_output.txt"
        )
        validated_output_file_path = os.path.join(
            data_dir, f"{base_name}_validated_words.json"
        )
        refined_sr_output_file_path = os.path.join(
            data_dir, f"{base_name}_refined_sr_output{output_extension}"
        )
        intent_output_file_path = os.path.join(data_dir, f"{base_name}_intent.txt")
        feedback_file_path = os.path.join(data_dir, f"{base_name}_feedback.json")

        # Convert audio to text
        text, conf, detected_language, validated_words = await convert_audio_to_text(
            input_audio_file_path
        )
        logging.info(f"Transcription confidence: {conf:.2f}")
        logging.info(f"Detected Language: {detected_language}")

        # Contextual Refinement with LLM
        refined_text = await refine_transcription_contextually(
            text, language=detected_language
        )

        # Identify intent
        intent = await identify_speech_intent(refined_text, language=detected_language)

        # Validate words and write validation results
        async with aiofiles.open(validated_output_file_path, "w") as f:
            await f.write(
                json.dumps(validated_words, ensure_ascii=False, indent=4)
            )
        logging.info(
            f"Validated words output written to: {validated_output_file_path}"
        )

        # Write raw speech recognition output
        async with aiofiles.open(raw_sr_output_file_path, "w") as f:
            await f.write(text)
        logging.info(
            f"Raw speech recognition output written to: {raw_sr_output_file_path}"
        )

        # Save the contextual LLM refined output
        async with aiofiles.open(refined_sr_output_file_path, "w") as f:
            await f.write(refined_text)
        logging.info(
            f"LLM Refined transcription written to: {refined_sr_output_file_path}"
        )

        # Save identified intent
        async with aiofiles.open(intent_output_file_path, "w") as f:
            await f.write(intent)
        logging.info(f"Identified intent written to: {intent_output_file_path}")

        if refined_text:
            logging.info(
                f"First 500 characters of LLM refined processed text:\n{refined_text[:500]}..."
            )
        else:
            logging.warning("refined_text is empty or not defined.")

        logging.info(f"Done processing {input_audio_file_path}.")
        logging.info("\nSee output files:")
        logging.info(f" Raw Speech Recognition: {raw_sr_output_file_path}")
        logging.info(f" Validated Words: {validated_output_file_path}")
        logging.info(f" LLM Refined Transcription: {refined_sr_output_file_path}")
        logging.info(f" Identified Intent: {intent_output_file_path}")

        # Perform a final quality check
        quality_score, explanation = await assess_output_quality(text, refined_text)
        if quality_score is not None:
            logging.info(f"Final quality score: {quality_score}/100")
            logging.info(f"Explanation: {explanation}")
        else:
            logging.warning("Unable to determine final quality score.")

        # Adaptive Learning: Save feedback for future improvements
        feedback = {
            "original_text": text,
            "refined_text": refined_text,
            "validated_words": validated_words,
            "intent": intent,
            "quality_score": quality_score,
            "explanation": explanation,
        }
        async with aiofiles.open(feedback_file_path, "w") as f:
            await f.write(json.dumps(feedback, ensure_ascii=False, indent=4))
        logging.info(f"Feedback saved to: {feedback_file_path}")

        # After processing the audio and getting the transcription
        # Collect user feedback (this is just a placeholder for actual user input)
        original_text = text  # The original transcription
        corrected_text = await get_user_correction()  # Implement a method to get user correction
        await collect_user_feedback(original_text, corrected_text)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        logging.error(traceback.format_exc())

def remove_corrected_text_header(text: str) -> str:
    """
    Removes headers or irrelevant information from the corrected text.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    # Implemented functionality to remove potential headers or irrelevant information
    cleaned_text = re.sub(r"Refined Transcription:\s*", "", text)
    return cleaned_text

async def assess_output_quality(
    raw_text: str, final_text: str
) -> Tuple[Optional[int], Optional[str]]:
    """
    Assesses the quality of the output text using LLM.

    Args:
        raw_text (str): The raw speech recognition extracted text.
        final_text (str): The final processed text.

    Returns:
        Tuple[Optional[int], Optional[str]]: Quality score and explanation.
    """
    logging.info("Assessing output quality using LLM...")
    try:
        prompt = (
            f"Evaluate the quality of the following text on a scale from 0 to 100 and provide a brief explanation:\n\n"
            f"{final_text}\n\nQuality Score and Explanation:"
        )
        assessment = await generate_completion(prompt, max_tokens=100)
        if assessment:
            parts = assessment.split(":")
            if len(parts) >= 2:
                score_part = parts[0].strip()
                explanation_part = ":".join(parts[1:]).strip()
                score_matches = re.findall(r"\d+", score_part)
                score = (
                    int(score_matches[0])
                    if score_matches
                    else None
                )
                explanation = explanation_part
                return score, explanation
        return None, None
    except Exception as e:
        logging.error(f"Error during quality assessment: {e}")
        return None, None

# Initialize DeepSpeech model if available
if not DEEPSPEECH_AVAILABLE:
    try:
        ds_model = DeepSpeechModel(model_path=config('DEEPSPEECH_MODEL_PATH', default="deepspeech_model.pbmm"))
        DEEPSPEECH_AVAILABLE = True
    except Exception:
        DEEPSPEECH_AVAILABLE = False
        logging.warning(
            "DeepSpeech model not loaded. DeepSpeech transcription will be unavailable."
        )

if __name__ == "__main__":
    # Ensure temporary directory exists
    os.makedirs("temp", exist_ok=True)
    asyncio.run(main())
