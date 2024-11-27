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
from pdf2image import convert_from_path
import pytesseract
from llama_cpp import Llama, LlamaGrammar
import tiktoken
import numpy as np
from PIL import Image
from decouple import Config as DecoupleConfig, RepositoryEnv
import cv2
from filelock import FileLock, Timeout
from transformers import AutoTokenizer
import subprocess
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Added Math OCR dependency
try:
    import mathpixocr  # Replace with the actual Math OCR library you are using
    MATH_OCR_AVAILABLE = True
except ImportError:
    MATH_OCR_AVAILABLE = False
    logging.warning("Math OCR library not found. Math formulas will not be processed with specialized OCR.")

try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Added PaddleOCR dependency
try:
    from paddleocr import PaddleOCR, draw_ocr
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logging.warning("PaddleOCR library not found. PaddleOCR will not be used as a backup OCR engine.")

# Added Text Detection Models
try:
    from east import EASTTextDetector  # Placeholder for actual EAST import
    from craft import CRAFTTextDetector  # Placeholder for actual CRAFT import
    TEXT_DETECTION_AVAILABLE = True
except ImportError:
    TEXT_DETECTION_AVAILABLE = False
    logging.warning("Text detection models (EAST/CRAFT) not found. Advanced text detection will not be used.")

# Added pix2tex dependency for LaTeX OCR
try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_OCR_AVAILABLE = True
except ImportError:
    PIX2TEX_OCR_AVAILABLE = False
    logging.warning("pix2tex library not found. LaTeX OCR will not be available.")

# Configuration
config = DecoupleConfig(RepositoryEnv('.env'))

USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=True, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OLLAMA", cast=str)  # OLLAMA or LOCAL
OLLAMA_API_URL = config.get("OLLAMA_API_URL", default="http://localhost:11434", cast=str)
OLLAMA_MODEL_NAME = config.get("OLLAMA_MODEL_NAME", default="ggml-gpt4all-j-v1.3-groovy", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = 4096  # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300  # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = 12000  # Maximum allowed tokens for OpenAI API
DEFAULT_LOCAL_MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = 2048
USE_VERBOSE = False

# Added Math OCR configuration
MATH_OCR_API_KEY = config.get("MATH_OCR_API_KEY", default="", cast=str)
MATH_OCR_ENDPOINT = config.get("MATH_OCR_ENDPOINT", default="", cast=str)  # If required by the Math OCR tool

# Added LLM Models for Error Correction and Layout Identification
LLM_ERROR_CORRECTION_MODEL = config.get("LLM_ERROR_CORRECTION_MODEL", default=DEFAULT_LOCAL_MODEL_NAME, cast=str)
LLM_LAYOUT_MODEL = config.get("LLM_LAYOUT_MODEL", default=DEFAULT_LOCAL_MODEL_NAME, cast=str)

# Added Preprocessing and Progress Tracking configurations
PREPROCESSING_ENABLED = config.get("PREPROCESSING_ENABLED", default=True, cast=bool)
PROGRESS_TRACKING_ENABLED = config.get("PROGRESS_TRACKING_ENABLED", default=True, cast=bool)

# Added OCR Engine Selection
OCR_ENGINE = config.get("OCR_ENGINE", default="pytesseract", cast=str)  # Options: pytesseract, easyocr

# Added PaddleOCR configuration
PADDLEOCR_ENABLED = config.get("PADDLEOCR_ENABLED", default=True, cast=bool)
PADDLEOCR_LANGUAGE = config.get("PADDLEOCR_LANGUAGE", default="en", cast=str)  # e.g., 'en' for English
PADDLEOCR_USE_GPU = config.get("PADDLEOCR_USE_GPU", default=False, cast=bool)

# Added Text Detection configuration
TEXT_DETECTION_MODEL = config.get("TEXT_DETECTION_MODEL", default="EAST", cast=str)  # Options: EAST, CRAFT
TEXT_DETECTION_THRESHOLD = config.get("TEXT_DETECTION_THRESHOLD", default=0.5, cast=float)

# Added LaTeX OCR configuration
LATEX_OCR_ENABLED = config.get("LATEX_OCR_ENABLED", default=True, cast=bool)
LATEX_OCR_LANGUAGE = config.get("LATEX_OCR_LANGUAGE", default="en", cast=str)  # e.g., 'en' for English

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU Check
if GPU_AVAILABLE:
    logging.info("GPU is available and will be utilized where applicable.")
else:
    logging.info("GPU is not available. Processing will proceed on CPU.")

# Initialize OCR Engines
if OCR_ENGINE.lower() == "pytesseract":
    logging.info("Using pytesseract as the primary OCR engine.")
elif OCR_ENGINE.lower() == "easyocr":
    try:
        import easyocr
        OCR_ENGINE_AVAILABLE = True
        logging.info("Using EasyOCR as the primary OCR engine.")
    except ImportError:
        OCR_ENGINE_AVAILABLE = False
        logging.error("EasyOCR is selected as the OCR engine but is not installed.")
        OCR_ENGINE = "pytesseract"
else:
    logging.warning(f"Unsupported OCR engine: {OCR_ENGINE}. Falling back to pytesseract.")
    OCR_ENGINE = "pytesseract"

# Initialize PaddleOCR
if PADDLEOCR_ENABLED and PADDLE_OCR_AVAILABLE:
    paddleocr_instance = PaddleOCR(
        lang=PADDLEOCR_LANGUAGE,
        use_angle_cls=True,
        use_gpu=PADDLEOCR_USE_GPU
    )
    logging.info("PaddleOCR is enabled and initialized.")
else:
    paddleocr_instance = None
    if PADDLEOCR_ENABLED:
        logging.warning("PaddleOCR is enabled but not available.")
    else:
        logging.info("PaddleOCR is disabled.")

# Initialize Text Detection Models
if TEXT_DETECTION_AVAILABLE:
    if TEXT_DETECTION_MODEL.upper() == "EAST":
        text_detector = EASTTextDetector(model_path=config.get("EAST_MODEL_PATH", default="models/east_text_detection.pth", cast=str))
        logging.info("EAST Text Detector initialized.")
    elif TEXT_DETECTION_MODEL.upper() == "CRAFT":
        text_detector = CRAFTTextDetector(model_path=config.get("CRAFT_MODEL_PATH", default="models/craft_text_detection.pth", cast=str))
        logging.info("CRAFT Text Detector initialized.")
    else:
        text_detector = None
        logging.warning(f"Unsupported Text Detection Model: {TEXT_DETECTION_MODEL}. Advanced text detection will not be used.")
else:
    text_detector = None
    logging.info("Advanced Text Detection models are not available.")

async def convert_pdf_to_images(pdf_path: str, max_pages: int = 0, skip_first_n_pages: int = 0) -> List[Image.Image]:
    """
    Converts a PDF file to a list of PIL Image objects asynchronously.

    Args:
        pdf_path (str): Path to the PDF file.
        max_pages (int, optional): Maximum number of pages to convert. Defaults to 0 (all pages).
        skip_first_n_pages (int, optional): Number of initial pages to skip. Defaults to 0.

    Returns:
        List[Image.Image]: List of images converted from PDF pages.
    """
    try:
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(None, convert_from_path, pdf_path, skip_first_n_pages + 1, max_pages if max_pages > 0 else None)
        logging.info(f"Converted PDF to {len(images)} images.")
        return images
    except Exception as e:
        logging.error(f"Failed to convert PDF to images: {e}")
        return []

def detect_text_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detects text regions in an image using the initialized text detector.

    Args:
        image (Image.Image): The image to process.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes for detected text regions.
    """
    if not text_detector:
        logging.warning("Text detection is not initialized.")
        return [(0, 0, image.width, image.height)]  # Whole image as one region

    logging.info(f"Detecting text regions using {TEXT_DETECTION_MODEL}.")
    regions = text_detector.detect(image)
    logging.info(f"Detected {len(regions)} text regions.")
    return regions

def classify_text_type(text: str) -> str:
    """
    Classifies the type of text based on heuristics.

    Args:
        text (str): The text to classify.

    Returns:
        str: The classified text type ('Math', 'Handwrite', 'LaTeX', 'Table', 'Figure', or 'Text').
    """
    math_pattern = r'[\$\\](.*?)[\$\\]'  # Detect LaTeX math
    latex_pattern = r'\\[a-zA-Z]+\{.*?\}'  # Detect LaTeX commands
    handwriting_pattern = r'[A-Za-z]{1,}\b'  # Placeholder for handwriting detection
    table_pattern = r'(?:\|.*\|)'  # Detect simple table patterns
    figure_pattern = r'(Figure\s+\d+)|(Fig\.\s+\d+)'  # Detect figure references

    if re.search(latex_pattern, text):
        return "LaTeX"
    elif re.search(math_pattern, text):
        return "Math"
    elif re.search(table_pattern, text):
        return "Table"
    elif re.search(figure_pattern, text):
        return "Figure"
    elif re.search(handwriting_pattern, text):
        return "Handwrite"
    else:
        return "Text"
    
def pytesseract_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs OCR using pytesseract on the given image.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted text and average confidence.
    """
    try:
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        text = ' '.join(data['text'])
        confidences = [int(conf) for conf in data['conf'] if conf.isdigit()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf
    except Exception as e:
        logging.error(f"pytesseract OCR failed: {e}")
        return "", 0.0

def easyocr_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs OCR using EasyOCR on the given image.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted text and average confidence.
    """
    try:
        reader = easyocr.Reader(['en'], gpu=PADDLEOCR_USE_GPU if PADDLE_OCR_AVAILABLE else False)
        result = reader.readtext(np.array(image), detail=1, paragraph=False)
        text = ' '.join([res[1] for res in result])
        confidences = [res[2] for res in result]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf
    except Exception as e:
        logging.error(f"EasyOCR failed: {e}")
        return "", 0.0

def paddleocr_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs OCR using PaddleOCR on the given image.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted text and average confidence.
    """
    if not paddleocr_instance:
        logging.warning("PaddleOCR is not initialized.")
        return "", 0.0
    try:
        result = paddleocr_instance.ocr(np.array(image), rec=True, cls=True)
        text = ' '.join([line[1][0] for line in result])
        confidences = [line[1][1] for line in result]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf
    except Exception as e:
        logging.error(f"PaddleOCR failed: {e}")
        return "", 0.0

def latex_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs LaTeX OCR using pix2tex on the given image.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted LaTeX code and confidence score.
    """
    if not PIX2TEX_OCR_AVAILABLE or not LATEX_OCR_ENABLED:
        logging.warning("LaTeX OCR is not available or disabled.")
        return "", 0.0
    try:
        model = LatexOCR()
        latex_code = model(image)
        # pix2tex does not provide confidence scores, so we'll set a default value
        default_confidence = 100.0
        return latex_code, default_confidence
    except Exception as e:
        logging.error(f"LaTeX OCR failed: {e}")
        return "", 0.0

def math_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs Math OCR using MathPixOCR on the given image.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted math text and confidence score.
    """
    if not MATH_OCR_AVAILABLE:
        logging.warning("Math OCR is not available.")
        return "", 0.0
    try:
        # Assuming mathpixocr.process returns the math text
        math_text = mathpixocr.process(image, MATH_OCR_API_KEY, MATH_OCR_ENDPOINT)
        # MathPixOCR may not provide confidence scores; setting default
        default_confidence = 100.0
        return math_text, default_confidence
    except Exception as e:
        logging.error(f"Math OCR failed: {e}")
        return "", 0.0

def handwrite_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs Handwriting OCR using specialized settings.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted handwritten text and confidence score.
    """
    try:
        # Placeholder for handwriting-specific OCR processing
        # This could involve using a specialized model or service
        # For demonstration, we'll use pytesseract with different configs
        custom_config = r'--oem 1 --psm 7'  # OEM and PSM settings optimized for handwriting
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        text = ' '.join(data['text'])
        confidences = [int(conf) for conf in data['conf'] if conf.isdigit()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf
    except Exception as e:
        logging.error(f"Handwriting OCR failed: {e}")
        return "", 0.0

def table_ocr(image: Image.Image) -> Tuple[str, float]:
    """
    Performs Table OCR using specialized settings or libraries.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted table text and confidence score.
    """
    try:
        # Placeholder for table-specific OCR processing
        # This could involve using libraries like camelot or tabula-py for structured tables
        # For demonstration, we'll use pytesseract with grid recognition
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        lines = {}
        for i, word in enumerate(data['text']):
            if word.strip() != '':
                (x, y, w, h, conf) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['conf'][i])
                lines.setdefault(y, []).append(word)
        sorted_lines = [ ' '.join(words) for y, words in sorted(lines.items()) ]
        text = '\n'.join(sorted_lines)
        confidences = [int(conf) for conf in data['conf'] if conf.isdigit()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return text, avg_conf
    except Exception as e:
        logging.error(f"Table OCR failed: {e}")
        return "", 0.0

async def ocr_image(image: Image.Image) -> Tuple[str, float]:
    """
    Performs unified OCR on the given image using specialized OCRs based on text type.

    Args:
        image (Image.Image): The image to process.

    Returns:
        Tuple[str, float]: Extracted text and average confidence.
    """
    try:
        # Detect text regions
        regions = detect_text_regions(image)

        extracted_text = []
        confidences = []

        for region in regions:
            x, y, w, h = region
            cropped_image = image.crop((x, y, x + w, y + h))
            cropped_np = np.array(cropped_image)

            # Perform OCR based on text type
            text_type = classify_text_type(pytesseract.image_to_string(cropped_image))  # Initial classification
            logging.info(f"Region classified as: {text_type}")

            if text_type == "LaTeX":
                text, conf = latex_ocr(cropped_image)
            elif text_type == "Math":
                text, conf = math_ocr(cropped_image)
            elif text_type == "Handwrite":
                text, conf = handwrite_ocr(cropped_image)
            elif text_type == "Table":
                text, conf = table_ocr(cropped_image)
            elif text_type == "Figure":
                # Assuming figures contain captions or labels; using standard OCR
                text, conf = pytesseract_ocr(cropped_image)
            else:
                # Default OCR
                if OCR_ENGINE.lower() == "pytesseract":
                    text, conf = pytesseract_ocr(cropped_image)
                elif OCR_ENGINE.lower() == "easyocr" and OCR_ENGINE_AVAILABLE:
                    text, conf = easyocr_ocr(cropped_image)
                else:
                    logging.warning(f"Unsupported OCR engine: {OCR_ENGINE}. Falling back to pytesseract.")
                    text, conf = pytesseract_ocr(cropped_image)

            extracted_text.append(text)
            confidences.append(conf)

        combined_text = '\n'.join(extracted_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # If confidence is low, use PaddleOCR as backup
        CONFIDENCE_THRESHOLD = 80.0
        if avg_conf < CONFIDENCE_THRESHOLD and PADDLEOCR_ENABLED and PADDLE_OCR_AVAILABLE:
            logging.info(f"Average OCR confidence ({avg_conf:.2f}) is below threshold ({CONFIDENCE_THRESHOLD}). Using PaddleOCR as backup.")
            backup_text, backup_conf = paddleocr_ocr(image)
            if backup_conf > avg_conf:
                combined_text = backup_text
                avg_conf = backup_conf
                logging.info(f"Switched to PaddleOCR with higher confidence: {avg_conf:.2f}")

        return combined_text, avg_conf
    except Exception as e:
        logging.error(f"Error during OCR processing: {e}")
        return "", 0.0

async def classify_and_process_text(text: str) -> str:
    """
    Classifies the text type and processes it accordingly using LLM.

    Args:
        text (str): The text to process.

    Returns:
        str: Processed text.
    """
    text_type = classify_text_type(text)
    logging.info(f"Classified text as: {text_type}")

    if text_type == "LaTeX" and PIX2TEX_OCR_AVAILABLE and LATEX_OCR_ENABLED:
        try:
            # Convert text back to image if necessary
            # Assuming text contains LaTeX code already extracted
            # If text is image path or similar, adjust accordingly
            # For demonstration, skipping image reconstruction
            latex_text, conf = latex_ocr(Image.new('RGB', (100, 100)))  # Placeholder image
            logging.info("Processed LaTeX text with LaTeX OCR.")
            return latex_text if latex_text else text
        except Exception as e:
            logging.error(f"LaTeX OCR processing failed: {e}")
            return text
    elif text_type == "Math" and MATH_OCR_AVAILABLE:
        try:
            math_text = await asyncio.to_thread(mathpixocr.process, text, MATH_OCR_API_KEY, MATH_OCR_ENDPOINT)
            logging.info("Processed math text with MathOCR.")
            return math_text
        except Exception as e:
            logging.error(f"MathOCR processing failed: {e}")
            return text
    elif text_type == "Handwrite":
        processed_text = await process_handwritten_text(text)
        return processed_text
    elif text_type == "Table":
        # Optionally process tables differently
        processed_text = await correct_ocr_errors(text)
        return processed_text
    else:
        # Standard text processing with OCR and LLM corrections
        corrected_text = await correct_ocr_errors(text)
        return corrected_text

async def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Downloads the required models if they are not already present asynchronously.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: List of model names and download statuses.
    """
    download_status = []
    model_url = "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    model_name = os.path.basename(model_url)
    base_dir = os.path.join(os.getcwd(), 'data')  # Changed to './data'
    models_dir = os.path.join(base_dir, 'models')

    os.makedirs(models_dir, exist_ok=True)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, model_name)

    try:
        async with asyncio.Lock():
            if not os.path.exists(filename):
                logging.info(f"Downloading model {model_name} from {model_url}...")
                async with aiohttp.ClientSession() as session:
                    async with session.get(model_url) as resp:
                        if resp.status == 200:
                            f = await aiofiles.open(filename, mode='wb')
                            await f.write(await resp.read())
                            await f.close()
                            file_size = os.path.getsize(filename) / (1024 * 1024)
                            if file_size < 100:
                                os.remove(filename)
                                status["status"] = "failure"
                                status["message"] = f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                                logging.error(f"Error: {status['message']}")
                            else:
                                logging.info(f"Successfully downloaded: {filename} (Size: {file_size:.2f} MB)")
                        else:
                            status["status"] = "failure"
                            status["message"] = f"Failed to download model. HTTP Status: {resp.status}"
                            logging.error(f"Error: {status['message']}")
            else:
                logging.info(f"Model file already exists: {filename}")
    except Timeout:
        logging.error(f"Error: Could not acquire lock for downloading {model_name}")
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."

    download_status.append(status)
    logging.info("Model download process completed.")
    return [model_name], download_status

def load_model(llm_model_name: str, raise_exception: bool = True):
    """
    Loads the specified LLM model, attempting GPU acceleration first.

    Args:
        llm_model_name (str): Name of the LLM model to load.
        raise_exception (bool, optional): Whether to raise exceptions on failure. Defaults to True.

    Returns:
        Llama or None: Loaded model instance or None if failed.
    """
    global USE_VERBOSE
    try:
        base_dir = os.path.join(os.getcwd(), 'data')  # Changed to './data'
        models_dir = os.path.join(base_dir, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logging.error(f"Error: No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        model_file_path = max(matching_files, key=os.path.getmtime)
        logging.info(f"Loading model: {model_file_path}")
        try:
            logging.info("Attempting to load model with GPU acceleration...")
            model_instance = Llama(
                model_path=model_file_path,
                n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                verbose=USE_VERBOSE,
                n_gpu_layers=-1
            )
            logging.info("Model loaded successfully with GPU acceleration.")
        except Exception as gpu_e:
            logging.warning(f"Failed to load model with GPU acceleration: {gpu_e}")
            logging.info("Falling back to CPU...")
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0
                )
                logging.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logging.error(f"Failed to load model with CPU: {cpu_e}")
                if raise_exception:
                    raise
                return None
        return model_instance
    except Exception as e:
        logging.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_exception:
            raise
        return None

async def generate_completion(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    """
    Generates text completion using the configured LLM provider.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 5000.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    if USE_LOCAL_LLM:
        if API_PROVIDER.upper() == "OLLAMA":
            return await generate_completion_from_ollama(prompt, max_tokens)
        else:
            return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    else:
        logging.error("Local LLM usage is disabled.")
        return None

async def generate_completion_from_ollama(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    """
    Generates text completion using Ollama's API asynchronously.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 5000.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    try:
        logging.info("Generating completion using Ollama...")
        process = await asyncio.create_subprocess_exec(
            "ollama",
            "prompt",
            OLLAMA_MODEL_NAME,
            "--max-tokens",
            str(max_tokens),
            "--prompt",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            output = stdout.decode().strip()
            logging.info(f"Ollama response received. Output length: {len(output):,} characters")
            return output
        else:
            logging.error(f"Ollama error: {stderr.decode().strip()}")
            return None
    except Exception as e:
        logging.error(f"Error while communicating with Ollama: {e}")
        return None

async def generate_completion_from_local_llm(llm_model_name: str, input_prompt: str, number_of_tokens_to_generate: int = 100, temperature: float = 0.7, grammar_file_string: str = None):
    """
    Generates text completion using a local LLM asynchronously.

    Args:
        llm_model_name (str): Name of the LLM model to use.
        input_prompt (str): The input prompt for text generation.
        number_of_tokens_to_generate (int, optional): Number of tokens to generate. Defaults to 100.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        grammar_file_string (str, optional): Grammar file identifier. Defaults to None.

    Returns:
        dict or str: Generated text and related metadata.
    """
    logging.info(f"Starting text completion using model: '{llm_model_name}' for input prompt: '{input_prompt}'")
    llm = await asyncio.to_thread(load_model, llm_model_name)
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(number_of_tokens_to_generate, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = await chunk_text(input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION, llm_model_name)
        results = []
        for chunk in chunks:
            try:
                output = await asyncio.to_thread(
                    llm,
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION,
                    temperature=temperature,
                )
                results.append(output['choices'][0]['text'])
                logging.info(f"Chunk processed. Output tokens: {output['usage']['completion_tokens']:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
        if grammar_file_string_lower:
            list_of_grammar_files = glob.glob(os.path.join(os.getcwd(), 'data', 'grammar_files', '*.gbnf'))  # Changed to './data/grammar_files'
            matching_grammar_files = [x for x in list_of_grammar_files if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
            if len(matching_grammar_files) == 0:
                logging.error(f"No grammar file found matching: {grammar_file_string}")
                raise FileNotFoundError
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            logging.info(f"Loading selected grammar file: '{grammar_file_path}'")
            llama_grammar = await asyncio.to_thread(LlamaGrammar.from_file, grammar_file_path)
            output = await asyncio.to_thread(
                llm,
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                grammar=llama_grammar
            )
        else:
            output = await asyncio.to_thread(
                llm,
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature
            )
        generated_text = output['choices'][0]['text']
        if grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(output['choices'][0]['finish_reason'])
        llm_model_usage_json = json.dumps(output['usage'])
        logging.info(f"Completed text completion in {output['usage']['total_time']:.2f} seconds. Beginning of generated text: \n'{generated_text[:150]}'...")
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "usage": llm_model_usage_json
        }

async def correct_ocr_errors(text: str) -> str:
    """
    Corrects OCR errors in the provided text using LLM.

    Args:
        text (str): The text to correct.

    Returns:
        str: Corrected text.
    """
    logging.info("Starting OCR error correction using LLM...")
    try:
        prompt = f"Correct the following OCR errors in the text:\n\n{text}\n\nCorrected Text:"
        corrected_text = await generate_completion(prompt, max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS)
        return corrected_text if corrected_text else text
    except Exception as e:
        logging.error(f"Error during OCR error correction: {e}")
        return text

async def process_math_formulas(text: str) -> str:
    """
    Processes math formulas in the text using Math OCR and LLM.

    Args:
        text (str): The text containing math formulas.

    Returns:
        str: Text with processed math formulas.
    """
    logging.info("Processing math formulas using Math OCR and LLM...")
    try:
        formula_pattern = r'\$[^$]+\$'  # Simple regex for LaTeX formulas
        formulas = re.findall(formula_pattern, text)
        for formula in formulas:
            if MATH_OCR_AVAILABLE:
                processed_formula = await asyncio.to_thread(mathpixocr.process, formula, MATH_OCR_API_KEY, MATH_OCR_ENDPOINT)
                text = text.replace(formula, processed_formula)
            else:
                logging.warning("Math OCR is not available. Skipping formula processing.")
        return text
    except Exception as e:
        logging.error(f"Error during math formula processing: {e}")
        return text

async def identify_document_structure(text: str) -> str:
    """
    Identifies and structures the document layout using LLM.

    Args:
        text (str): The text to structure.

    Returns:
        str: Structured text with appropriate formatting.
    """
    logging.info("Identifying and structuring document layout using LLM...")
    try:
        prompt = f"Analyze the structure of the following text and format it with appropriate headers, lists, and tables using markdown syntax:\n\n{text}\n\nStructured Text:"
        structured_text = await generate_completion(prompt, max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS)
        return structured_text if structured_text else text
    except Exception as e:
        logging.error(f"Error during document structure identification: {e}")
        return text

async def generate_completion_async(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    """
    Wrapper for generate_completion to ensure it's called asynchronously.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 5000.

    Returns:
        Optional[str]: Generated text or None if failed.
    """
    return await generate_completion(prompt, max_tokens)

async def process_document(extracted_texts: List[str], reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> str:
    """
    Processes the entire document by performing error correction, handwriting recognition,
    math formula processing, LaTeX OCR, and document structuring.

    Args:
        extracted_texts (List[str]): List of extracted texts from OCR.
        reformat_as_markdown (bool): Whether to reformat the text as Markdown.
        suppress_headers_and_page_numbers (bool): Whether to suppress headers and page numbers.

    Returns:
        str: The final processed text.
    """
    logging.info("Starting document processing with LLM...")
    try:
        combined_text = "\n\n".join(extracted_texts)
        
        # Error Correction
        corrected_text = await correct_ocr_errors(combined_text)
        
        # Handwriting Recognition
        handwritten_text = await process_handwritten_text(corrected_text)
        
        # Math OCR Processing
        math_processed_text = await process_math_formulas(handwritten_text)
        
        # LaTeX OCR Processing
        if PIX2TEX_OCR_AVAILABLE and LATEX_OCR_ENABLED:
            latex_processed_text = await classify_and_process_text(math_processed_text)
        else:
            latex_processed_text = math_processed_text
        
        # Document Structure Identification
        structured_text = await identify_document_structure(latex_processed_text)
        
        # Reformat as Markdown if required
        if reformat_as_markdown:
            final_text = await reformat_as_markdown_function(structured_text)
        else:
            final_text = structured_text  # Placeholder for actual reformatting logic
        
        return final_text
    except Exception as e:
        logging.error(f"Error during document processing: {e}")
        return ""

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
        encoding = tiktoken.get_encoding("cl100k_base")  # Example encoding
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logging.error(f"Error estimating tokens: {e}")
        return 0

async def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200, llm_model_name: str = DEFAULT_LOCAL_MODEL_NAME) -> List[str]:
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
        sentences = re.split(r'(?<=[.!?]) +', text)
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

async def main():
    """
    The main function orchestrating the OCR processing workflow.
    """
    try:
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        input_pdf_file_path = '160301289-Warren-Buffett-Katharine-Graham-Letter.pdf'
        max_test_pages = 0
        skip_first_n_pages = 0
        reformat_as_markdown = True
        suppress_headers_and_page_numbers = True

        # Download the model if using local LLM
        if USE_LOCAL_LLM:
            _, download_status = await download_models()
            logging.info(f"Model download status: {download_status}")
            logging.info(f"Using Local LLM with Model: {DEFAULT_LOCAL_MODEL_NAME}")
        else:
            logging.info(f"Using API for completions: {API_PROVIDER}")
            logging.info(f"Using OpenAI model for embeddings: {OPENAI_EMBEDDING_MODEL}")

        base_name = os.path.splitext(os.path.basename(input_pdf_file_path))[0]
        output_extension = '.md' if reformat_as_markdown else '.txt'

        data_dir = os.path.join(os.getcwd(), 'data')  # Changed to './data'
        os.makedirs(data_dir, exist_ok=True)

        raw_ocr_output_file_path = os.path.join(data_dir, f"{base_name}__raw_ocr_output.txt")
        llm_corrected_output_file_path = os.path.join(data_dir, f"{base_name}_llm_corrected{output_extension}")

        list_of_scanned_images = await convert_pdf_to_images(input_pdf_file_path, max_test_pages, skip_first_n_pages)

        extracted_texts = []
        if PROGRESS_TRACKING_ENABLED:
            ocr_tasks = [ocr_image(image) for image in list_of_scanned_images]
            for future in tqdm_asyncio.as_completed(ocr_tasks, desc="Performing OCR on Images"):
                text, conf = await future
                if conf < 80.0:  # Threshold for low confidence
                    corrected_text = await classify_and_process_text(text)
                    extracted_texts.append(corrected_text)
                else:
                    extracted_texts.append(text)
        else:
            for image in list_of_scanned_images:
                text, conf = await ocr_image(image)
                if conf < 80.0:
                    corrected_text = await classify_and_process_text(text)
                    extracted_texts.append(corrected_text)
                else:
                    extracted_texts.append(text)

        raw_ocr_output = "\n\n".join(extracted_texts)
        async with aiofiles.open(raw_ocr_output_file_path, "w") as f:
            await f.write(raw_ocr_output)
        logging.info(f"Raw OCR output written to: {raw_ocr_output_file_path}")

        logging.info("Processing document...")
        final_text = await process_document(extracted_texts, reformat_as_markdown, suppress_headers_and_page_numbers)
        cleaned_text = await asyncio.to_thread(remove_corrected_text_header, final_text)

        # Save the LLM corrected output
        async with aiofiles.open(llm_corrected_output_file_path, 'w') as f:
            await f.write(cleaned_text)
        logging.info(f"LLM Corrected text written to: {llm_corrected_output_file_path}") 

        if final_text:
            logging.info(f"First 500 characters of LLM corrected processed text:\n{final_text[:500]}...")
        else:
            logging.warning("final_text is empty or not defined.")

        logging.info(f"Done processing {input_pdf_file_path}.")
        logging.info("\nSee output files:")
        logging.info(f" Raw OCR: {raw_ocr_output_file_path}")
        logging.info(f" LLM Corrected: {llm_corrected_output_file_path}")

        # Perform a final quality check
        quality_score, explanation = await assess_output_quality(raw_ocr_output, final_text)
        if quality_score is not None:
            logging.info(f"Final quality score: {quality_score}/100")
            logging.info(f"Explanation: {explanation}")
        else:
            logging.warning("Unable to determine final quality score.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        logging.error(traceback.format_exc())

def remove_corrected_text_header(text: str) -> str:
    """
    Removes headers or page numbers from the corrected text.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    # Placeholder for removing headers or page numbers
    # Implement as needed
    return text

async def assess_output_quality(raw_text: str, final_text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Assesses the quality of the output text using LLM.

    Args:
        raw_text (str): The raw OCR extracted text.
        final_text (str): The final processed text.

    Returns:
        Tuple[Optional[int], Optional[str]]: Quality score and explanation.
    """
    logging.info("Assessing output quality using LLM...")
    try:
        prompt = f"Evaluate the quality of the following text on a scale from 0 to 100 and provide a brief explanation:\n\n{final_text}\n\nQuality Score and Explanation:"
        assessment = await generate_completion(prompt, max_tokens=100)
        if assessment:
            parts = assessment.split(':')
            if len(parts) >= 2:
                score_part = parts[0].strip()
                explanation_part = parts[1].strip()
                score = int(re.findall(r'\d+', score_part)[0]) if re.findall(r'\d+', score_part) else None
                explanation = explanation_part
                return score, explanation
        return None, None
    except Exception as e:
        logging.error(f"Error during quality assessment: {e}")
        return None, None

async def reformat_as_markdown_function(text: str) -> str:
    """
    Reformats the text as Markdown using LLM.

    Args:
        text (str): The text to reformat.

    Returns:
        str: Reformatted Markdown text.
    """
    logging.info("Reformatting text as Markdown using LLM...")
    try:
        prompt = f"Convert the following text into well-structured Markdown format:\n\n{text}\n\nMarkdown Format:"
        markdown_text = await generate_completion(prompt, max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS)
        return markdown_text if markdown_text else text
    except Exception as e:
        logging.error(f"Error during Markdown reformatting: {e}")
        return text

if __name__ == '__main__':
    # Run
    asyncio.run(main())