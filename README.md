# OCR Module

## Description

The OCR (Optical Character Recognition) module is a comprehensive tool designed to extract text from PDF documents and images. It leverages multiple OCR engines to ensure high accuracy and includes advanced features such as error correction using Language Models (LLMs), math formula processing, and document structure identification. This module is highly configurable and supports GPU acceleration for enhanced performance.

## Features

- **PDF to Image Conversion**: Converts PDF files to images for OCR processing.
- **Multiple OCR Engines**: Supports `pytesseract`, `EasyOCR`, and `PaddleOCR` as primary and backup OCR engines.
- **Text Detection Models**: Utilizes advanced text detection models like EAST and CRAFT for accurate region identification.
- **Error Correction**: Integrates with LLMs to correct OCR errors and improve text quality.
- **Math Formula Processing**: Detects and processes mathematical formulas using specialized OCR tools.
- **Document Structure Identification**: Analyzes and formats the extracted text into structured Markdown.
- **GPU Acceleration**: Supports GPU usage for faster processing with compatible models.
- **Asynchronous Processing**: Implements asynchronous operations for efficient handling of large documents.
- **Progress Tracking**: Provides progress indicators during OCR and processing tasks.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ocr-module.git
   cd ocr-module
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - **Ubuntu**
     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
     ```
   - **macOS**
     ```bash
     brew install tesseract
     ```
   - **Windows**
     - Download the installer from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and follow installation instructions.

5. **Download Additional Models**
   - Ensure that the required models for EAST, CRAFT, and LLMs are downloaded and placed in the appropriate directories as specified in the configuration.

## Configuration

The module uses environment variables for configuration. Create a `.env` file in the root directory with the following structure:

```env
USE_LOCAL_LLM=True
API_PROVIDER=OLLAMA
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL_NAME=ggml-gpt4all-j-v1.3-groovy
CLAUDE_MODEL_STRING=claude-3-haiku-20240307
MATH_OCR_API_KEY=your_math_ocr_api_key
MATH_OCR_ENDPOINT=your_math_ocr_endpoint
LLM_ERROR_CORRECTION_MODEL=Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf
LLM_LAYOUT_MODEL=Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf
PREPROCESSING_ENABLED=True
PROGRESS_TRACKING_ENABLED=True
OCR_ENGINE=pytesseract
PADDLEOCR_ENABLED=True
PADDLEOCR_LANGUAGE=en
PADDLEOCR_USE_GPU=False
TEXT_DETECTION_MODEL=EAST
TEXT_DETECTION_THRESHOLD=0.5
```

**Note**: Replace placeholder values with your actual configuration details.

## Usage

Run the OCR processing workflow using the following command:

```bash
python OCR.py
```

### Parameters

- **Input PDF File**: Specify the path to the PDF file you want to process. Update the `input_pdf_file_path` variable in the `main` function.
- **Reformat as Markdown**: Set `reformat_as_markdown` to `True` to convert the extracted text into Markdown format.
- **Suppress Headers and Page Numbers**: Set `suppress_headers_and_page_numbers` to `True` to remove headers and page numbers from the final output.

### Example

```python
input_pdf_file_path = 'path/to/your/document.pdf'
max_test_pages = 0  # Set to 0 to process all pages
skip_first_n_pages = 0  # Set to skip initial pages if needed
reformat_as_markdown = True
suppress_headers_and_page_numbers = True
```

## Dependencies

- **Python Libraries**
  - `aiohttp`
  - `aiofiles`
  - `pdf2image`
  - `pytesseract`
  - `llama_cpp`
  - `tiktoken`
  - `numpy`
  - `Pillow`
  - `decouple`
  - `opencv-python`
  - `filelock`
  - `transformers`
  - `tqdm`

- **OCR Engines**
  - `Tesseract OCR`
  - `EasyOCR` (optional)
  - `PaddleOCR` (optional)

- **Text Detection Models**
  - `EAST`
  - `CRAFT`

- **Language Models**
  - Compatible LLMs for error correction and document structuring.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add your feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

Please ensure that your code follows the project's coding standards and includes appropriate documentation.


## Future Work

- **Voice Recognition Integration**: Implement voice recognition features to enhance accessibility and provide additional input methods.
- **Enhanced Error Handling**: Improve error handling mechanisms to cover more edge cases and provide detailed logging.
- **Support for Additional OCR Engines**: Integrate more OCR engines to increase flexibility and accuracy.
- **Web Interface**: Develop a web-based interface for easier interaction and processing of documents.
- **Real-Time Processing**: Enable real-time OCR processing for live document feeds.
- **Multilingual Support**: Expand OCR capabilities to support multiple languages beyond English.
- **User Authentication**: Add authentication mechanisms for secure access to OCR functionalities in shared environments.
- **Cloud Deployment**: Adapt the module for deployment on cloud platforms to leverage scalable resources.
- **API Development**: Create a RESTful API to allow other applications to interact with the OCR module programmatically.
- **Performance Optimization**: Continuously optimize the module for faster processing times and reduced resource consumption.
