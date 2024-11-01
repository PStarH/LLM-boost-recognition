# OCR and Voice Recognition Module

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

## Description

The OCR and Voice Recognition Module is a comprehensive tool designed to extract and process text from PDF documents, images, and audio files. Leveraging multiple OCR engines and advanced voice recognition technologies, this module ensures high accuracy and includes features such as error correction using Language Models (LLMs), math formula processing, and document structure identification. Highly configurable and supporting GPU acceleration, it caters to a wide range of applications from document digitization to voice-controlled systems.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [FAQs](#faqs)
- [Contact](#contact)
- [Roadmap](#roadmap)
- [Changelog](#changelog)
- [Demo](#demo)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (e.g., `venv` or `virtualenv`)
- Tesseract OCR installed on your system

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/PStarH/ocr-voice-recognition-module.git
    cd ocr-voice-recognition-module
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
        - Download the installer from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and follow the installation instructions.

5. **Download Additional Models**
    Ensure that the required models for EAST, CRAFT, and LLMs are downloaded and placed in the appropriate directories as specified in the configuration.

6. **Configure Environment Variables**
    Create a `.env` file in the root directory with the following structure:
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

### Run the OCR and Voice Recognition Workflow
```bash
python OCR.py
```

### Parameters

- **Input PDF File**: Specify the path to the PDF file you want to process by updating the `input_pdf_file_path` variable in the `main` function.
- **Reformat as Markdown**: Set `reformat_as_markdown` to `True` to convert the extracted text into Markdown format.
- **Suppress Headers and Page Numbers**: Set `suppress_headers_and_page_numbers` to `True` to remove headers and page numbers from the final output.

### Example
```python
input_pdf_file_path = 'path/to/your/document.pdf'
max_test_pages = 0 # Set to 0 to process all pages
skip_first_n_pages = 0 # Set to skip initial pages if needed
reformat_as_markdown = True
suppress_headers_and_page_numbers = True
```

### Voice Recognition Usage
```bash
python Voice-Recognition.py
```

Configure the input audio file path and other settings in the `main` function as needed.

## Features

- **PDF to Image Conversion**: Converts PDF files to images for OCR processing.
- **Multiple OCR Engines**: Supports `pytesseract`, `EasyOCR`, and `PaddleOCR` as primary and backup OCR engines.
- **Text Detection Models**: Utilizes advanced text detection models like EAST and CRAFT for accurate region identification.
- **Error Correction**: Integrates with LLMs to correct OCR and voice recognition errors, enhancing text quality.
- **Math Formula Processing**: Detects and processes mathematical formulas using specialized OCR tools.
- **Document Structure Identification**: Analyzes and formats the extracted text into structured Markdown.
- **Voice Recognition**: Implements advanced voice recognition with multiple ASR engines and validation mechanisms.
- **GPU Acceleration**: Supports GPU usage for faster processing with compatible models.
- **Asynchronous Processing**: Implements asynchronous operations for efficient handling of large documents and audio files.
- **Progress Tracking**: Provides progress indicators during OCR and processing tasks.
- **Language Support**: Configurable to support multiple languages for OCR and voice recognition.

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

### Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and respectful environment for all contributors.

## License

This project is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) License.

## Acknowledgements

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EAST Text Detector](https://github.com/argman/EAST)
- [CRAFT Text Detector](https://github.com/clovaai/CRAFT-pytorch)
- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [librosa](https://librosa.org/)
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Ollama](https://ollama.com/)
- [PStarH](https://github.com/PStarH)

## FAQs

### 1. **How do I switch between different OCR engines?**
Update the `OCR_ENGINE` variable in your `.env` file to `pytesseract`, `easyocr`, or `paddleocr` based on your preference.

### 2. **Can I use this module without GPU?**
Yes, the module is fully functional on CPU. However, GPU acceleration is available and recommended for faster processing if your system supports it.

### 3. **How do I add support for additional languages?**
Ensure that the required language packs are installed for your chosen OCR engines and update the `SUPPORTED_LANGUAGES` configuration in the `.env` file.

### 4. **What should I do if I encounter an error during installation?**
Check the error logs for specific issues, ensure all prerequisites are met, and verify that all dependencies are correctly installed. Feel free to open an issue on the repository for further assistance.

### 5. **Is there a way to contribute feedback on OCR accuracy?**
Yes, the module includes a feedback mechanism. Refer to the `collect_user_feedback` function in the code for details on how to provide feedback.

## Contact

For support or inquiries, please reach out via [GitHub Issues](https://github.com/PStarH/ocr-voice-recognition-module/issues).

## Roadmap

- **Voice Recognition Integration**: Enhance voice recognition features for improved accessibility and additional input methods.
- **Enhanced Error Handling**: Expand error handling mechanisms to cover more edge cases and provide detailed logging.
- **Support for Additional OCR Engines**: Integrate more OCR engines to increase flexibility and accuracy.
- **Web Interface**: Develop a web-based interface for easier interaction and processing of documents.
- **Real-Time Processing**: Enable real-time OCR and voice recognition processing for live document feeds and audio streams.
- **Multilingual Support**: Expand OCR and voice recognition capabilities to support multiple languages beyond English.
- **User Authentication**: Add authentication mechanisms for secure access to OCR and voice recognition functionalities in shared environments.
- **Cloud Deployment**: Adapt the module for deployment on cloud platforms to leverage scalable resources.
- **API Development**: Create a RESTful API to allow other applications to interact with the OCR and Voice Recognition module programmatically.
- **Performance Optimization**: Continuously optimize the module for faster processing times and reduced resource consumption.

## Changelog

### v1.0.0
- Initial release with OCR and Voice Recognition capabilities.
- Supported OCR engines: pytesseract, EasyOCR, PaddleOCR.
- Integrated text detection models: EAST and CRAFT.
- Implemented error correction using LLMs.
- Added math formula processing.
- Configured GPU acceleration support.

### v1.1.0
- Enhanced error handling and logging mechanisms.
- Added support for additional languages.
- Improved performance optimizations for faster processing.

### v1.2.0
- Integrated new OCR engines and updated existing ones.
- Added real-time processing features.
- Expanded Contributing and Acknowledgements sections.


