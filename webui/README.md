# PDF to Markdown Web Converter

A Flask-based web application that uses the pdf-craft library to convert PDF files to Markdown format.

## Features

- Batch upload of PDF files
- Real-time progress tracking
- Download converted Markdown files (with images)
- Drag and drop file upload
- Responsive web interface

## Requirements

- Python 3.10 or above
- Flask
- pdf-craft library
- onnxruntime

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create the necessary directories:

```bash
mkdir -p uploads results models
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload PDF files using the web interface

4. Monitor the conversion progress in real-time

5. Download the converted Markdown files when processing is complete

## Directory Structure

- `uploads/`: Temporary storage for uploaded PDF files
- `results/`: Storage for converted Markdown files and images
- `models/`: Storage for AI models used by pdf-craft
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates

## Notes

- The application uses the CPU for PDF processing by default. For GPU acceleration, modify the `device` parameter in the `PDFPageExtractor` initialization in `app.py`.
- Large PDF files may take some time to process.
- The first run will download the required AI models, which may take some time depending on your internet connection.