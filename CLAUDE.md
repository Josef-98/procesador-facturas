# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based invoice processing application that uses neural networks to extract data from invoice images. The system combines a simple REST API with a Jupyter notebook containing a sophisticated YOLO-based invoice processing pipeline.

## Core Architecture

- **app.py**: Flask API with a single POST endpoint `/procesar-factura` that accepts JSON with a `nombre` field
- **Red_neuronal_unificada.ipynb**: Main processing notebook containing the complete invoice analysis pipeline using YOLO models, OCR, and image processing
- The system uses two YOLO models: a general model (`best_general.pt`) and a specialized model (`best_extra.pt`) stored in a `./models/` directory
- Results are saved to `./resultados/` directory with timestamped JSON, CSV, and annotated image files

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running the API
```bash
python app.py
```
The API runs on `http://localhost:5000` with debug mode enabled.

### Available Endpoints
- `GET /` - Welcome message
- `POST /procesar-factura` - Processes invoice (expects JSON with `nombre` field)

## Key Components

### Invoice Processing Pipeline
The Jupyter notebook contains the main processing logic:
- Uses YOLO models for object detection on invoice images
- Combines predictions from general and specialized models
- Applies OCR (pytesseract) to extract text from detected regions
- Supports 26 different invoice field types including client data, invoice details, and line items
- Implements confidence thresholds and duplicate filtering
- Outputs structured data in JSON/CSV formats plus annotated images

### Dependencies
- Flask 2.3.3 and Werkzeug 2.3.7 for the API
- YOLO (ultralytics) for object detection
- pytesseract for OCR
- OpenCV and PIL for image processing
- Standard data processing libraries (numpy, pandas, etc.)

### File Structure
- Sample invoice files: `factura133.pdf`, `facturaPrueba.jpg`, `testfile.png`
- Models directory: `./models/` (contains YOLO model files)
- Output directory: `./resultados/` (generated results)
- Processing directory: `./facturas_procesadas/` (alternative output location)

## Testing
Test the API endpoints using curl:
```bash
curl http://localhost:5000/
curl -X POST -H "Content-Type: application/json" -d '{"nombre":"test"}' http://localhost:5000/procesar-factura
```

## Notes
- The application expects YOLO model files in `./models/` directory
- OCR is configured for Spanish language (`lang="spa"`)
- The system processes common invoice fields like client information, line items, totals, and payment details
- Image processing includes adaptive thresholding and margin adjustments for better OCR accuracy