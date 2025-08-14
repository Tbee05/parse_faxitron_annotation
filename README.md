# Faxitron Final Detector

A powerful tool for detecting and extracting text from yellow annotations in medical images, specifically designed for Faxitron X-ray images.

## Features

- **Yellow Annotation Detection**: Automatically detects yellow-colored regions in medical images
- **Text Extraction**: Uses EasyOCR to extract text content from detected annotations
- **Basic Rectangle Detection**: Identifies main regions and large rectangles containing text
- **Duplicate Removal**: Automatically removes overlapping regions to avoid redundant processing
- **Quality Control Visualizations**: Generates comprehensive QC images showing detection results
- **Batch Processing**: Process single images or entire folders of images
- **Multiple Output Formats**: JSON results with both detailed and simplified outputs

## What It Detects

The detector identifies several types of annotations:
- **Main Regions**: Large rectangular areas (typically containing multiple annotations)
- **Large Rectangles**: Medium-sized rectangular areas (often containing single text strings)
- **Text Annotations**: Small regions containing individual text elements
- **Other Shapes**: Medium/small rectangles, tiny regions, and irregular shapes

## Installation

### Option 1: Using conda (Recommended)

```bash
# Create a new conda environment
conda create -n faxitron python=3.9
conda activate faxitron

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using virtual environment

```bash
# Create a virtual environment
python3 -m venv faxitron_env
source faxitron_env/bin/activate  # On Windows: faxitron_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Direct pip install

```bash
pip install opencv-python-headless Pillow numpy easyocr matplotlib
```

## Usage

### Command Line Interface

The tool provides a flexible CLI for both single image and batch processing:

#### Process a Single Image

```bash
python faxitron_final_detector.py -i /path/to/image.jpg
```

#### Process All Images in a Folder

```bash
python faxitron_final_detector.py -f /path/to/folder
```

#### Specify Custom Output Directory

```bash
python faxitron_final_detector.py -i /path/to/image.jpg -o /custom/output
```

#### Additional Options

```bash
# Enable verbose output
python faxitron_final_detector.py -f /path/to/folder -v

# Skip QC visualizations for faster processing
python faxitron_final_detector.py -f /path/to/folder --no-visualization

# Get help
python faxitron_final_detector.py --help
```

### Demo Mode

Run without arguments to process the default test image:

```bash
python faxitron_final_detector.py
```

## Output

For each processed image, the tool creates:

### Files
- `detection_results.json`: Complete detection results with all annotations
- `simplified_basic_rectangles.json`: Simplified output with just basic rectangles and text
- `qc_visualization.png`: Comprehensive QC visualization
- `simplified_qc_visualization.png`: Focused QC visualization for basic rectangles

### Directory Structure
```
output_directory/
├── image_name_1/
│   ├── detection_results.json
│   ├── simplified_basic_rectangles.json
│   ├── qc_visualization.png
│   └── simplified_qc_visualization.png
├── image_name_2/
│   └── ...
└── ...
```

## Expected Text Format

The tool is designed to extract text in formats like:
- `E5` - Single region identifier
- `E14` - Single region identifier  
- `E21-E24` - Range of region identifiers

## Performance Notes

- **CPU Mode**: Default mode, suitable for most use cases
- **GPU Mode**: For faster processing, ensure CUDA is available and uncomment torch dependencies in requirements.txt
- **Batch Processing**: Processing multiple images sequentially with progress tracking
- **Memory Usage**: Moderate memory usage, scales with image size

## Troubleshooting

### Common Issues

1. **EasyOCR Initialization**: If EasyOCR fails to initialize, the tool will fall back to text estimation
2. **Memory Issues**: For very large images, consider resizing before processing
3. **Dependency Conflicts**: Use the provided conda environment to avoid version conflicts

### Error Messages

- `Image not found`: Check the image path and file permissions
- `No image files found`: Ensure the folder contains supported image formats
- `OCR not available`: EasyOCR failed to initialize, check installation

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

## Requirements

- Python 3.8+
- OpenCV 4.8+
- EasyOCR 1.7+
- Matplotlib 3.5+
- Pillow 9.0+
- NumPy 1.21+

## License

This tool is designed for research and medical imaging applications.

## Citation

If you use this tool in your research, please cite the relevant dependencies:
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- OpenCV: https://opencv.org/
- Matplotlib: https://matplotlib.org/
