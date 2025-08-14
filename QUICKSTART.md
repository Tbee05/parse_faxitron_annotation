# Quick Start Guide

Get the Faxitron Final Detector running in minutes!

## 🚀 Fastest Way (Using conda)

```bash
# 1. Clone or download this repository
# 2. Navigate to the directory
cd /path/to/faxitron_detector

# 3. Run the installation script
./install.sh

# 4. Activate the environment
conda activate faxitron

# 5. Test with demo image
python faxitron_final_detector.py
```

## 🔧 Manual Installation

### Using conda
```bash
conda env create -f environment.yml
conda activate faxitron
```

### Using pip
```bash
python3 -m venv faxitron_env
source faxitron_env/bin/activate  # On Windows: faxitron_env\Scripts\activate
pip install -r requirements.txt
```

## 📸 Test It Out

### Demo Mode (Default Test Image)
```bash
python faxitron_final_detector.py
```

### Process Your Own Image
```bash
python faxitron_final_detector.py -i /path/to/your/image.jpg
```

### Process a Folder of Images
```bash
python faxitron_final_detector.py -f /path/to/your/images/
```

## 📁 What You'll Get

After processing, you'll find:
- **JSON files** with detection results
- **QC visualizations** showing what was detected
- **Simplified output** with just the essential information

## 🆘 Need Help?

- Run `python faxitron_final_detector.py --help` for all options
- Check the full [README.md](README.md) for detailed documentation
- Look at the output files to understand the results

## ⚡ Pro Tips

- Use `--no-visualization` for faster processing when you don't need QC images
- The tool automatically handles multiple image formats (JPG, PNG, TIFF, BMP)
- Results are organized by image name for easy navigation
