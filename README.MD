# Plant Disease Detection System

A comprehensive AI-powered application for identifying plant diseases from leaf images using deep learning technology. Built with Streamlit and TensorFlow, this system provides accurate disease classification to support agricultural decision-making.

## Overview

This application leverages a pre-trained InceptionV3 convolutional neural network to analyze plant leaf images and identify potential diseases. The system supports detection of 38 different plant conditions across various crop types including apples, tomatoes, corn, grapes, and more.

## Demo  
<p align="center">
  <img src="assets/demo1.jpg" width="30%" />
  <img src="assets/demo2.jpg" width="30%" />
  <img src="assets/demo3.jpg" width="30%" />
</p>

## Features

- **Real-time Disease Detection**: Upload leaf images for instant disease classification
- **High Accuracy**: Utilizes InceptionV3 architecture trained on extensive plant disease datasets
- **Multi-crop Support**: Covers 38 disease classes across 14 different plant species
- **Confidence Scoring**: Provides prediction confidence levels to guide decision-making
- **User-friendly Interface**: Clean, intuitive web interface built with Streamlit
- **Professional Styling**: Modern UI with custom CSS and responsive design

## Supported Plants and Diseases

### Apple
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted)
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

### Corn
- Cercospora Leaf Spot / Gray Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

### Additional Crops
- Blueberry, Cherry, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry

## Technical Specifications

- **Framework**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Model Architecture**: InceptionV3
- **Image Processing**: PIL (Python Imaging Library)
- **Input Requirements**: JPG, JPEG, PNG formats
- **Image Resolution**: 224x224 pixels (automatically resized)
- **Model File**: `src/inceptionv3_plant_disease_model.keras`

## Installation and Setup

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Project Setup
1. Clone or download the project repository
2. Navigate to the project directory:
```bash
cd NewProjectTemp
```

3. Create and activate virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Running the Application
1. Ensure you're in the project root directory and virtual environment is activated
2. Execute the application:
```bash
streamlit run src/main.py
```
3. Access the web interface at `http://localhost:8501`

## Usage Instructions

1. **Launch Application**: Run the Streamlit application using the command above
2. **Upload Image**: Click the upload button and select a clear image of a plant leaf
3. **View Results**: The system will display the predicted disease and confidence score
4. **Interpret Results**: Use the confidence percentage to assess prediction reliability

## Model Performance

The InceptionV3-based model provides:
- High accuracy across multiple plant species
- Robust performance on various image qualities
- Fast inference time for real-time applications
- Confidence scoring for prediction reliability assessment

## Applications

- **Agricultural Monitoring**: Early disease detection for crop management
- **Research Support**: Plant pathology research and documentation
- **Educational Tool**: Learning resource for agricultural students and professionals
- **Field Diagnostics**: On-site disease identification for farmers and consultants

## Project Structure

```
NewProjectTemp/
│   .gitignore
│   OpenHere.code-workspace
│   requirements.txt
│   __copy_me__.txt
│
├── .venv/                # Virtual environment
├── .vscode/              # VS Code config
├── src/                  # Source code
│   ├── data_loader.py    # (placeholder / helper for dataset loading)
│   ├── inceptionv3_plant_disease_model.keras   # trained model
│   ├── main.py           # Streamlit app entry point
│   ├── model.ipynb       # training notebook
│   └── __init__.py
└── utils/                # Helper scripts (empty for now)
```

## Development Environment

The project includes VS Code workspace configuration (`OpenHere.code-workspace`) for optimal development experience. The virtual environment (`.venv/`) isolates project dependencies and ensures consistent development across different machines.

## Contributing

This project welcomes contributions for:
- Model improvements and optimization
- Additional plant species and disease classes
- UI/UX enhancements
- Performance optimizations
- Documentation updates

## License

This project is available for educational and research purposes. Please ensure compliance with relevant agricultural and software licensing requirements when deploying in commercial environments.

## Acknowledgments

- TensorFlow and Keras development teams
- Streamlit framework contributors
- Plant disease dataset providers
- Agricultural research community
