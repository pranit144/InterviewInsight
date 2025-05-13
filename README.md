# Technical Interview Analyzer

An AI-powered tool that analyzes technical interviews through video recordings, providing comprehensive feedback on technical knowledge, problem-solving abilities, communication skills, and emotional expressions.

## Features

- Video recording analysis
- Emotion detection
- Audio transcription
- Technical interview assessment
- Comprehensive report generation

## Prerequisites

1. Python 3.8 or higher
2. Required model files:
   - Download the emotion detection model (`model.h5`) from [Kaggle FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
   - The face detection cascade file (`haarcascade_frontalface_default.xml`) is included in the repository

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd technical-interview-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

5. Place the model files:
- Download `model.h5` and place it in the root directory
- Ensure `haarcascade_frontalface_default.xml` is in the root directory

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter candidate information
2. Upload interview recording
3. Wait for analysis to complete
4. View comprehensive results including:
   - Technical assessment
   - Problem-solving evaluation
   - Communication analysis
   - Emotion analysis
5. Download or print the assessment report

## Troubleshooting

If you encounter any issues:

1. Ensure all required files are present:
   - model.h5
   - haarcascade_frontalface_default.xml
   - .env with API keys

2. Check logs for specific error messages

3. Verify video format compatibility (supported: .mp4, .mov, .avi)

## License

MIT License


