# Automatic-Confidence-and-Public-Speaking-Evaluator
This project is an advanced AI-driven system that evaluates video presentations and speeches across multiple communication dimensions — including body gestures, speech sentiment, audio clarity, and presentation relevance. It combines deep learning, natural language processing, and computer vision to give presenters real-time, quantitative feedback.

🚀 Features
🔊 Speech & Sentiment Analysis
Extracts audio from video files using Audio_Extractor.py.

Performs Voice Activity Detection and Sentiment Analysis using VADER in Speech_Sentiment_Analyzer.py.

Analyzes speech emotion and tone to assess audience engagement.

🧍‍♂️ Gesture & Body Language Evaluation
gesture.py uses MediaPipe Pose to track and score gesture dynamics.

Detects confidence, openness, and expressiveness in body movements.

📄 Presentation Content & Relevance Scoring
subject_pdf_generator.py and decision.py handle rubric-based evaluation.

Evaluates slide content, transcript clarity, and topic alignment.

Final decision logic merges gesture, speech, and content scores.

📊 Multimodal Feedback & Scoring
Combines gesture analysis, sentiment score, and content relevance.

Generates a comprehensive performance report in PDF.

🌐 Flask-Based Web Interface
application.py serves a web-based dashboard.

Users can upload a video and receive detailed evaluation results.

Uses templates/ for UI and static/ for media handling.

🧠 Tech Stack
Area	Tools & Frameworks
Programming Language	Python 3.x
Machine Learning	TensorFlow, Scikit-learn, VADER Sentiment
Audio Processing	Whisper, PyDub, Silero VAD
Computer Vision	MediaPipe, OpenCV
Web Framework	Flask
Frontend	HTML, CSS, JS (via templates/static)
Report Generation	FPDF (PDF Generation)

🗂 Project Structure
php
Copy
Edit
vid_predict/
│
├── application.py                # Flask app entry point
├── gesture.py                    # Gesture analysis using MediaPipe
├── Speech_Sentiment_Analyzer.py  # Sentiment scoring via VADER
├── Audio_Extractor.py            # Audio extraction from video
├── decision.py                   # Logic for combining multiple scores
├── subject_pdf_generator.py      # Generates result PDF
├── test.py                       # CLI tester
├── vid_predict.py                # Core video prediction pipeline
├── requirements.txt              # Python dependencies
├── static/                       # CSS/JS/assets
├── templates/                    # HTML templates
└── Model/                        # Pretrained model(s)

🧪 How to Run

🔧 Setup

git clone https://github.com/your-username/AutoCon.git

cd AutoCon

pip install -r requirements.txt

▶️ Run Locally

python application.py
Open your browser and go to http://localhost:5000 to access the UI.

🧪 Run Tests

python test.py
