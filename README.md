# ISL Sign2Speech

Real-time Indian Sign Language (ISL) recognition pipeline that translates hand signs into sentences and speaks them aloud.
Runs entirely **local and offline** on your Windows PC — built with MediaPipe Hands, ML models, and a Streamlit web UI.

**Now refactored with separate frontend and backend directories for better code organization!**

---

## 🎯 Features

- **A–Z Fingerspelling Recognition** — static hand poses for every letter
- **Dataset-derived ISL words and sentences** — trained on real ISL video data
- **Sentence Builder** — append recognized words/letters, edit on the fly
- **Text-to-Speech (TTS)** — speaks the full sentence aloud
- **Live Webcam Feed** — low-latency OpenCV capture with MediaPipe landmarks
- **Offline / Privacy-Focused** — no cloud calls, video stays on your machine

---

## 📋 Requirements

| Item | Required |
|---|---|
| **OS** | Windows 10 or Windows 11 (64-bit) |
| **Python** | 3.10, 3.11, or 3.12 — **NOT 3.13+** |
| **Hardware** | Working webcam |
| **Audio** | Speakers / headphones for TTS |
| **Disk** | ~4 GB free for datasets + ~500 MB for packages |
| **RAM** | 8 GB minimum (16 GB recommended) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bat
git clone <your-repo-url>
cd ISL-Sign2Speech
```

### 2. Install Backend Dependencies

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bat
cd ..\frontend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the ISL video dataset from Kaggle:
👉 **https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset**

Extract and place in `backend\dataset\words\` so the structure is:

```
backend\
  dataset\
    words\
      <ClassName1>\
        001.mp4
        002.mp4
        ...
      <ClassName2>\
        ...
```

### 5. Preprocess & Train Models

```bat
cd backend
venv\Scripts\activate

:: Run preprocessing
python src\preprocess_alphabet.py
python src\preprocess_words.py
python src\preprocess_sentences.py

:: Train models
python src\train_alphabet.py
python src\train_words.py
python src\train_sentences.py
```

**Quick preprocessing and training:**
```bat
cd backend
preprocess_all.bat
train_all.bat
```

### 6. Start the Frontend

```bat
cd frontend
venv\Scripts\activate
streamlit run app.py
```

**Or use the convenient launcher:**
```bat
cd frontend
start.bat
```

The app will open at `http://localhost:8501/`

---

## 📁 Project Structure

```
ISL-Sign2Speech\
├── frontend\                       # Streamlit UI application
│   ├── app.py                     # Main Streamlit app
│   ├── requirements.txt           # Frontend dependencies
│   ├── start.bat                  # Windows launcher script
│   └── src\                       # Frontend components (if any)
│
├── backend\                        # ML models, training, and core logic
│   ├── config.py                  # Configuration and paths
│   ├── requirements.txt           # Backend dependencies
│   ├── preprocess_all.bat         # Preprocess all datasets
│   ├── train_all.bat              # Train all models
│   │
│   ├── src\                       # Backend source code
│   │   ├── hand_tracker.py       # MediaPipe hands wrapper
│   │   ├── alphabet_recognizer.py # Alphabet model
│   │   ├── word_recognizer.py    # Word sequence model
│   │   ├── sentence_recognizer.py # Sentence model
│   │   ├── sentence_builder.py   # Token management
│   │   ├── speech.py             # TTS engine
│   │   ├── feature_extraction.py # Feature utilities
│   │   ├── gesture_rules.py      # Rule-based recognition
│   │   ├── preprocess_*.py       # Data preprocessing scripts
│   │   └── train_*.py            # Model training scripts
│   │
│   ├── models\                    # Trained model files
│   │   ├── *.keras / *.joblib    # Model weights
│   │   ├── labels_*.json         # Class mappings
│   │   └── scaler_*.json         # Normalization parameters
│   │
│   ├── data\                      # Processed landmarks
│   │   └── landmarks\            # Preprocessed .npy arrays
│   │
│   ├── dataset\                   # Raw training data
│   │   ├── alphabet\             # A-Z images
│   │   ├── words\                # Word videos
│   │   └── sentences\            # Sentence videos
│   │
│   └── archive\                   # ISL-CSLRT corpus (if present)
│
├── README.md                       # This file
└── .gitignore
```
# 🇮🇳 ISL Sign2Speech

> **Real-Time Indian Sign Language Recognition & Speech Translation System**

ISL Sign2Speech is a real-time **Indian Sign Language (ISL) recognition system** that uses computer vision and machine learning to recognize hand signs, convert them into text/sentences, and speak the recognized sentence aloud.

The project runs **locally and offline on Windows**, using **MediaPipe Hands, machine-learning models, OpenCV, and Streamlit**.

---

## ✨ Features

- 🤟 **A–Z Fingerspelling Recognition**
  - Recognizes static hand poses representing English alphabets.

- 📝 **ISL Word Recognition**
  - Recognizes dataset-derived Indian Sign Language words from video sequences.

- 💬 **Sentence Recognition**
  - Supports sentence-level ISL video processing and recognition.

- 🧩 **Sentence Builder**
  - Combines recognized letters and words into a complete sentence.

- 🔊 **Text-to-Speech**
  - Converts the generated sentence into spoken audio.

- 📷 **Real-Time Webcam Recognition**
  - Uses the webcam for live sign detection.

- ✋ **MediaPipe Hand Tracking**
  - Extracts hand landmarks for machine-learning based recognition.

- 🔒 **Offline & Privacy Focused**
  - Processing is performed locally without sending webcam video to cloud services.

- 🖥️ **Streamlit Web Interface**
  - Provides an interactive interface for real-time translation.

---

## 🧠 How It Works

```text
             Webcam
                │
                ▼
      MediaPipe Hand Tracking
                │
                ▼
       Hand Landmark Extraction
                │
                ▼
        Feature Preprocessing
                │
                ▼
       ┌────────┼────────┐
       │        │        │
       ▼        ▼        ▼
   Alphabet    Word   Sentence
     Model     Model     Model
       │        │        │
       └────────┼────────┘
                ▼
           Prediction
                │
                ▼
         Sentence Builder
                │
                ▼
               Text
                │
                ▼
         Text-to-Speech
                │
                ▼
             🔊 Audio
---

## 🔧 Development Workflow

### Backend Development

```bat
cd backend
venv\Scripts\activate

:: Preprocess data
python src\preprocess_words.py

:: Train a model
python src\train_words.py --epochs 100

:: Test a recognizer
python src\word_recognizer.py
```

### Frontend Development

```bat
cd frontend
venv\Scripts\activate

:: Run the UI
streamlit run app.py

:: Or use the launcher
start.bat
```

---

## 🧱 Architecture

The application follows a modular architecture:

**Frontend (Streamlit)**
- User interface and webcam interaction
- Real-time video display
- Settings and controls
- Sentence editing and TTS playback

**Backend (Python ML)**
- MediaPipe hand tracking
- ML model inference (alphabet, word, sentence)
- Feature extraction and preprocessing
- Data pipeline and training scripts

**Data Flow:**
```
Webcam → MediaPipe → 63-d landmarks → Models → Predictions → UI → TTS
```

---

## 🤔 Troubleshooting

### Backend Issues

**TensorFlow install fails**
- Use Python 3.10–3.12 (NOT 3.13+)
- Install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**Models not loading**
- Ensure you've run all preprocessing and training scripts
- Check that `backend/models/` contains `.joblib` or `.keras` files

**MediaPipe not finding hands**
- Ensure good lighting and plain background
- Keep your palm facing the camera
- Check webcam permissions in Windows settings

### Frontend Issues

**Import errors from backend**
- Ensure backend modules are accessible
- Check that `sys.path` includes backend directory (handled automatically in `app.py`)

**Streamlit won't start**
- Verify frontend dependencies are installed
- Check port 8501 isn't already in use
- Try: `streamlit run app.py --server.port 8502`

**No webcam feed**
- Change `WEBCAM_INDEX` in `backend/config.py` (try 0, 1, or 2)
- Close other apps using the webcam
- Grant camera permission to Python

### General Issues

**Wrong virtual environment**
- Always activate the correct venv:
  - Backend: `backend\venv\Scripts\activate`
  - Frontend: `frontend\venv\Scripts\activate`

**Package conflicts**
- Use separate virtual environments for frontend and backend
- Clear pip cache if needed: `pip cache purge`

---

## 📊 Training Models

All training scripts support these flags:

| Flag | Description |
|---|---|
| `--epochs N` | Set number of training epochs |
| `--quick` | Fast training for testing (1 epoch, limited data) |
| `--landmarks-dir PATH` | Override input landmarks directory |
| `--model-out PATH` | Override output model path |

**Examples:**

```bat
cd backend
venv\Scripts\activate

:: Quick test training
python src\train_words.py --quick

:: Full training with custom epochs
python src\train_words.py --epochs 120

:: Train all models (uses batch scripts)
train_all.bat
```

---

## 🧪 Testing

### Backend Unit Tests
```bat
cd backend
venv\Scripts\activate
python src\hand_tracker.py
python src\word_recognizer.py
```

### Frontend Testing
```bat
cd frontend
venv\Scripts\activate
streamlit run app.py --server.headless true
```

---

## 📝 Configuration

Edit `backend/config.py` to customize:

- **Paths**: Dataset, models, landmarks directories
- **Model architecture**: Hidden units, dropout, LSTM size
- **Training**: Batch size, epochs, learning rate
- **Recognition**: Confidence thresholds, sequence length
- **TTS**: Speech rate, volume
- **Webcam**: Camera index, FPS target

---

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Credits

- **MediaPipe Hands**: © Google LLC (Apache 2.0)
- **Dataset**: [ISL Video Dataset by prasadshet](https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset)
- **ISL-CSLRT Corpus**: Sentence-level ISL data

---

## 🔗 Links

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow Documentation](https://www.tensorflow.org)

---

**Built with ❤️ for the Indian Sign Language community**
