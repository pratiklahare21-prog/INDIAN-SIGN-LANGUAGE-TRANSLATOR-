# Setup Verification Guide

This guide helps you verify that the frontend/backend refactoring is working correctly.

## Structure Verification

### ✓ Directory Structure Created

```
ISL-Sign2Speech/
├── frontend/
│   ├── app.py                 ✓ Created
│   ├── requirements.txt       ✓ Created
│   ├── start.bat             ✓ Created
│   └── src/
│       └── __init__.py       ✓ Created
│
├── backend/
│   ├── __init__.py           ✓ Created
│   ├── config.py             ✓ Copied
│   ├── requirements.txt      ✓ Created
│   ├── preprocess_all.bat    ✓ Created
│   ├── train_all.bat         ✓ Created
│   ├── src/                  ✓ Copied (23 Python files)
│   ├── models/               ✓ Copied (10 files)
│   ├── data/                 ✓ Copied (landmarks)
│   ├── dataset/              ✓ Copied (videos)
│   └── archive/              ✓ Copied (corpus)
│
├── README.md                  ✓ Updated
└── test_imports.py           ✓ Created
```

## Step-by-Step Verification

### Step 1: Install Backend Dependencies

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** All packages install successfully (tensorflow, mediapipe, opencv-python, scikit-learn, etc.)

**Verify:**
```bat
python -c "import tensorflow; import mediapipe; import cv2; print('Backend deps OK')"
```

### Step 2: Install Frontend Dependencies

```bat
cd ..\frontend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** Streamlit and UI packages install successfully

**Verify:**
```bat
python -c "import streamlit; import cv2; import numpy; print('Frontend deps OK')"
```

### Step 3: Test Backend Imports

From the project root, run:

```bat
venv\Scripts\python test_imports.py
```

**Expected output:**
```
============================================================
Testing ISL Sign2Speech Frontend/Backend Import Structure
============================================================

[1/7] Testing config import...
✓ config imported successfully
  BASE_DIR: D:\ISL-Sign2Speech\backend
  MODELS_DIR: D:\ISL-Sign2Speech\backend\models

[2/7] Testing hand_tracker import...
✓ HandTracker imported successfully

[3/7] Testing alphabet_recognizer import...
✓ AlphabetRecognizer imported successfully

[4/7] Testing word_recognizer import...
✓ WordRecognizer imported successfully

[5/7] Testing sentence_recognizer import...
✓ SentenceRecognizer imported successfully

[6/7] Testing sentence_builder import...
✓ SentenceBuilder imported successfully

[7/7] Testing speech import...
✓ SpeechEngine imported successfully

============================================================
✓ ALL IMPORTS SUCCESSFUL!
============================================================
```

### Step 4: Verify Backend Config Paths

```bat
cd backend
venv\Scripts\activate
python -c "import config; print('MODELS_DIR:', config.MODELS_DIR); print('DATA_DIR:', config.DATA_DIR)"
```

**Expected output:**
```
MODELS_DIR: D:\ISL-Sign2Speech\backend\models
DATA_DIR: D:\ISL-Sign2Speech\backend\data
```

### Step 5: Test Frontend App Loading

```bat
cd frontend
venv\Scripts\activate
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent / 'backend')); import config; print('Frontend can access backend:', config.BASE_DIR)"
```

**Expected output:**
```
Frontend can access backend: D:\ISL-Sign2Speech\backend
```

### Step 6: Start Frontend (Dry Run)

```bat
cd frontend
venv\Scripts\activate
streamlit run app.py
```

**Expected behavior:**
- Streamlit starts without import errors
- Opens browser at http://localhost:8501
- UI loads with setup instructions (if models not yet trained)
- No Python import errors in terminal

**Stop with:** Ctrl+C

### Step 7: Verify Backend Scripts Work

```bat
cd backend
venv\Scripts\activate

:: Test one preprocessing script
python src\preprocess_words.py --help
```

**Expected:** Script runs and shows help/usage information (or processes data if dataset exists)

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError for backend modules

**Symptom:** `ModuleNotFoundError: No module named 'config'` or similar

**Solution:**
- Ensure you're using the correct virtual environment
- Verify `sys.path` modifications in `frontend/app.py` are correct
- Check that `backend/__init__.py` exists

### Issue 2: Wrong BASE_DIR in config

**Symptom:** File paths pointing to wrong directories

**Solution:**
- `backend/config.py` should have `BASE_DIR = Path(__file__).resolve().parent`
- This makes all paths relative to the backend directory

### Issue 3: Models not found

**Symptom:** "Model file not found" errors

**Solution:**
- Models are now in `backend/models/`
- Run preprocessing and training from the backend directory:
  ```bat
  cd backend
  preprocess_all.bat
  train_all.bat
  ```

### Issue 4: Dataset not found

**Symptom:** "Dataset directory not found"

**Solution:**
- Datasets should be in `backend/dataset/words/` and `backend/dataset/alphabet/`
- Update any hardcoded paths to use `config.DATASET_DIR`

### Issue 5: Streamlit import errors

**Symptom:** Frontend can't import backend modules

**Solution:**
- Check that `frontend/app.py` has the path setup at the top:
  ```python
  BASE_DIR = Path(__file__).resolve().parent.parent
  BACKEND_DIR = BASE_DIR / "backend"
  sys.path.insert(0, str(BACKEND_DIR))
  sys.path.insert(0, str(BACKEND_DIR / "src"))
  ```

## Quick Test Checklist

- [ ] Backend venv created and activated
- [ ] Backend dependencies installed
- [ ] Frontend venv created and activated  
- [ ] Frontend dependencies installed
- [ ] test_imports.py runs successfully
- [ ] Backend config paths correct
- [ ] Frontend can import backend modules
- [ ] Streamlit starts without errors
- [ ] Preprocessing scripts accessible from backend/
- [ ] Training scripts accessible from backend/

## Success Criteria

**The refactoring is successful when:**

1. ✅ Backend and frontend have separate virtual environments
2. ✅ Backend and frontend have separate requirements.txt files
3. ✅ All backend code (models, training, preprocessing) is in backend/
4. ✅ Frontend UI code is in frontend/
5. ✅ Frontend can import and use backend modules via sys.path
6. ✅ Config paths are relative to backend directory
7. ✅ All scripts run from their respective directories
8. ✅ Streamlit app starts successfully
9. ✅ No duplicate code between frontend and backend
10. ✅ README provides clear setup instructions

## Next Steps After Verification

Once verification passes:

1. **Train models** (if not already trained):
   ```bat
   cd backend
   preprocess_all.bat
   train_all.bat
   ```

2. **Start the application**:
   ```bat
   cd frontend
   start.bat
   ```

3. **Test full workflow**:
   - Enable webcam
   - Perform sign language gestures
   - Verify recognition works
   - Test TTS output
   - Check sentence building

## Rollback Plan

If issues arise, the original code is still in the root directory:
- `app.py` (original)
- `config.py` (original)
- `src/` (original)
- `models/` (original)
- `data/` (original)

You can continue using the original structure until issues are resolved.
