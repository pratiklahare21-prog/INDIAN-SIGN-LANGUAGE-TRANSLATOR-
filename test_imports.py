"""Quick test to verify frontend can import backend modules."""
import sys
from pathlib import Path

# Simulate frontend import path setup
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

print("=" * 60)
print("Testing ISL Sign2Speech Frontend/Backend Import Structure")
print("=" * 60)

# Test 1: Import config
print("\n[1/7] Testing config import...")
try:
    import config
    print(f"✓ config imported successfully")
    print(f"  BASE_DIR: {config.BASE_DIR}")
    print(f"  MODELS_DIR: {config.MODELS_DIR}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Import hand_tracker
print("\n[2/7] Testing hand_tracker import...")
try:
    from hand_tracker import HandTracker
    print(f"✓ HandTracker imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Import alphabet_recognizer
print("\n[3/7] Testing alphabet_recognizer import...")
try:
    from alphabet_recognizer import AlphabetRecognizer
    print(f"✓ AlphabetRecognizer imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Import word_recognizer
print("\n[4/7] Testing word_recognizer import...")
try:
    from word_recognizer import WordRecognizer
    print(f"✓ WordRecognizer imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Import sentence_recognizer
print("\n[5/7] Testing sentence_recognizer import...")
try:
    from sentence_recognizer import SentenceRecognizer
    print(f"✓ SentenceRecognizer imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Import sentence_builder
print("\n[6/7] Testing sentence_builder import...")
try:
    from sentence_builder import SentenceBuilder
    print(f"✓ SentenceBuilder imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 7: Import speech
print("\n[7/7] Testing speech import...")
try:
    from speech import SpeechEngine
    print(f"✓ SpeechEngine imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL IMPORTS SUCCESSFUL!")
print("=" * 60)
print("\nThe frontend/backend structure is working correctly.")
print("You can now start the frontend with:")
print("  cd frontend")
print("  streamlit run app.py")
print("=" * 60)
