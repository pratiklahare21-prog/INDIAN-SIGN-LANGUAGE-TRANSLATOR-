import logging
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class SpeechEngine:
    def __init__(self, rate=None, volume=None):
        self._available = False
        self._rate = rate if rate is not None else config.PYTTSX_RATE
        self._volume = volume if volume is not None else config.PYTTSX_VOLUME
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None

        try:
            import pyttsx3  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            return

        # Start dedicated speech worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as e:
            logger.warning("Failed to initialize pyttsx3 worker: %s", e)
            self._available = False
            return

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            cmd, val = item
            try:
                if cmd == "speak":
                    engine.setProperty("rate", self._rate)
                    engine.setProperty("volume", self._volume)
                    engine.say(val)
                    engine.runAndWait()
                elif cmd == "rate":
                    self._rate = val
                    engine.setProperty("rate", val)
                elif cmd == "volume":
                    self._volume = val
                    engine.setProperty("volume", val)
                elif cmd == "stop":
                    engine.stop()
            except Exception as e:
                logger.warning("Speech worker execution error: %s", e)
            finally:
                self._queue.task_done()

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str, block: bool = False) -> bool:
        if not text or not self._available:
            return False
        logger.info("Speaking: %s", text)
        self._queue.put(("speak", str(text).strip()))
        if block:
            self._queue.join()
        return True

    def stop(self) -> None:
        if self._available:
            self._queue.put(("stop", None))
            if self._worker_thread is not None and self._worker_thread is not threading.current_thread():
                self._worker_thread.join(timeout=3.0)
            self._worker_thread = None
            self._available = False

    def set_rate(self, wpm: int) -> None:
        self._rate = wpm
        if self._available:
            self._queue.put(("rate", wpm))

    def set_volume(self, v: float) -> None:
        v = max(0.0, min(1.0, v))
        self._volume = v
        if self._available:
            self._queue.put(("volume", v))

    def speak_last_word(self, sentence_text: str) -> bool:
        stripped = sentence_text.strip()
        if not stripped:
            return False
        parts = stripped.rsplit(" ", 1)
        last_word = parts[-1] if parts else stripped
        return self.speak(last_word)


if __name__ == "__main__":
    engine = SpeechEngine()
    if not engine.is_available():
        print("SpeechEngine smoke (no real audio): PASSED")
    else:
        assert engine.is_available() is True
        engine.set_rate(150)
        engine.set_volume(0.5)
        assert engine.speak("", block=True) is False
        print("SpeechEngine smoke PASSED")
