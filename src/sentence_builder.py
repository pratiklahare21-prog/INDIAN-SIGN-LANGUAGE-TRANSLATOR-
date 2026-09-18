from datetime import datetime, timezone

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class SentenceBuilder:
    def __init__(self, debounce_frames: int = 15, max_history: int = 20):
        self._text = ""
        self._history = []
        self._last_token_frames = {}
        self._frame_counter = 0
        self._debounce_frames = debounce_frames
        self._max_history = max_history

    def add_token(self, token: str, confidence: float) -> bool:
        self._frame_counter += 1
        last_seen = self._last_token_frames.get(token)
        if last_seen is None or (self._frame_counter - last_seen) > self._debounce_frames:
            if self._text and not self._text.endswith(" "):
                self._text += " "
            self._text += token
            self._last_token_frames[token] = self._frame_counter
            self._history.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "token": token,
                "confidence": confidence,
            })
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            return True
        return False

    def add_space(self) -> None:
        if not self._text.endswith(" "):
            self._text += " "

    def backspace_char(self) -> None:
        if self._text:
            self._text = self._text[:-1]

    def delete_last_word(self) -> None:
        stripped = self._text.rstrip()
        if not stripped:
            self._text = ""
            return
        parts = stripped.rsplit(" ", 1)
        if len(parts) == 1:
            self._text = ""
        else:
            self._text = parts[0]

    def clear(self) -> None:
        self._text = ""

    def set_text(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text

    def get_history(self) -> list:
        return list(self._history)


if __name__ == "__main__":
    sb = SentenceBuilder(
        debounce_frames=config.DEBOUNCE_FRAMES,
        max_history=config.MAX_HISTORY_TOKENS,
    )

    results = []
    for i in range(5):
        results.append(sb.add_token("Hi", 0.95))
    assert sb.get_text() == "Hi", f"Expected 'Hi', got '{sb.get_text()}'"
    assert results[0] is True, "First add_token should return True"
    assert all(r is False for r in results[1:]), "Subsequent add_token should return False"

    sb.set_text("Hello world")
    sb.delete_last_word()
    assert sb.get_text() == "Hello", f"Expected 'Hello', got '{sb.get_text()}'"

    sb.set_text("Hi")
    sb.backspace_char()
    assert sb.get_text() == "H", f"Expected 'H', got '{sb.get_text()}'"

    sb.clear()
    assert sb.get_text() == "", f"Expected '', got '{sb.get_text()}'"

    print("SentenceBuilder smoke PASSED")
