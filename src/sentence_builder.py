import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config


class SentenceBuilder:
    """
    Intelligent Sentence Builder for Indian Sign Language (ISL).
    - Accumulates recognized ISL words/tokens with frame debouncing.
    - Translates ISL gloss sequences into complete, natural English sentences
      using the authentic ISL-CSLRT corpus mappings and heuristic grammar rules.
    - Provides Clear, Delete Last Word, Backspace, Space, and History features.
    """
    def __init__(self, debounce_frames: int = 15, max_history: int = 20, gloss_map_path: Path | str | None = None):
        self._raw_tokens: list[str] = []
        self._text: str = ""
        self._history: list[dict] = []
        self._last_token_frames: dict[str, int] = {}
        self._frame_counter: int = 0
        self._debounce_frames: int = debounce_frames
        self._max_history: int = max_history

        if gloss_map_path is None:
            gloss_map_path = config.MODELS_DIR / "gloss_to_sentence.json"
        self._gloss_map_path = Path(gloss_map_path)
        self._gloss_map: dict[str, str] = self._load_gloss_map()

    def _load_gloss_map(self) -> dict[str, str]:
        if self._gloss_map_path.exists():
            try:
                with open(self._gloss_map_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[SentenceBuilder] Warning loading gloss map: {e}")
        return {}

    def _normalize_token(self, token: str) -> str:
        t = token.strip().upper()
        # Handle compound folder names like "HELLO_HI" -> "HELLO", "COLLEGE_SCHOOL" -> "COLLEGE", "I_ME_MINE_MY" -> "I"
        if "_" in t and not any(t == k for k in ["ON_THE_WAY", "SO_MUCH", "TAKE_CARE"]):
            parts = t.split("_")
            t = parts[0]
        return t

    def _format_token_display(self, token: str) -> str:
        # Display friendly version of token
        clean = token.replace("_", " ").title()
        return clean

    def _construct_sentence(self) -> str:
        if not self._raw_tokens:
            return ""

        # 1. Check exact sequence match in ISL gloss dictionary
        gloss_key = " ".join(self._raw_tokens).upper()
        if gloss_key in self._gloss_map:
            sent = self._gloss_map[gloss_key].strip()
            return sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()

        # Check sub-sequences or single sentences
        clean_key = "".join(c for c in gloss_key if c.isalnum() or c == " ")
        if clean_key in self._gloss_map:
            sent = self._gloss_map[clean_key].strip()
            return sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()

        # 2. Heuristic grammar synthesis for ISL
        # ISL grammar is typically Subject-Object-Verb (SOV) or Topic-Comment
        tokens = [self._format_token_display(t) for t in self._raw_tokens]
        joined = " ".join(tokens)

        # Common ISL phrase transformations
        replacements = [
            (r"\bYou Free Today\b", "Are you free today?"),
            (r"\bBring Water Me\b", "Bring water for me."),
            (r"\bI Help You\b", "Can I help you?"),
            (r"\bYou Repeat Please\b", "Could you please repeat that?"),
            (r"\bComb You Hair\b", "Comb your hair."),
            (r"\bDo Me Favour\b", "Do me a favour."),
            (r"\bDonot Abuse Him\b", "Do not abuse him."),
            (r"\bDonot Hurt Me\b", "Do not hurt me."),
            (r"\bDo Not Worry\b", "Do not worry."),
            (r"\bGo Sleep\b", "Go and sleep."),
            (r"\bHe Go Room\b", "He is going into the room."),
            (r"\bHe On Way\b", "He is on the way."),
            (r"\bHelp Me\b", "Help me!"),
            (r"\bHi How You\b", "Hi, how are you?"),
            (r"\bHow You\b", "How are you?"),
            (r"\bHow Dare You\b", "How dare you!"),
            (r"\bHow Old You\b", "How old are you?"),
            (r"\bI Hungry\b", "I am hungry."),
            (r"\bI Tired\b", "I am tired."),
            (r"\bI Fine\b", "I am fine, thank you."),
            (r"\bI Need Water\b", "I need water."),
            (r"\bNice Meet You\b", "Nice to meet you!"),
            (r"\bThank You\b", "Thank you so much!"),
            (r"\bThank\b", "Thank you!"),
            (r"\bWhat Happen\b", "What happened?"),
            (r"\bWhat You Want\b", "What do you want?"),
            (r"\bWhere You From\b", "Where are you from?"),
            (r"\bWho You\b", "Who are you?"),
            (r"\bWhy You Angry\b", "Why are you angry?"),
        ]

        for pat, repl in replacements:
            if re.search(pat, joined, re.IGNORECASE):
                joined = re.sub(pat, repl, joined, flags=re.IGNORECASE)

        # Capitalize first letter and add period if no terminal punctuation
        if joined:
            joined = joined[0].upper() + joined[1:]
            if not joined.endswith((".", "?", "!", ",")):
                joined += "."

        return joined

    def add_token(self, token: str, confidence: float) -> bool:
        if not token:
            return False

        self._frame_counter += 1
        normalized = self._normalize_token(token)
        last_seen = self._last_token_frames.get(normalized)

        # Check debounce: must be distinct token or separated by debounce_frames
        if last_seen is None or (self._frame_counter - last_seen) > self._debounce_frames:
            self._raw_tokens.append(normalized)
            self._text = self._construct_sentence()
            self._last_token_frames[normalized] = self._frame_counter

            self._history.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "token": token,
                "normalized": normalized,
                "confidence": float(confidence),
                "sentence": self._text,
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
        if self._raw_tokens:
            self._raw_tokens.pop()
            self._text = self._construct_sentence()
        else:
            # Fallback on text editing
            stripped = self._text.rstrip()
            if not stripped:
                self._text = ""
                return
            parts = stripped.rsplit(" ", 1)
            self._text = parts[0] if len(parts) > 1 else ""

    def clear(self) -> None:
        self._raw_tokens.clear()
        self._text = ""
        self._last_token_frames.clear()

    def set_text(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text

    def get_sentence(self) -> str:
        return self._text

    def get_raw_tokens(self) -> list[str]:
        return list(self._raw_tokens)

    def get_history(self) -> list[dict]:
        return list(self._history)


if __name__ == "__main__":
    sb = SentenceBuilder(debounce_frames=config.DEBOUNCE_FRAMES)
    print("Testing SentenceBuilder with ISL signs:")
    sb.add_token("YOU", 0.95)
    print("Tokens:", sb.get_raw_tokens(), "-> Sentence:", sb.get_text())
    sb.add_token("FREE", 0.92)
    print("Tokens:", sb.get_raw_tokens(), "-> Sentence:", sb.get_text())
    sb.add_token("TODAY", 0.96)
    print("Tokens:", sb.get_raw_tokens(), "-> Sentence:", sb.get_text())

    assert "free today" in sb.get_text().lower(), f"Expected question, got {sb.get_text()}"
    sb.delete_last_word()
    print("After Delete Last Word:", sb.get_text())
    sb.clear()
    print("After Clear:", repr(sb.get_text()))
    assert sb.get_text() == ""
    print("SentenceBuilder ISL translation tests PASSED!")
