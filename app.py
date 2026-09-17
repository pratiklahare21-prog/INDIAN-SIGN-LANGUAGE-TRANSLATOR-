import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    import config
except Exception as _e_config:
    print(f"[app.py] Failed to import config.py: {_e_config}")
    config = None


def _lazy_st():
    try:
        import streamlit as _st
        return _st
    except Exception as _e:
        print(f"[app.py] Streamlit is not installed: {_e}")
        return None


def _lazy_src():
    mods = {"HandTracker": None, "AlphabetRecognizer": None,
            "WordRecognizer": None, "SentenceBuilder": None,
            "SpeechEngine": None}
    src_dir = BASE_DIR / "src"
    if src_dir not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from hand_tracker import HandTracker as _HT
        mods["HandTracker"] = _HT
    except Exception as _e:
        print(f"[app.py] HandTracker import soft-fail: {_e}")
    try:
        from alphabet_recognizer import AlphabetRecognizer as _AR
        mods["AlphabetRecognizer"] = _AR
    except Exception as _e:
        print(f"[app.py] AlphabetRecognizer import soft-fail: {_e}")
    try:
        from word_recognizer import WordRecognizer as _WR
        mods["WordRecognizer"] = _WR
    except Exception as _e:
        print(f"[app.py] WordRecognizer import soft-fail: {_e}")
    try:
        from sentence_builder import SentenceBuilder as _SB
        mods["SentenceBuilder"] = _SB
    except Exception as _e:
        print(f"[app.py] SentenceBuilder import soft-fail: {_e}")
    try:
        from speech import SpeechEngine as _SE
        mods["SpeechEngine"] = _SE
    except Exception as _e:
        print(f"[app.py] SpeechEngine import soft-fail: {_e}")
    return mods


def _install_fallback_page(st_inst):
    st_inst.title("🤟 ISL Sign2Speech — Indian Sign Language → Speech")
    st_inst.error("Missing required dependencies.")
    st_inst.markdown(
        """
**To set up the project, please run these steps:**

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Kaggle word dataset** into `dataset/words/` folder.
   See links in `config.py` → `KAGGLE_DATASET_LINKS`.

3. **Run preprocessing scripts:**
   ```bash
   python src/preprocess_alphabet.py
   python src/preprocess_words.py
   ```

4. **Train the models:**
   ```bash
   python src/train_alphabet.py
   python src/train_words.py
   ```

5. **Restart the app:**
   ```bash
   streamlit run app.py
   ```
"""
    )


def _init_session_state(st_inst, src_mods, cfg):
    if "hand_tracker" not in st_inst.session_state:
        st_inst.session_state.hand_tracker = None
        if src_mods["HandTracker"] is not None:
            try:
                st_inst.session_state.hand_tracker = src_mods["HandTracker"]()
            except Exception as _e:
                print(f"[app.py] HandTracker init fail: {_e}")
                st_inst.warning(
                    f"HandTracker could not be initialized (Mediapipe issue: {_e}). "
                    "Webcam hand detection will be unavailable."
                )

    if "alphabet_recognizer" not in st_inst.session_state:
        st_inst.session_state.alphabet_recognizer = None
        if src_mods["AlphabetRecognizer"] is not None:
            try:
                st_inst.session_state.alphabet_recognizer = src_mods["AlphabetRecognizer"]()
            except Exception as _e:
                print(f"[app.py] AlphabetRecognizer init fail: {_e}")

    if "word_recognizer" not in st_inst.session_state:
        st_inst.session_state.word_recognizer = None
        if src_mods["WordRecognizer"] is not None:
            try:
                seq_len = cfg.SEQUENCE_LENGTH if cfg is not None else 60
                st_inst.session_state.word_recognizer = src_mods["WordRecognizer"](seq_len=seq_len)
            except Exception as _e:
                print(f"[app.py] WordRecognizer init fail: {_e}")

    if "sentence_builder" not in st_inst.session_state:
        st_inst.session_state.sentence_builder = None
        if src_mods["SentenceBuilder"] is not None:
            try:
                debounce = cfg.DEBOUNCE_FRAMES if cfg is not None else 15
                max_hist = cfg.MAX_HISTORY_TOKENS if cfg is not None else 20
                st_inst.session_state.sentence_builder = src_mods["SentenceBuilder"](
                    debounce_frames=debounce, max_history=max_hist
                )
            except Exception as _e:
                print(f"[app.py] SentenceBuilder init fail: {_e}")

    if "speech_engine" not in st_inst.session_state:
        st_inst.session_state.speech_engine = None
        if src_mods["SpeechEngine"] is not None:
            try:
                rate = cfg.PYTTSX_RATE if cfg is not None else 175
                vol = cfg.PYTTSX_VOLUME if cfg is not None else 0.9
                st_inst.session_state.speech_engine = src_mods["SpeechEngine"](rate=rate, volume=vol)
            except Exception as _e:
                print(f"[app.py] SpeechEngine init fail: {_e}")


def _rebuild_on_param_change(st_inst, src_mods, debounce_frames, seq_len):
    sb = st_inst.session_state.sentence_builder
    needs_new_sb = False
    if sb is not None and hasattr(sb, "_debounce_frames"):
        if sb._debounce_frames != debounce_frames:
            needs_new_sb = True
    if needs_new_sb and src_mods["SentenceBuilder"] is not None:
        try:
            old_text = sb.get_text() if sb else ""
            old_hist = sb.get_history() if sb else []
            max_hist = config.MAX_HISTORY_TOKENS if config is not None else 20
            new_sb = src_mods["SentenceBuilder"](
                debounce_frames=debounce_frames, max_history=max_hist
            )
            new_sb.set_text(old_text)
            if hasattr(new_sb, "_history"):
                new_sb._history = list(old_hist)
            st_inst.session_state.sentence_builder = new_sb
        except Exception as _e:
            print(f"[app.py] Rebuild SentenceBuilder fail: {_e}")

    wr = st_inst.session_state.word_recognizer
    needs_new_wr = False
    if wr is not None and hasattr(wr, "_seq_len"):
        if wr._seq_len != seq_len:
            needs_new_wr = True
    if needs_new_wr and src_mods["WordRecognizer"] is not None:
        try:
            new_wr = src_mods["WordRecognizer"](seq_len=seq_len)
            st_inst.session_state.word_recognizer = new_wr
        except Exception as _e:
            print(f"[app.py] Rebuild WordRecognizer fail: {_e}")


def _sidebar(st_inst):
    cfg = config
    with st_inst.sidebar:
        st_inst.header("⚙️ Settings")

        mode = st_inst.radio(
            "Recognition Mode",
            ["Alphabet only", "Words only", "Both"],
            index=2,
        )

        alphabet_th_default = cfg.ALPHABET_THRESHOLD if cfg is not None else 0.70
        alphabet_th = st_inst.slider(
            "Alphabet confidence threshold",
            0.0, 1.0, value=alphabet_th_default, step=0.01,
        )

        word_th_default = cfg.WORD_THRESHOLD if cfg is not None else 0.75
        word_th = st_inst.slider(
            "Word confidence threshold",
            0.0, 1.0, value=word_th_default, step=0.01,
        )

        debounce_default = cfg.DEBOUNCE_FRAMES if cfg is not None else 15
        debounce_frames = st_inst.number_input(
            "Debounce frames", min_value=1, max_value=60, value=debounce_default,
        )

        seq_len_default = cfg.SEQUENCE_LENGTH if cfg is not None else 60
        seq_len = st_inst.number_input(
            "Sliding window length", min_value=10, max_value=200, value=seq_len_default,
        )

        st_inst.subheader("🔊 TTS")
        rate_default = cfg.PYTTSX_RATE if cfg is not None else 175
        tts_rate = st_inst.slider(
            "Speech rate (wpm)", 80, 300, value=rate_default,
        )
        vol_default = cfg.PYTTSX_VOLUME if cfg is not None else 0.9
        tts_volume = st_inst.slider(
            "Speech volume", 0.0, 1.0, value=vol_default, step=0.01,
        )
        auto_speak = st_inst.checkbox("Auto-speak on new word", value=False)

        st_inst.subheader("📷 Webcam")
        cam_default = cfg.WEBCAM_INDEX if cfg is not None else 0
        cam_index = st_inst.number_input(
            "Camera index", min_value=0, max_value=10, value=cam_default,
        )
        mirror = st_inst.checkbox("Mirror image", value=True)

    return {
        "mode": mode,
        "alphabet_th": alphabet_th,
        "word_th": word_th,
        "debounce_frames": int(debounce_frames),
        "seq_len": int(seq_len),
        "tts_rate": int(tts_rate),
        "tts_volume": float(tts_volume),
        "auto_speak": auto_speak,
        "cam_index": int(cam_index),
        "mirror": mirror,
    }


def _section_webcam(st_inst, settings, cfg):
    st_inst.header("📷 Webcam")

    ht = st_inst.session_state.hand_tracker
    ar = st_inst.session_state.alphabet_recognizer
    wr = st_inst.session_state.word_recognizer
    sb = st_inst.session_state.sentence_builder
    sp = st_inst.session_state.speech_engine

    setup_ok = True
    reasons = []
    if ht is None:
        setup_ok = False
        reasons.append("HandTracker / Mediapipe unavailable")
    if sb is None:
        setup_ok = False
        reasons.append("SentenceBuilder unavailable")

    run_webcam = st_inst.checkbox("Run Webcam", value=False, key="run_webcam_cb")

    frame_placeholder = st_inst.empty()
    status_placeholder = st_inst.empty()

    if not run_webcam:
        if not setup_ok:
            st_inst.info(
                "Set up required:\n"
                "1) pip install -r requirements.txt, "
                "2) download Kaggle word dataset into dataset/words/, "
                "3) run preprocess + train scripts, "
                "4) restart app."
            )
        else:
            st_inst.info("Tick **Run Webcam** above to start capture.")
        return

    if not setup_ok:
        st_inst.error(
            "Set up required: 1) pip install -r requirements.txt, "
            "2) download Kaggle word dataset into dataset/words/, "
            "3) run preprocess + train scripts, 4) restart app."
        )
        if reasons:
            st_inst.write("Missing components:")
            for r in reasons:
                st_inst.write(f"- {r}")
        return

    try:
        import cv2
    except Exception as _e:
        st_inst.warning(f"OpenCV (cv2) is not installed: {_e}. Install opencv-python.")
        return

    cap = None
    try:
        cap = cv2.VideoCapture(settings["cam_index"])
    except Exception as _e:
        st_inst.warning(f"cv2.VideoCapture failed: {_e}. Retry with a different Camera index.")
        return

    if cap is None or not cap.isOpened():
        st_inst.warning(
            "Could not open the webcam. "
            "Check the camera index, ensure it's not used by another app, and retry."
        )
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return

    try:
        max_frames = 600
        frame_sleep = 1.0 / 20.0
        for _frame_i in range(max_frames):
            if not st_inst.session_state.get("run_webcam_cb", False):
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                status_placeholder.caption("Last frame: (no frame read)")
                time.sleep(frame_sleep)
                continue

            try:
                if settings["mirror"]:
                    frame = cv2.flip(frame, 1)
            except Exception:
                pass

            vec_63 = None
            annotated = frame
            try:
                vec_63, annotated, _detected = ht.process_frame(frame)
            except Exception as _e:
                print(f"[app.py] process_frame error: {_e}")
                continue

            try:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = annotated

            try:
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            except TypeError:
                frame_placeholder.image(rgb, channels="RGB", use_column_width=True)
            status_placeholder.caption(f"Last frame: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            alpha_label, alpha_conf = "", 0.0
            word_label, word_conf = "", 0.0

            if settings["mode"] in ("Alphabet only", "Both"):
                if ar is not None:
                    try:
                        alpha_label, alpha_conf = ar.predict(vec_63, threshold=settings["alphabet_th"])
                    except Exception as _e:
                        print(f"[app.py] Alphabet predict error: {_e}")

            if settings["mode"] in ("Words only", "Both"):
                if wr is not None:
                    try:
                        wr.update(vec_63)
                        word_label, word_conf = wr.predict(threshold=settings["word_th"])
                    except Exception as _e:
                        print(f"[app.py] Word predict error: {_e}")

            if settings["mode"] in ("Alphabet only", "Both"):
                if alpha_label and alpha_conf >= settings["alphabet_th"] and sb is not None:
                    try:
                        added = sb.add_token(alpha_label, alpha_conf)
                        if added and settings["auto_speak"] and sp is not None:
                            try:
                                cur = sb.get_text()
                                stripped = cur.strip()
                                if stripped:
                                    last = stripped.rsplit(" ", 1)[-1]
                                    sp.speak(last, block=False)
                            except Exception:
                                pass
                    except Exception as _e:
                        print(f"[app.py] add alpha token error: {_e}")

            if settings["mode"] in ("Words only", "Both"):
                if word_label and word_conf >= settings["word_th"] and sb is not None:
                    try:
                        added = sb.add_token(word_label, word_conf)
                        if added:
                            try:
                                if hasattr(wr, "reset"):
                                    wr.reset()
                            except Exception:
                                pass
                            if settings["auto_speak"] and sp is not None:
                                try:
                                    cur = sb.get_text()
                                    stripped = cur.strip()
                                    if stripped:
                                        last = stripped.rsplit(" ", 1)[-1]
                                        sp.speak(last, block=False)
                                except Exception:
                                    pass
                    except Exception as _e:
                        print(f"[app.py] add word token error: {_e}")

            time.sleep(frame_sleep)
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def _section_prediction(st_inst, settings):
    st_inst.header("🎯 Live Prediction")

    ar = st_inst.session_state.alphabet_recognizer
    wr = st_inst.session_state.word_recognizer

    col_a, col_b = st_inst.columns(2)

    with col_a:
        st_inst.subheader("Alphabet")
        if settings["mode"] in ("Alphabet only", "Both"):
            status_txt = "N/A"
            if ar is not None:
                try:
                    if hasattr(ar, "status_text"):
                        status_txt = ar.status_text
                except Exception:
                    pass
            st_inst.metric("Label", "—")
            st_inst.progress(0.0)
            st_inst.caption(f"{status_txt}")
        else:
            st_inst.markdown(":gray[N/A (mode: Words only)]")

    with col_b:
        st_inst.subheader("Word")
        if settings["mode"] in ("Words only", "Both"):
            status_txt = "N/A"
            if wr is not None:
                try:
                    if hasattr(wr, "status_text"):
                        status_txt = wr.status_text
                except Exception:
                    pass
            st_inst.metric("Label", "—")
            st_inst.progress(0.0)
            st_inst.caption(f"{status_txt}")
        else:
            st_inst.markdown(":gray[N/A (mode: Alphabet only)]")


def _sentence_text_changed(st_inst):
    sb = st_inst.session_state.sentence_builder
    if sb is None:
        return
    try:
        new_val = st_inst.session_state["sentence_area"]
        sb.set_text(new_val)
    except Exception as _e:
        print(f"[app.py] sentence text change error: {_e}")


def _section_sentence(st_inst, settings):
    st_inst.header("✍️ Sentence")

    sb = st_inst.session_state.sentence_builder
    sp = st_inst.session_state.speech_engine

    current_text = ""
    if sb is not None:
        try:
            current_text = sb.get_text()
        except Exception:
            current_text = ""

    def _on_change():
        _sentence_text_changed(st_inst)

    st_inst.text_area(
        "Current sentence",
        value=current_text,
        key="sentence_area",
        height=120,
        on_change=_on_change,
    )

    c1, c2, c3, c4, c5 = st_inst.columns(5)

    with c1:
        if st_inst.button("␣ Space", use_container_width=True):
            if sb is not None:
                try:
                    sb.add_space()
                except Exception:
                    pass
                st_inst.rerun()

    with c2:
        if st_inst.button("⌫ Del Word", use_container_width=True):
            if sb is not None:
                try:
                    sb.delete_last_word()
                except Exception:
                    pass
                st_inst.rerun()

    with c3:
        if st_inst.button("⌫⌫ Bksp", use_container_width=True):
            if sb is not None:
                try:
                    sb.backspace_char()
                except Exception:
                    pass
                st_inst.rerun()

    with c4:
        if st_inst.button("🗑️ Clear", use_container_width=True):
            if sb is not None:
                try:
                    sb.clear()
                except Exception:
                    pass
                st_inst.rerun()

    with c5:
        if st_inst.button("🔊 Speak", use_container_width=True):
            spoken = False
            if sp is not None and sb is not None:
                try:
                    spoken = sp.speak(sb.get_text(), block=False)
                except Exception as _e:
                    print(f"[app.py] speak error: {_e}")
            if not spoken:
                st_inst.warning(
                    "Speech engine unavailable. Install pyttsx3 and ensure a TTS voice is configured."
                )


def _section_history(st_inst):
    st_inst.header("🕒 History")

    sb = st_inst.session_state.sentence_builder
    if sb is None:
        st_inst.markdown("_No tokens yet._")
        return

    try:
        hist = sb.get_history()
    except Exception:
        hist = []

    if not hist:
        st_inst.markdown("_No tokens yet._")
        return

    rows = []
    for item in hist[-20:]:
        ts = item.get("ts", "")
        try:
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_str = dt.strftime("%H:%M:%S")
            else:
                ts_str = ""
        except Exception:
            ts_str = str(ts)
        rows.append({
            "Time": ts_str,
            "Token": item.get("token", ""),
            "Confidence": round(float(item.get("confidence", 0.0)), 3),
        })

    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["Time", "Token", "Confidence"])
        st_inst.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        for r in rows:
            st_inst.write(f"- [{r['Time']}] {r['Token']}  (conf {r['Confidence']})")


def main():
    st_inst = _lazy_st()
    if st_inst is None:
        print("=" * 60)
        print("ISL Sign2Speech — Streamlit is not installed.")
        print("Please run:")
        print("  pip install streamlit opencv-python numpy mediapipe tensorflow pyttsx3 pandas")
        print("Then:")
        print("  streamlit run app.py")
        print("=" * 60)
        return

    try:
        st_inst.set_page_config(page_title="ISL Sign2Speech", layout="wide")
    except Exception:
        pass

    src_mods = _lazy_src()

    any_core = (src_mods["SentenceBuilder"] is not None or
                src_mods["SpeechEngine"] is not None)
    if not any_core and st_inst is not None:
        _install_fallback_page(st_inst)
        return

    try:
        _init_session_state(st_inst, src_mods, config)
    except Exception as _e:
        print(f"[app.py] session init error: {_e}")

    st_inst.title("🤟 ISL Sign2Speech — Indian Sign Language → Speech")

    settings = _sidebar(st_inst)

    try:
        _rebuild_on_param_change(
            st_inst, src_mods,
            settings["debounce_frames"], settings["seq_len"]
        )
    except Exception as _e:
        print(f"[app.py] param rebuild error: {_e}")

    try:
        sp = st_inst.session_state.speech_engine
        if sp is not None:
            try:
                if hasattr(sp, "set_rate"):
                    sp.set_rate(settings["tts_rate"])
            except Exception:
                pass
            try:
                if hasattr(sp, "set_volume"):
                    sp.set_volume(settings["tts_volume"])
            except Exception:
                pass
    except Exception as _e:
        print(f"[app.py] speech param error: {_e}")

    _section_webcam(st_inst, settings, config)
    _section_prediction(st_inst, settings)
    _section_sentence(st_inst, settings)
    _section_history(st_inst)


if __name__ == "__main__":
    main()
