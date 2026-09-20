import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

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
            "WordRecognizer": None, "SentenceRecognizer": None, "SentenceBuilder": None,
            "SpeechEngine": None}
    
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
        from sentence_recognizer import SentenceRecognizer as _SR
        mods["SentenceRecognizer"] = _SR
    except Exception as _e:
        print(f"[app.py] SentenceRecognizer import soft-fail: {_e}")
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

1. **Install Backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install Frontend dependencies:**
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

3. **Download the Kaggle word dataset** into `backend/dataset/words/` folder.
   See links in `backend/config.py` → `KAGGLE_DATASET_LINKS`.

4. **Run preprocessing scripts:**
   ```bash
   cd backend
   python src/preprocess_alphabet.py
   python src/preprocess_words.py
   python src/preprocess_sentences.py
   ```

5. **Train the models:**
   ```bash
   cd backend
   python src/train_alphabet.py
   python src/train_words.py
   python src/train_sentences.py
   ```

6. **Restart the app:**
   ```bash
   cd frontend
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

    if "sentence_recognizer" not in st_inst.session_state:
        st_inst.session_state.sentence_recognizer = None
        if src_mods["SentenceRecognizer"] is not None:
            try:
                seq_len = cfg.SEQUENCE_LENGTH if cfg is not None else 60
                st_inst.session_state.sentence_recognizer = src_mods["SentenceRecognizer"](seq_len=seq_len)
            except Exception as _e:
                print(f"[app.py] SentenceRecognizer init fail: {_e}")

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

    sr = st_inst.session_state.sentence_recognizer
    if sr is not None and getattr(sr, "_seq_len", seq_len) != seq_len and src_mods["SentenceRecognizer"] is not None:
        try:
            st_inst.session_state.sentence_recognizer = src_mods["SentenceRecognizer"](seq_len=seq_len)
        except Exception as _e:
            print(f"[app.py] Rebuild SentenceRecognizer fail: {_e}")


def _sidebar(st_inst):
    cfg = config
    with st_inst.sidebar:
        st_inst.header("⚙️ Settings")

        mode = st_inst.radio(
            "Recognition Mode",
            ["Alphabet only", "Words only", "Sentences only", "Both"],
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
        sentence_th = st_inst.slider(
            "Sentence confidence threshold",
            0.0, 1.0, value=(cfg.SENTENCE_THRESHOLD if cfg is not None else 0.80), step=0.01,
        )

        debounce_default = 8
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
        auto_speak = st_inst.checkbox("Auto-speak on new word", value=True)
        if st_inst.button("🎵 Test Sound Output", use_container_width=True):
            sp = st_inst.session_state.speech_engine
            test_phrase = "Namaste, Indian Sign Language translator sound is working perfectly."
            if sp is not None:
                try:
                    sp.speak(test_phrase, block=False)
                except Exception:
                    pass
            try:
                import json
                import streamlit.components.v1 as components
                js = f"""
                <script>
                try {{
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel();
                        var u = new SpeechSynthesisUtterance({json.dumps(test_phrase)});
                        u.rate = 1.0;
                        u.volume = 1.0;
                        window.speechSynthesis.speak(u);
                    }}
                }} catch(e) {{
                    console.warn(e);
                }}
                </script>
                """
                components.html(js, height=0, width=0)
            except Exception:
                pass
            st_inst.toast("🔊 Playing test voice...")

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
        "sentence_th": sentence_th,
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
    sr = st_inst.session_state.sentence_recognizer
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
                "1) pip install -r requirements.txt (in both backend and frontend), "
                "2) download Kaggle word dataset into backend/dataset/words/, "
                "3) run preprocess + train scripts in backend/, "
                "4) restart app."
            )
        else:
            st_inst.info("Tick **Run Webcam** above to start capture.")
        return

    if not setup_ok:
        st_inst.error(
            "Set up required: 1) pip install -r requirements.txt (in both backend and frontend), "
            "2) download Kaggle word dataset into backend/dataset/words/, "
            "3) run preprocess + train scripts in backend/, 4) restart app."
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

    audio_placeholder = st_inst.empty()

    def _trigger_speech(text_to_speak: str):
        if not text_to_speak:
            return
        clean_text = str(text_to_speak).strip()
        # 1. Server TTS
        if sp is not None:
            try:
                sp.speak(clean_text, block=False)
            except Exception as _e:
                print(f"[app.py] server speak error: {_e}")
        # 2. Browser Web Speech API for 100% audible sound
        try:
            import json
            import streamlit.components.v1 as components
            safe_val = json.dumps(clean_text)
            js = f"""
            <script>
            try {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var u = new SpeechSynthesisUtterance({safe_val});
                    u.rate = 1.0;
                    u.pitch = 1.0;
                    u.volume = 1.0;
                    window.speechSynthesis.speak(u);
                }}
            }} catch(err) {{
                console.warn('TTS error:', err);
            }}
            </script>
            """
            with audio_placeholder:
                components.html(js, height=0, width=0)
        except Exception:
            pass

    # Hold-to-confirm tracker variables
    current_candidate = ""
    candidate_conf = 0.0
    hold_frames = 0
    REQUIRED_HOLD_FRAMES = max(6, int(settings.get("debounce_frames", 8)))
    locked_sign = ""
    last_confirmed_sign = ""
    confirmed_flash_frames = 0

    try:
        max_frames = 1200
        frame_sleep = 1.0 / 22.0
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
                vec_63, annotated, detected = ht.process_frame(frame)
            except Exception as _e:
                print(f"[app.py] process_frame error: {_e}")
                continue

            hands_count = getattr(ht, "last_num_hands", 0)
            active_gest = getattr(ht, "last_gesture", "")
            conf_val = float(getattr(ht, "last_confidence", 0.0))

            alpha_label, alpha_conf = "", 0.0
            word_label, word_conf = "", 0.0
            sentence_label, sentence_conf = "", 0.0

            # 1. Model-based recognizers
            if settings["mode"] in ("Alphabet only", "Both") and ar is not None:
                try:
                    alpha_label, alpha_conf = ar.predict(vec_63, threshold=settings["alphabet_th"])
                except Exception as _e:
                    print(f"[app.py] Alphabet predict error: {_e}")

            if settings["mode"] in ("Words only", "Both") and wr is not None:
                try:
                    wr.update(vec_63)
                    word_label, word_conf = wr.predict(threshold=settings["word_th"])
                except Exception as _e:
                    print(f"[app.py] Word predict error: {_e}")

            if settings["mode"] in ("Sentences only", "Both") and sr is not None:
                try:
                    sr.update(vec_63)
                    sentence_label, sentence_conf = sr.predict(threshold=settings["sentence_th"])
                except Exception as _e:
                    print(f"[app.py] Sentence predict error: {_e}")

            # Use an explicit offline fallback when no trained model exists.
            # Trained model predictions always take precedence.
            if not alpha_label and not word_label and active_gest and conf_val >= 0.70:
                if len(active_gest) == 1 and settings["mode"] in ("Alphabet only", "Both"):
                    if ar is None or not ar.model_available:
                        alpha_label, alpha_conf = active_gest, conf_val
                elif len(active_gest) > 1 and settings["mode"] in ("Words only", "Both"):
                    if wr is None or not wr.model_available:
                        word_label, word_conf = active_gest, conf_val

            # Determine best candidate sign in this frame.
            frame_best_sign = ""
            frame_best_conf = 0.0
            if sentence_label and sentence_conf >= settings["sentence_th"]:
                frame_best_sign = sentence_label
                frame_best_conf = sentence_conf
            elif word_label and word_conf >= settings["word_th"]:
                frame_best_sign = word_label
                frame_best_conf = word_conf
            elif alpha_label and alpha_conf >= settings["alphabet_th"]:
                frame_best_sign = alpha_label
                frame_best_conf = alpha_conf

            # 3. Hold-to-Confirm stability logic
            h_img, w_img = annotated.shape[:2]

            if frame_best_sign:
                if frame_best_sign == current_candidate:
                    hold_frames += 1
                else:
                    current_candidate = frame_best_sign
                    candidate_conf = frame_best_conf
                    hold_frames = 1
                    if current_candidate != locked_sign:
                        locked_sign = ""  # unlock when gesture changes
            else:
                if hold_frames > 0:
                    hold_frames -= 1
                if hold_frames == 0:
                    current_candidate = ""
                    locked_sign = ""

            hold_progress = min(1.0, hold_frames / float(REQUIRED_HOLD_FRAMES))

            # Visual Hold-to-Confirm banner at bottom of frame
            if current_candidate and hold_frames > 0:
                bar_w = int((w_img - 40) * hold_progress)
                # Bottom panel background
                cv2.rectangle(annotated, (15, h_img - 55), (w_img - 15, h_img - 15), (20, 24, 33), -1)
                cv2.rectangle(annotated, (15, h_img - 55), (w_img - 15, h_img - 15), (100, 100, 100), 1)

                if hold_progress >= 1.0 or confirmed_flash_frames > 0:
                    # Confirmed state (Bright Green)
                    cv2.rectangle(annotated, (15, h_img - 55), (w_img - 15, h_img - 15), (0, 180, 80), -1)
                    confirm_text = f"CONFIRMED & SPOKEN: {current_candidate.upper()}"
                    cv2.putText(annotated, confirm_text, (25, h_img - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Progress bar fill (Orange / Gold)
                    cv2.rectangle(annotated, (20, h_img - 50), (20 + bar_w, h_img - 20), (0, 165, 255), -1)
                    hold_text = f"Holding: {current_candidate.upper()} ({int(hold_progress * 100)}%) - Hold 1s to confirm"
                    cv2.putText(annotated, hold_text, (25, h_img - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if confirmed_flash_frames > 0:
                confirmed_flash_frames -= 1

            # 4. Confirmation Trigger when threshold reached
            if hold_progress >= 1.0 and current_candidate and current_candidate != locked_sign:
                locked_sign = current_candidate
                last_confirmed_sign = current_candidate
                confirmed_flash_frames = 8
                if sb is not None:
                    sb.add_token(current_candidate, candidate_conf)
                if settings.get("auto_speak", True):
                    _trigger_speech(current_candidate)

            # Render image
            try:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = annotated

            try:
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            except TypeError:
                frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

            # Status caption
            status_info = f"🕒 `{datetime.now().strftime('%H:%M:%S')}` | 🖐️ **Hands: {hands_count}**"
            if current_candidate:
                status_info += f" | 🎯 Sign: **{current_candidate.upper()}** ({int(candidate_conf * 100)}%)"
                if sr is not None and sr.model_available and " " in current_candidate:
                    status_info += " | 📝 Sentence model"
                elif ((ar is None or not ar.model_available) and (wr is None or not wr.model_available)):
                    status_info += " | ℹ️ Rule-based fallback"
                if hold_progress < 1.0:
                    status_info += f" | ⏳ *Holding {int(hold_progress * 100)}%*"
                else:
                    status_info += " | 🔊 **Confirmed!**"
            if sb is not None:
                cur_s = sb.get_text()
                if cur_s:
                    status_info += f" | ✍️ *\"{cur_s}\"*"
            status_placeholder.markdown(status_info)

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
    sr = st_inst.session_state.sentence_recognizer

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

    if settings["mode"] in ("Sentences only", "Both"):
        sentence_status = sr.status_text if sr is not None else "Sentence model unavailable"
        st_inst.caption(f"Sentence model: {sentence_status}")


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
            sentence_val = sb.get_text() if sb is not None else ""
            if sentence_val.strip():
                # 1. Server TTS
                if sp is not None:
                    try:
                        sp.speak(sentence_val, block=False)
                    except Exception as _e:
                        print(f"[app.py] speak error: {_e}")
                # 2. Browser Web Speech API
                try:
                    import json
                    import streamlit.components.v1 as components
                    js = f"""
                    <script>
                    try {{
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            var u = new SpeechSynthesisUtterance({json.dumps(sentence_val)});
                            u.rate = 1.0;
                            u.volume = 1.0;
                            window.speechSynthesis.speak(u);
                        }}
                    }} catch(e) {{
                        console.warn(e);
                    }}
                    </script>
                    """
                    components.html(js, height=0, width=0)
                except Exception:
                    pass
                st_inst.toast(f"🔊 Spoken: \"{sentence_val}\"")
            else:
                st_inst.info("Sentence is empty. Perform signs or type a message to speak.")


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
        print("  cd frontend")
        print("  pip install -r requirements.txt")
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
