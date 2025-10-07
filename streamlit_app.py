import streamlit as st
import requests
import base64
import os
from typing import Optional, Tuple, List, Dict
try:
    import ormsgpack  # local server cloning
except Exception:  # noqa: PIE786
    ormsgpack = None
try:
    from fish_audio_sdk import Session as FishSession, TTSRequest, ReferenceAudio
except Exception:  # noqa: PIE786
    FishSession = None  # type: ignore
    TTSRequest = None  # type: ignore
    ReferenceAudio = None  # type: ignore

st.set_page_config(page_title="Fish Audio TTS", page_icon="🐟", layout="centered")

def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def call_tts(
    api_key: str,
    text: str,
    model: str,
    speed: float,
    voice_id: Optional[str],
    ref_audio_b64: Optional[str],
    ref_text: Optional[str],
) -> Tuple[Optional[bytes], str]:
    if not api_key:
        return None, "Missing API key"
    if not text.strip():
        return None, "Enter text"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, object] = {"text": text, "model": model, "speech_speed": speed, "format": "wav"}
    if voice_id:
        payload["voice_id"] = voice_id
    # Prefer references array schema for cloning; fallback to older fields if needed
    primary_payload = dict(payload)
    if ref_audio_b64 and ref_text:
        primary_payload.pop("voice_id", None)
        primary_payload["references"] = [{"audio": ref_audio_b64, "text": ref_text}]
    try:
        # Hint parameters that some backends expect
        payload.setdefault("latency", "normal")
        primary_payload.setdefault("latency", "normal")
        if ref_audio_b64 and ref_text:
            # Attempt 1: JSON with references[] schema
            r = requests.post("https://api.fish.audio/v1/tts", json=primary_payload, headers=headers, timeout=90)
            if r.status_code != 200:
                # Attempt 2: Legacy JSON with reference_audio/reference_text fields
                fallback_payload = dict(payload)
                fallback_payload.pop("voice_id", None)
                fallback_payload["reference_audio"] = ref_audio_b64
                fallback_payload["reference_text"] = ref_text
                r = requests.post("https://api.fish.audio/v1/tts", json=fallback_payload, headers=headers, timeout=90)
                if r.status_code != 200:
                    # Attempt 3: Multipart/form-data with raw audio file
                    try:
                        import io, base64 as _b64
                        audio_bytes = _b64.b64decode(ref_audio_b64)
                        files = {"reference_audio": ("ref.wav", io.BytesIO(audio_bytes), "audio/wav")}
                        data = {
                            "text": text,
                            "model": model,
                            "speech_speed": str(speed),
                            "format": "wav",
                            "reference_text": ref_text,
                            "latency": "normal",
                        }
                        if voice_id:
                            data["voice_id"] = voice_id
                        # Use no JSON header for multipart
                        form_headers = {"Authorization": f"Bearer {api_key}"}
                        r = requests.post("https://api.fish.audio/v1/tts", data=data, files=files, headers=form_headers, timeout=90)
                    except Exception:
                        pass
        else:
            r = requests.post("https://api.fish.audio/v1/tts", json=payload, headers=headers, timeout=90)
        if r.status_code == 200:
            return r.content, ""
        return None, f"API {r.status_code}: {r.text}"
    except Exception as e:
        return None, str(e)

def call_tts_sdk(
    api_key: str,
    text: str,
    reference_id: Optional[str],
    ref_audio_bytes: Optional[bytes],
    ref_text: Optional[str],
) -> Tuple[Optional[bytes], str]:
    if FishSession is None or TTSRequest is None:
        return None, "fish-audio-sdk not installed"
    if not api_key:
        return None, "Missing API key"
    if not text.strip():
        return None, "Enter text"
    try:
        session = FishSession(api_key)
        req_kwargs: Dict[str, object] = {"text": text}
        if reference_id:
            req_kwargs["reference_id"] = reference_id
        if ref_audio_bytes is not None and (ref_text and ref_text.strip()):
            req_kwargs["references"] = [ReferenceAudio(audio=ref_audio_bytes, text=ref_text)]
        req = TTSRequest(**req_kwargs)
        audio_chunks: List[bytes] = []
        for chunk in session.tts(req):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks), ""
    except Exception as e:
        return None, str(e)

def fetch_voices(api_key: str, model: str) -> Tuple[List[Dict[str, str]], str]:
    if not api_key:
        return [], "Missing API key"
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        endpoints = [
            f"https://api.fish.audio/v1/voices?model={model}",
            f"https://api.fish.audio/v1/voices/list?model={model}",
            f"https://api.fish.audio/v1/voice/list?model={model}",
            "https://api.fish.audio/v1/voices",
            "https://api.fish.audio/v1/voices/list",
            "https://api.fish.audio/v1/voice/list",
            "https://api.fish.audio/voices",
        ]
        for url in endpoints:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                voices: List[Dict[str, str]] = []
                raw = data.get("voices", data)
                if isinstance(raw, list):
                    for v in raw:
                        vid = str(v.get("id") or v.get("voice_id") or v.get("key") or "").strip()
                        name = str(v.get("name") or v.get("display_name") or vid).strip()
                        if vid:
                            voices.append({"id": vid, "name": name})
                if voices:
                    return voices, ""
        return [], "No voices available from API (404)"
    except Exception as e:
        return [], str(e)

STATIC_VOICES: List[Dict[str, str]] = [
    {"id": "en_male_1", "name": "English Male 1"},
    {"id": "en_female_1", "name": "English Female 1"},
    {"id": "en_male_2", "name": "English Male 2"},
    {"id": "en_female_2", "name": "English Female 2"},
]

def ui() -> None:
    st.title("🐟 Fish Audio TTS")
    with st.sidebar:
        default_api_key = os.environ.get("FISH_AUDIO_API_KEY", "")
        api_key = st.text_input("API Key", type="password", value=default_api_key)
        model = st.selectbox("Model", ["speech-1.5", "speech-1.6", "s1", "s1-mini"], index=0)
        speech_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        backend = st.radio("Backend", ["Fish Cloud API", "Local Fish Server"], index=0)
        server_url = st.text_input("Local Server URL", value="http://127.0.0.1:8080/v1/tts", disabled=(backend != "Local Fish Server"))
        voice_mode = st.radio("Voice Source", ["Default voice", "Reference tape"], index=0, horizontal=False)
        selected_voice_id: Optional[str] = None
        reference_model_id: Optional[str] = None
        if voice_mode == "Default voice":
            st.markdown("Load and choose from available default voices, or enter a custom voice ID.")
            if "voices" not in st.session_state:
                st.session_state["voices"] = []
            if st.button("Load voices"):
                voices, err = fetch_voices(api_key, model)
                if voices:
                    st.session_state["voices"] = voices
                    st.success(f"Loaded {len(voices)} voices")
                else:
                    st.warning("No voices from API; using built-in presets")
                    st.session_state["voices"] = STATIC_VOICES.copy()
            voices = st.session_state.get("voices", [])
            if voices:
                options = [f"{v['name']} ({v['id']})" for v in voices]
                choice = st.selectbox("Default Voices", options)
                try:
                    idx = options.index(choice)
                    selected_voice_id = voices[idx]["id"]
                except Exception:
                    selected_voice_id = None
            custom_voice = st.text_input("Or custom Voice ID", value="")
            if custom_voice.strip():
                selected_voice_id = custom_voice.strip()
            reference_model_id = st.text_input("Reference Model ID (reference_id)", value="")

    text = st.text_area("Text", placeholder="Type text to synthesize...", height=160)

    col_a, col_b = st.columns(2)
    with col_a:
        ref_audio = st.file_uploader(
            "Reference Audio" + (" (required)" if voice_mode == "Reference tape" else " (optional)"),
            type=["wav", "mp3", "flac", "ogg"]
        )
    with col_b:
        ref_text = st.text_area(
            "Reference Text" + (" (required)" if voice_mode == "Reference tape" else " (optional)"),
            height=120
        )

    if st.button("Generate"):
        if voice_mode == "Reference tape":
            if not ref_audio or not (ref_text and ref_text.strip()):
                st.error("Reference audio and matching reference text are required when using Reference tape")
                return
        ref_b64 = None
        send_voice_id: Optional[str] = None
        send_ref_text = None
        if voice_mode == "Default voice":
            send_voice_id = selected_voice_id or None
        else:
            if ref_audio is not None:
                data = ref_audio.getvalue()
                if len(data) > 10 * 1024 * 1024:
                    st.error("Audio too large (max 10MB)")
                    return
                ref_b64 = to_b64(data)
            send_ref_text = (ref_text or None)
        with st.spinner("Generating..."):
            if backend == "Local Fish Server" and voice_mode == "Reference tape":
                if ormsgpack is None:
                    st.error("Local cloning requires 'ormsgpack'. Install with: pip install ormsgpack")
                    return
                try:
                    import io
                    raw_bytes = ref_audio.getvalue()
                    payload = {
                        "text": text,
                        "references": [{"audio": raw_bytes, "text": send_ref_text or ""}],
                        "format": "wav",
                        "streaming": False,
                        "use_memory_cache": "off",
                        "chunk_length": 300,
                        "max_new_tokens": 0,
                        "top_p": 0.8,
                        "repetition_penalty": 1.1,
                        "temperature": 0.8,
                        "seed": None,
                    }
                    headers = {"content-type": "application/msgpack"}
                    r = requests.post(
                        server_url,
                        data=ormsgpack.packb(payload),
                        headers=headers,
                        timeout=90,
                    )
                    if r.status_code == 200:
                        audio = r.content
                        err = ""
                    else:
                        audio, err = None, f"Local API {r.status_code}: {r.text}"
                except Exception as e:  # noqa: PIE786
                    audio, err = None, str(e)
            else:
                if FishSession is not None:
                    raw_bytes = None
                    if voice_mode == "Reference tape" and ref_audio is not None:
                        raw_bytes = ref_audio.getvalue()
                    audio, err = call_tts_sdk(api_key, text, reference_model_id or send_voice_id, raw_bytes, send_ref_text)
                    if not audio:
                        audio, err = call_tts(api_key, text, model, speech_speed, send_voice_id, ref_b64, send_ref_text)
                else:
                    audio, err = call_tts(api_key, text, model, speech_speed, send_voice_id, ref_b64, send_ref_text)
        if audio:
            st.success("Done")
            st.audio(audio, format="audio/wav")
            st.download_button("Download WAV", data=audio, file_name="tts.wav", mime="audio/wav")
        else:
            st.error(err or "Unknown error")

if __name__ == "__main__":
    ui()