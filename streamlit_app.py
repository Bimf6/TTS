import streamlit as st
import requests
import base64
import os

st.set_page_config(page_title="Fish Audio TTS", page_icon="🐟", layout="centered")

def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def call_tts(api_key: str, text: str, model: str, speed: float, voice_id: str | None, ref_audio_b64: str | None, ref_text: str | None) -> tuple[bytes | None, str]:
    if not api_key:
        return None, "Missing API key"
    if not text.strip():
        return None, "Enter text"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, object] = {"text": text, "model": model, "speech_speed": speed, "format": "wav"}
    if voice_id:
        payload["voice_id"] = voice_id
    if ref_audio_b64 and ref_text:
        payload["reference_audio"] = ref_audio_b64
        payload["reference_text"] = ref_text
    try:
        r = requests.post("https://api.fish.audio/v1/tts", json=payload, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.content, ""
        return None, f"API {r.status_code}: {r.text}"
    except Exception as e:
        return None, str(e)

def ui() -> None:
    st.title("🐟 Fish Audio TTS")
    with st.sidebar:
        default_api_key = os.environ.get("FISH_AUDIO_API_KEY", "")
        api_key = st.text_input("API Key", type="password", value=default_api_key)
        model = st.selectbox("Model", ["speech-1.5", "speech-1.6", "s1", "s1-mini"], index=0)
        speech_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        voice_mode = st.radio("Voice Source", ["Default voice", "Reference tape"], index=0, horizontal=False)
        voice_id = None
        if voice_mode == "Default voice":
            voice_id = st.text_input("Voice ID", help="Choose a default voice ID; leave empty to let the model pick")

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
        send_voice_id = None
        send_ref_text = None
        if voice_mode == "Default voice":
            send_voice_id = (voice_id or None)
        else:
            if ref_audio is not None:
                data = ref_audio.getvalue()
                if len(data) > 10 * 1024 * 1024:
                    st.error("Audio too large (max 10MB)")
                    return
                ref_b64 = to_b64(data)
            send_ref_text = (ref_text or None)
        with st.spinner("Generating..."):
            audio, err = call_tts(api_key, text, model, speech_speed, send_voice_id, ref_b64, send_ref_text)
        if audio:
            st.success("Done")
            st.audio(audio, format="audio/wav")
            st.download_button("Download WAV", data=audio, file_name="tts.wav", mime="audio/wav")
        else:
            st.error(err or "Unknown error")

if __name__ == "__main__":
    ui()