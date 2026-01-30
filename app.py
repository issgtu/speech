import streamlit as st
from google import genai
from google.genai import types
import wave
import io

# --- 1. Page Config (Minimalist) ---
st.set_page_config(page_title="Gemini TTS", page_icon="🗣️")
st.title("Gemini Text-to-Speech")

# --- 2. Sidebar / Setup ---
# We get the API key securely from Streamlit Secrets (see Step 3)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("API Key not found. Please set GEMINI_API_KEY in your secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# --- 3. User Inputs ---
text_input = st.text_area("Enter text:", "Say cheerfully: Have a wonderful day!", height=100)

col1, col2 = st.columns(2)
with col1:
    voice_name = st.selectbox("Voice", ["Kore", "Fenrir", "Puck", "Aoede", "Charon"])
with col2:
    is_pro = st.checkbox("Use Pro Model (Higher Quality)", value=False)

# --- 4. Logic ---
def create_wav_bytes(pcm_data, channels=1, rate=24000, sample_width=2):
    """Helper to convert raw PCM to WAV in memory."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return wav_buffer.getvalue()

if st.button("Generate Audio", type="primary"):
    if not text_input:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating voice..."):
            try:
                model = f"gemini-2.5-{'pro' if is_pro else 'flash'}-preview-tts"
                
                response = client.models.generate_content(
                    model=model,
                    contents=text_input,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                    )
                )

                # Process Audio
                raw_data = response.candidates[0].content.parts[0].inline_data.data
                audio_bytes = create_wav_bytes(raw_data)
                
                # Display Audio Player
                st.audio(audio_bytes, format="audio/wav")

                # Display Cost/Tokens (Minimalist Expander)
                with st.expander("View Usage & Cost Details"):
                    if response.usage_metadata:
                        in_tok = response.usage_metadata.prompt_token_count
                        out_tok = response.usage_metadata.candidates_token_count
                        total = response.usage_metadata.total_token_count
                        
                        # Cost calc: Input: $0.50/1M | Output: $10.00/1M (Approx based on standard rates)
                        cost = ((in_tok / 1_000_000) * 0.50) + ((out_tok / 1_000_000) * 10.00)
                        
                        st.write(f"**Input Tokens:** {in_tok}")
                        st.write(f"**Output Tokens:** {out_tok}")
                        st.write(f"**Estimated Cost:** ${cost:.6f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")