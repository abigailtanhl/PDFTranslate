import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import io
import time
import os

# --- Configuration and Constants ---

MODEL_NAME = "gemini-2.5-flash"
MAX_CHUNK_SIZE = 10000 
API_SLEEP_TIME = 1.0

# Load API key from Streamlit secrets or environment variable
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key) 


# --- Core Functions ---

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_buffer):
    """Extracts text from an uploaded PDF file buffer using PyPDF2."""
    try:
        reader = PdfReader(file_buffer)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error during PDF text extraction: {e}")
        return None

def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Splits a long string into smaller chunks by looking for sentence/paragraph boundaries."""
    if not text:
        return []
    
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')
    
    for p in paragraphs:
        # Check if adding the next paragraph fits in the chunk
        if len(current_chunk) + len(p) + 4 < chunk_size: 
            current_chunk += p + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + '\n\n'
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def translate_chunk(client: genai.Client, chunk: str, target_lang: str) -> str:
    """Sends a single chunk of text to the Gemini API for translation."""
    
    system_instruction = (
        f"You are a professional, high-quality document translator. "
        f"Translate the following text into **{target_lang}**. Maintain the original formatting, "
        f"including paragraph breaks and spacing. Do not add any extra commentary or introductory phrases."
    )
    
    prompt = f"Translate the following document text:\n\n---\n\n{chunk}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except APIError as e:
        st.error(f"Gemini API Error: {e}")
        return f"[Translation Error for this chunk: {e}]"
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return f"[Unexpected Error for this chunk: {e}]"


# --- Streamlit App Layout ---

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Secure PDF Translator (Gemini API)", layout="wide")
    st.title("🔒 Secure PDF Translator with Gemini")
    st.markdown("Upload a document and translate it to English using the free-tier Gemini API.")

    # 1. API Key Setup using Streamlit Secrets
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    
    if not gemini_key:
        st.error("🚨 API Key Missing!")
        st.warning(
            "Please configure your Gemini API Key using Streamlit Secrets. "
            "Create a file at **`.streamlit/secrets.toml`** and add:\n\n"
            "```toml\nGEMINI_API_KEY = \"YOUR_KEY_HERE\"\n```"
        )
        st.stop() # Stop execution if the key is missing

    try:
        # Initialize Gemini Client using the key from st.secrets
        client = genai.Client(api_key=gemini_key)
    except Exception:
        st.error("Failed to initialize the Gemini Client. Check if your API Key is valid.")
        st.stop()

    # 2. File Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    # 3. Translation Button and Logic
    if uploaded_file is not None:
        
        # Display extraction status
        with st.spinner("Extracting text from PDF..."):
            pdf_bytes = uploaded_file.read()
            extracted_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))

        if extracted_text:
            
            st.success(f"Text extracted! Total characters: **{len(extracted_text):,}**")
            
            # Chunk the text for translation
            text_chunks = chunk_text(extracted_text, MAX_CHUNK_SIZE)
            st.info(f"The text has been split into **{len(text_chunks)}** chunks for translation.")
            
            if st.button("Start Translation to English"):
                
                translated_text_parts = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = len(text_chunks)
                
                # Main translation loop
                for i, chunk in enumerate(text_chunks):
                    status_text.text(f"Translating chunk {i + 1} of {total_chunks}...")
                    
                    # Call the translation function
                    translated_chunk = translate_chunk(client, chunk, "English")
                    translated_text_parts.append(translated_chunk)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_chunks)
                    
                    # Throttle API calls to respect rate limits
                    time.sleep(API_SLEEP_TIME)
                
                status_text.success("✅ Translation Complete!")
                final_translated_text = "\n\n---\n\n".join(translated_text_parts)
                
                # --- Display Results ---
                st.subheader("Translated Document (English)")
                st.text_area(
                    "Translated Text",
                    final_translated_text,
                    height=500
                )
                
                # Download Button
                st.download_button(
                    label="Download Translated Text (.txt)",
                    data=final_translated_text.encode('utf-8'),
                    file_name="translated_document.txt",
                    mime="text/plain"
                )
        elif uploaded_file is not None:
            st.error("Could not extract any text from the PDF. It may be an image-only (scanned) document.")

if __name__ == "__main__":
    main()