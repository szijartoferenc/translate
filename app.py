import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="NLLB Translator", page_icon="🌍", layout="centered")

# --- Language Selector ---
lang = st.radio("Choose your language", ("Magyar", "English"))

if lang == "Magyar":
    title = "🌍 Meta NLLB Fordító (Magyar ↔ Angol)"
    description = "Ez az alkalmazás a Meta AI (Facebook) által készített NLLB modellt használja a fordításhoz – teljesen ingyenesen."
    lang_pair_text = "Válassz nyelvpárt:"
    src_input_text = "Írd be a lefordítandó szöveget:"
    button_text = "Fordítás"
    success_text = "Fordítás kész!"
    warning_text = "Kérlek, adj meg fordítandó szöveget."
    model_name = "facebook/nllb-200-distilled-600M"
    credits = """
    📘 Modell: `facebook/nllb-200-distilled-600M` \n
    🤖 Készült a Hugging Face Transformers könyvtárral \n
    👨‍💻 Nyílt forráskódú, offline is futtatható (GPU ajánlott)
    """
    footer_text = "Készítette Szijártó Ferenc"
else:
    title = "🌍 Meta NLLB Translator (Hungarian ↔ English)"
    description = "This app uses the Meta AI (Facebook) NLLB model for translation – completely free of charge."
    lang_pair_text = "Choose language pair:"
    src_input_text = "Enter text to be translated:"
    button_text = "Translate"
    success_text = "Translation complete!"
    warning_text = "Please enter some text to translate."
    model_name = "facebook/nllb-200-distilled-600M"
    credits = """
    📘 Model: `facebook/nllb-200-distilled-600M` \n
    🤖 Built with the Hugging Face Transformers library \n
    👨‍💻 Open-source, can be run offline (GPU recommended)
    """
    footer_text = "Created by Szijártó Ferenc"

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_nllb_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator = load_nllb_model()

# --- Language Pair Selection ---
LANGUAGES = {
    "Magyar → Angol": ("hun_Latn", "eng_Latn"),
    "Angol → Magyar": ("eng_Latn", "hun_Latn")
}

lang_pair = st.selectbox(lang_pair_text, list(LANGUAGES.keys()))
src_lang, tgt_lang = LANGUAGES[lang_pair]

# --- Text Input Area ---
text_input = st.text_area(src_input_text, height=150)

if st.button(button_text):
    if not text_input.strip():
        st.warning(warning_text)
    else:
        with st.spinner("Translating..."):
            result = translator(text_input, src_lang=src_lang, tgt_lang=tgt_lang)
            translated = result[0]["translation_text"]
            st.success(success_text)
            st.text_area("Translated text:", translated, height=150)

# --- Custom Footer with Styling ---
st.markdown("---")
st.markdown(
    credits,
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="text-align:center; color:#888;">{footer_text}</p>', unsafe_allow_html=True
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
        body {
            background-color: #f4f7fa;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            padding: 12px;
            font-size: 16px;
        }
        .stTextArea>div>div>textarea {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        /* Eltávolítva: .stSelectbox módosítása, mivel elcsúsztatta a szöveget */
        .stMarkdown {
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True
)