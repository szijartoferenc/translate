import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Streamlit UI beállítások
st.set_page_config(page_title="NLLB Fordító", page_icon="🌍", layout="centered")
st.title("🌍 Meta NLLB Fordító (Magyar ↔ Angol)")

st.markdown("Ez az alkalmazás a Meta AI (Facebook) által készített NLLB modellt használja a fordításhoz – teljesen ingyenesen.")

# 💾 Modell betöltése Hugging Face-ről (egyszer letölti, cache-eli)
@st.cache_resource
def load_nllb_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator = load_nllb_model()

# Nyelvkódok
LANGUAGES = {
    "Magyar → Angol": ("hun_Latn", "eng_Latn"),
    "Angol → Magyar": ("eng_Latn", "hun_Latn")
}

# UI: nyelvpár választása
lang_pair = st.selectbox("Válassz nyelvpárt:", list(LANGUAGES.keys()))
src_lang, tgt_lang = LANGUAGES[lang_pair]

# UI: szövegbevitel
text_input = st.text_area("Írd be a lefordítandó szöveget:", height=150)

if st.button("Fordítás"):
    if not text_input.strip():
        st.warning("Kérlek, adj meg fordítandó szöveget.")
    else:
        with st.spinner("Fordítás folyamatban..."):
            result = translator(text_input, src_lang=src_lang, tgt_lang=tgt_lang)
            translated = result[0]["translation_text"]
            st.success("Fordítás kész!")
            st.text_area("Fordított szöveg:", translated, height=150)

# Lábjegyzet
st.markdown("---")
st.markdown("Készítette **Szijártó Ferenc** • Modell: `facebook/nllb-200-distilled-600M` • Hugging Face + Streamlit")

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
