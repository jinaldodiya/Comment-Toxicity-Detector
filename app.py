import streamlit as st
import pickle

st.set_page_config(page_title="Toxicity Detector", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdfcfb, #e2d1c3);
    }
    .title {
        text-align: center;
        color: #5b4636;
        font-size: 40px;
        font-weight: 700;
    }
    label {
        color: #6b5e4a !important;
        font-size: 16px !important;
    }
    textarea {
        background-color: #ffffff !important;
        color: #3f3f3f !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea {
        color: #3f3f3f !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #d6ccc2, #edede9);
        color: #5b4636;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
    }
    .result-box {
        margin-top: 25px;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        animation: fadeIn 0.8s ease-in-out;
    }
    .toxic {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .nontoxic {
        background-color: #dcfce7;
        color: #166534;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.markdown("<div class='title'>✨ Comment Toxicity Detector</div>", unsafe_allow_html=True)

text = st.text_area("💬 Enter your comment here")

if st.button("🔍 Check "):
    with st.spinner("⏳ Analyzing comment..."):
        vec = vectorizer.transform([text])
        result = model.predict(vec)
        prob = model.predict_proba(vec)

    confidence = round(max(prob[0]) * 100, 2)

    if result[0] == 1:
        st.markdown(f"<div class='result-box toxic'>❌ Toxic Comment<br>Confidence: {confidence}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box nontoxic'>✅ Not Toxic Comment<br>Confidence: {confidence}%</div>", unsafe_allow_html=True)