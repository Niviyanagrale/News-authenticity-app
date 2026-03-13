import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import pandas as pd

true_df = pd.read_csv("dataset/True.csv", encoding="ISO-8859-1", engine="python")
fake_df = pd.read_csv("dataset/Fake.csv", encoding="ISO-8859-1", engine="python")


# Load Model & Vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Page Config
st.set_page_config(page_title="Fake News Detection", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
    .result-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        margin-top: 20px;
    }
    .badge {
        display: inline-block;
        padding: 12px 14px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .badge-real { background-color: #2ecc71; }
    .badge-fake { background-color: #e74c3c; }
    .keyword-chip {
        display: inline-block;
        background: #f0f0f0;
        padding: 6px 10px;
        margin: 4px;
        border-radius: 10px;
        font-size: 14px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# History Store
if "history" not in st.session_state:
    st.session_state.history = []

# Centered Logo
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='logo.png' width='160'>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- Header ----------------
st.title("📰 Fake News Detection System")
st.write("A system that classifies news as **REAL** or **FAKE**, with confidence scores and keyword analysis.")

st.subheader(" Quick Test with Random News")

colA, colB = st.columns(2)

with colA:
    if st.button("Random REAL News"):
        sample = true_df.sample(1).iloc[0]["text"]
        st.session_state["random_text"] = sample

with colB:
    if st.button("Random FAKE News"):
        sample = fake_df.sample(1).iloc[0]["text"]
        st.session_state["random_text"] = sample


# ---------------- Input Area ----------------
text = st.text_area("Enter a news article or headline:", height=180)

# ---------------- Prediction Logic ----------------
if st.button("Analyze News "):
    if text.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        tfidf_input = vectorizer.transform([text])
        prediction = model.predict(tfidf_input)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(tfidf_input)[0]
            confidence = float(np.max(proba) * 100)
        else:
            confidence = None

        feature_names = vectorizer.get_feature_names_out()
        dense_vec = tfidf_input.toarray()[0]
        top_idx = dense_vec.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_idx]

        record = {
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "keywords": top_words
        }
        st.session_state.history.append(record)

        # ---------------- Output Card ----------------
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if prediction == "REAL":
            st.markdown("<div class='badge badge-real'>✔ REAL NEWS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge badge-fake'>✖ unauthentic NEWS</div>", unsafe_allow_html=True)

        st.write(f"### Confidence: **{confidence:.2f}%**")
        st.progress(confidence / 100)

        st.write("###  Top Keywords:")
        for w in top_words:
            st.markdown(f"<span class='keyword-chip'>{w}</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if len(text.split()) < 10:
            st.info("⚠️ Short texts reduce reliability. Use longer news for accurate results.")

# ---------------- SIDE-BY-SIDE PANEL ----------------
st.write("---")
st.subheader("📊 Real vs unauthentic Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("### ✔ REAL News Characteristics")
    st.write("- Contains factual references")
    st.write("- Uses neutral or formal language")
    st.write("- Structured paragraphs")
    st.write("- Mentions credible sources")

with col2:
    st.write("### ✖ FAKE News Characteristics")
    st.write("- Uses emotional or dramatic language")
    st.write("- Often lacks verified sources")
    st.write("- Exaggerated claims")
    st.write("- Sensational keywords")

# ---------------- HISTORY PANEL ----------------
st.write("---")
st.subheader(" Prediction History")

if len(st.session_state.history) == 0:
    st.write("No predictions made yet.")
else:
    for item in st.session_state.history[::-1]:
        with st.expander(item["prediction"] + " — " + f"{item['confidence']:.2f}% confidence"):
            st.write(item["text"])
            st.write("**Keywords:** " + ", ".join(item["keywords"]))

# ---------------- PIE CHART OF REAL vs FAKE ----------------
# st.write("---")
# st.subheader(" Distribution of Predictions (REAL vs FAKE)")

# if len(st.session_state.history) == 0:
#     st.write("No predictions yet to display a chart.")
# else:
#     real_count = sum(1 for h in st.session_state.history if h["prediction"] == "REAL")
#     fake_count = sum(1 for h in st.session_state.history if h["prediction"] == "FAKE")

#     labels = ["REAL", "FAKE"]
#     sizes = [real_count, fake_count]

#     fig, ax = plt.subplots()
#     ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
#     ax.axis("equal")

#     st.pyplot(fig)

# # ---------------- PDF Export ----------------
# def generate_pdf(history):
    
#     pdf.set_auto_page_break(True, 10)
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     pdf.cell(0, 10, "Fake News Detection Report", ln=True)
#     pdf.ln(5)

#     for i, h in enumerate(history, 1):
#         pdf.set_font("Arial", "B", 12)
#         pdf.cell(0, 10, f"Entry {i}", ln=True)
#         pdf.set_font("Arial", size=11)
#         pdf.multi_cell(0, 8, f"Prediction: {h['prediction']}")
#         pdf.multi_cell(0, 8, f"Confidence: {h['confidence']:.2f}%")
#         pdf.multi_cell(0, 8, f"Keywords: {', '.join(h['keywords'])}")
#         pdf.multi_cell(0, 8, f"Text: {h['text']}")
#         pdf.ln(4)

#     return pdf

# if st.button("📄 Export Report as PDF"):
#     if len(st.session_state.history) == 0:
#         st.warning("No history to export yet.")
#     else:
#         pdf = generate_pdf(st.session_state.history)
#         pdf.output("FakeNews_Report.pdf")
#         st.success("PDF exported successfully! Saved as FakeNews_Report.pdf")
