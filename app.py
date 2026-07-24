import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from detoxify import Detoxify

st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return Detoxify("original")

model = load_model()

st.sidebar.title("About")

st.sidebar.info("""
Comment Toxicity Detector

Technology Used

• Python
• Streamlit
• Detoxify
• Deep Learning
""")

st.title("🛡️ Comment Toxicity Detector")

st.write("Analyze online comments and detect toxic language in real time.")

comment = st.text_area(
    "Enter Your Comment",
    placeholder="Type your comment here...",
    height=180
)

if comment.strip():

    with st.spinner("Analyzing..."):
        results = model.predict(comment)

    highest = max(results.values())

    st.subheader("Prediction")

    if highest > 0.5:
        st.error(f"🚨 Toxic Comment\n\nConfidence : {highest*100:.2f}%")
    else:
        st.success(f"✅ Not Toxic Comment\n\nConfidence : {(1-highest)*100:.2f}%")

    st.subheader("Toxicity Breakdown")

    df = pd.DataFrame({
        "Category": list(results.keys()),
        "Confidence (%)": [round(x * 100, 2) for x in results.values()]
    })

    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Confidence by Category")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["Category"], df["Confidence (%)"])
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Category")
    plt.xticks(rotation=45)

    st.pyplot(fig)

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    csv = pd.read_csv(uploaded_file).head(20)

    st.subheader("Uploaded CSV")
    st.dataframe(csv, use_container_width=True)

    if "comment_text" in csv.columns:
        text_column = "comment_text"
    elif "comment" in csv.columns:
        text_column = "comment"
    else:
        st.error("CSV must contain 'comment_text' or 'comment' column.")
        st.stop()

    predictions = []
    confidence = []

    with st.spinner("Analyzing CSV..."):

        for text in csv[text_column]:

            result = model.predict(str(text))

            highest = max(result.values())

            if highest > 0.5:
                predictions.append("Toxic")
            else:
                predictions.append("Not Toxic")

            confidence.append(round(highest * 100, 2))

    csv["Prediction"] = predictions
    csv["Confidence (%)"] = confidence

    st.subheader("Prediction Results")

    st.dataframe(csv, use_container_width=True)

    st.subheader("📊 Toxic vs Not Toxic")

    counts = csv["Prediction"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax2.axis("equal")

    st.pyplot(fig2)

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Developed using Streamlit • Detoxify • Deep Learning")