
import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Title and instructions
st.set_page_config(page_title="Review Intelligence", layout="wide")
st.title("🧠 Review Intelligence Dashboard")
st.markdown("Upload file CSV yang berisi kolom: `produk`, `review`, `tanggal`, dan `rating`.")

# File uploader
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Ensure necessary columns exist
    expected_cols = {'produk', 'review', 'tanggal', 'rating'}
    if not expected_cols.issubset(df.columns):
        st.error(f"File harus memiliki kolom: {expected_cols}")
    else:
        # Convert tanggal ke datetime
        df['tanggal'] = pd.to_datetime(df['tanggal'])

        # Sentiment analysis using HuggingFace model
        with st.spinner("Analisis sentimen..."):
            classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            df['sentiment'] = df['review'].astype(str).apply(lambda x: classifier(x[:512])[0]['label'])

        # Layout dashboard
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Ringkasan Sentimen")
            st.write(df['sentiment'].value_counts())
            st.bar_chart(df['sentiment'].value_counts())

        with col2:
            st.subheader("🔎 Keyword Populer (Sederhana)")
            vectorizer = CountVectorizer(stop_words='english', max_features=10)
            X = vectorizer.fit_transform(df['review'].astype(str))
            keywords = vectorizer.get_feature_names_out()
            counts = X.toarray().sum(axis=0)
            keyword_df = pd.DataFrame({'keyword': keywords, 'jumlah': counts}).sort_values(by='jumlah', ascending=False)
            st.dataframe(keyword_df)

        st.subheader("📅 Review dan Sentimen")
        st.dataframe(df[['produk', 'tanggal', 'review', 'sentiment']])

        # 📈 Grafik Tren Sentimen Mingguan
        st.subheader("📈 Tren Sentimen Mingguan")
        df['minggu'] = df['tanggal'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_sentiment = df.groupby(['minggu', 'sentiment']).size().unstack(fill_value=0)
        st.line_chart(weekly_sentiment)
