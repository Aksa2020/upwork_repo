# app.py
import os
import streamlit as st
import pandas as pd
import json
from core.pdf_utils import generate_all_pdfs_for_job
from core.utils import load_processed_jobs, save_processed_jobs, filter_jobs


st.set_page_config(page_title="Upwork AI Proposal Generator", layout="wide")

# Load Memory
memory_file = "processed_jobs.json"
processed_jobs = load_processed_jobs(memory_file)

# Upload CSV
st.title("Upwork Project PDF Generator with LLM + Memory")
uploaded_csv = st.file_uploader("1.csv", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df_filtered = filter_jobs(df)

    st.write("Filtered Jobs (Excludes South Asia, Bad Ratings, Unverified Payments):")
    st.dataframe(df_filtered)

    for _, row in df_filtered.iterrows():
        job_id = row["Job ID"]
        if job_id in processed_jobs:
            st.info(f"✅ Already Done: {job_id}")
            continue

        title = row["Title"]
        description = row["Description"]
        skills = row["Skills"]

        st.markdown(f"### {title} (Job ID: {job_id})")
        if st.button(f"Generate PDFs for {job_id}"):
            generate_all_pdfs_for_job(job_id, title, description, skills)
            processed_jobs.append(job_id)
            save_processed_jobs(processed_jobs, memory_file)
            st.success(f"PDFs generated for {job_id} ✅")

    st.write("Done Jobs (Memory):", processed_jobs)
