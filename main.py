import os
import streamlit as st
import pandas as pd
from pdf_utils import generate_all_pdfs_for_job
from utils import load_processed_jobs, save_processed_jobs, filter_jobs


st.set_page_config(page_title="Upwork Proposal Generator", layout="wide")
st.title("Upwork Project PDF Generator with LLM + Memory")

memory_file = "processed_jobs.json"
processed_jobs = load_processed_jobs(memory_file)


def download_pdf(file_path, label):
    with open(file_path, "rb") as file:
        st.download_button(
            label=f"📥 Download {label}",
            data=file,
            file_name=os.path.basename(file_path),
            mime='application/pdf'
        )


def open_pdf_locally(file_path, label):
    absolute_path = os.path.abspath(file_path)
    st.markdown(f"[➡️ Open {label} (Local Only)]({absolute_path})", unsafe_allow_html=True)


uploaded_csv = st.file_uploader("Upload your Jobs CSV", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df_filtered = filter_jobs(df)

    st.write("### Filtered Jobs (Clean & Ready for Action):")
    st.dataframe(df_filtered)

    for _, row in df_filtered.iterrows():
        job_id = row["Job ID"]

        if job_id in processed_jobs:
            st.info(f"✅ Already Done: {job_id}")
            continue

        title = row["Title"]
        description = row["Description"]
        skills = row["Skills"]

        with st.expander(f"{job_id} | {title}"):
            st.markdown(f"**Description:** {description}")
            st.markdown(f"**Skills:** {skills}")
            if st.button(f"Generate PDFs for {job_id}"):
                generate_all_pdfs_for_job(job_id, title, description, skills)
                processed_jobs.append(job_id)
                save_processed_jobs(processed_jobs, memory_file)
                st.success(f"📄 PDFs generated for {job_id} successfully ✅")

        # PDF Paths
        solution_pdf = f"outputs/{job_id}/{job_id}_solution_flow.pdf"
        cover_letter_pdf = f"outputs/{job_id}/{job_id}_cover_letter.pdf"

        if os.path.exists(solution_pdf):
            st.subheader(f"{job_id} Solution Flow PDF")
            download_pdf(solution_pdf, "Solution Flow PDF")
            open_pdf_locally(solution_pdf, "Solution Flow PDF")

        if os.path.exists(cover_letter_pdf):
            st.subheader(f"{job_id} Cover Letter PDF")
            download_pdf(cover_letter_pdf, "Cover Letter PDF")
            open_pdf_locally(cover_letter_pdf, "Cover Letter PDF")


    st.write("### Memory of Completed Jobs:")
    st.write(processed_jobs)
