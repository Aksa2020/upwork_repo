import os
import shutil
import streamlit as st
import pandas as pd
from pdf_utils import generate_all_pdfs_for_job
from utils import load_processed_jobs, save_processed_jobs, filter_jobs


st.set_page_config(page_title="Upwork Proposal Generator", layout="wide")
st.title("Upwork Project PDF Generator with LLM + Memory")

memory_file = "processed_jobs.json"
processed_jobs = load_processed_jobs(memory_file)

if "generated_pdfs" not in st.session_state:
    st.session_state.generated_pdfs = {}

# 🚨 Clear All PDFs
if st.button("🗑️ Clear All Generated PDFs & Memory"):
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    if os.path.exists(memory_file):
        with open(memory_file, "w") as f:
            f.write("[]")

    st.session_state.generated_pdfs = {}
    processed_jobs.clear()
    st.success("✅ All generated PDFs and memory cleared.")

# 🔼 File Upload
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
        else:
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

                    solution_pdf = f"outputs/{job_id}/{job_id}_solution_flow.pdf"
                    cover_letter_pdf = f"outputs/{job_id}/{job_id}_cover_letter.pdf"
                    st.session_state.generated_pdfs[job_id] = {
                        "solution": solution_pdf,
                        "cover": cover_letter_pdf
                    }
                    st.success(f"📄 PDFs generated for {job_id} successfully ✅")

# 🔥 Show Links + Downloads
if st.session_state.generated_pdfs:
    st.header("📂 View / Download Your PDFs")
    for job_id, pdfs in st.session_state.generated_pdfs.items():
        st.subheader(f"{job_id} Documents:")

        solution_pdf = pdfs["solution"]
        cover_pdf = pdfs["cover"]

        st.markdown(f"[🔗 Open Solution Flow PDF in Browser]({solution_pdf})", unsafe_allow_html=True)
        with open(solution_pdf, "rb") as f:
            st.download_button(
                label="📥 Download Solution Flow PDF",
                data=f,
                file_name=os.path.basename(solution_pdf),
                mime="application/pdf"
            )

        st.markdown(f"[🔗 Open Cover Letter PDF in Browser]({cover_pdf})", unsafe_allow_html=True)
        with open(cover_pdf, "rb") as f:
            st.download_button(
                label="📥 Download Cover Letter PDF",
                data=f,
                file_name=os.path.basename(cover_pdf),
                mime="application/pdf"
            )

# Show processed job memory
st.write("### Memory of Completed Jobs:")
st.write(processed_jobs)
