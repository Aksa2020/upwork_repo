import os
import shutil
import streamlit as st
import pandas as pd
from pdf_utils import generate_all_pdfs_for_job
from utils import load_processed_jobs, save_processed_jobs, filter_jobs

st.set_page_config(page_title="Upwork Proposal Generator", layout="wide")
st.title("Upwork Job Post")

# Memory and session state
memory_file = "processed_jobs.json"
processed_jobs = load_processed_jobs(memory_file)

if "generated_pdfs" not in st.session_state:
    st.session_state.generated_pdfs = {}

# Clear outputs
if st.button("🗑️ Clear All Displayed PDFs (Outputs Folder)"):
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    st.session_state.generated_pdfs = {}
    st.success("✅ All generated PDFs cleared from outputs folder.")

# CSV upload and processing
uploaded_csv = st.file_uploader("Upload your Jobs CSV", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df_filtered = filter_jobs(df)
    st.write("### Filtered Jobs:")
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
                    
                    st.session_state.generated_pdfs[job_id] = {
                        "solution": f"outputs/{job_id}/{job_id}_solution_flow.pdf",
                        "cover": f"outputs/{job_id}/{job_id}_cover_letter.pdf"
                    }
                    st.success(f"📄 PDFs generated for {job_id} ✅")

# Show current session PDFs
if st.session_state.generated_pdfs:
    st.header("📂 View / Download Your PDFs")
    
    for job_id, pdfs in st.session_state.generated_pdfs.items():
        st.subheader(f"{job_id} Documents:")
        
        # Solution PDF
        with open(pdfs["solution"], "rb") as f:
            st.download_button(
                label="📥 Download Solution Flow PDF",
                data=f,
                file_name=os.path.basename(pdfs["solution"]),
                mime="application/pdf"
            )
        
        # Cover Letter PDF
        with open(pdfs["cover"], "rb") as f:
            st.download_button(
                label="📥 Download Cover Letter PDF",
                data=f,
                file_name=os.path.basename(pdfs["cover"]),
                mime="application/pdf"
            )

# Show processed jobs list
st.write("### Processed Jobs:")
st.write(processed_jobs)
