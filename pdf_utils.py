from groq_utils import get_project_plan, get_cover_letter
from fpdf import FPDF
import os


def save_pdf(filename, title, description, project_plan):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, f"Job Title: {title}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Client Request:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, description)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Proposed Project Flow (with Technologies):", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, project_plan)

    pdf.output(filename)


def save_cover_letter_pdf(job_id, cover_letter, folder):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Bid Cover Letter", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, cover_letter)
    pdf.output(os.path.join(folder, f"{job_id}_cover_letter.pdf"))


def generate_all_pdfs_for_job(job_id, title, description, skills):
    folder = f"outputs/{job_id}"
    os.makedirs(folder, exist_ok=True)

    # Get LLM-generated plan and extract steps for visual display only (not in PDF diagram)
    project_plan, steps_dict = get_project_plan(title, description, skills)

    # Save PDF with details
    solution_pdf = os.path.join(folder, f"{job_id}_solution_flow.pdf")
    save_pdf(solution_pdf, title, description, project_plan)

    # Save PDF with cover letter
    cover_letter = get_cover_letter(title, description, skills)
    save_cover_letter_pdf(job_id, cover_letter, folder)

    # Return steps_dict so Streamlit can visualize with graphviz_chart (no PNG needed)
    return project_plan, steps_dict
