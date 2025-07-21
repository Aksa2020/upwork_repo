import os
from fpdf import FPDF
from groq_utils import get_project_plan, get_cover_letter
import matplotlib.pyplot as plt


# Create clean vertical flow diagram using matplotlib
def create_tools_flow_diagram(steps, file_path):
    fig, ax = plt.subplots(figsize=(4, len(steps)))  # Dynamic height
    ax.axis('off')

    for i, step in enumerate(steps):
        ax.text(0.5, 1 - (i / len(steps)), step, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.4", fc="lightblue", ec="black"))

        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, 1 - (i / len(steps)) - 0.05),
                        xytext=(0.5, 1 - ((i + 1) / len(steps)) + 0.05),
                        arrowprops=dict(arrowstyle="->", lw=2))

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


# Save clean PDF for solution with diagram
def save_solution_pdf(job_id, title, description, project_plan, diagram_path, pdf_path):
    from fpdf import FPDF

    # Failsafe if project_plan is None
    if not project_plan:
        project_plan = "No project plan was generated due to an API error."

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, f"Job Title: {title}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Client Request:", ln=True)
    pdf.set_font("Arial", style="", size=12)
    pdf.multi_cell(0, 10, description)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Proposed Project Flow:", ln=True)
    pdf.set_font("Arial", style="", size=12)
    for line in project_plan.split("\n"):
        clean_line = line.strip().lstrip("-").lstrip("*").strip()
        if clean_line:
            pdf.cell(0, 10, clean_line, ln=True)

    pdf.ln(10)
    if os.path.exists(diagram_path):
        pdf.image(diagram_path, x=20, y=pdf.get_y(), w=170)

    pdf.output(pdf_path)


# Save clean PDF for cover letter
def save_cover_letter_pdf(job_id, title, cover_letter, pdf_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, f"Job Title: {title}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Bid Cover Letter:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, cover_letter.strip())

    pdf.output(pdf_path)


# Master function for a single job
def generate_all_pdfs_for_job(job_id, title, description, skills):
    project_plan = get_project_plan(title, description, skills)
    cover_letter = get_cover_letter(title, description, skills)

    folder = f'outputs/{job_id}'
    os.makedirs(folder, exist_ok=True)

    # Create clean flow diagram with tools/technologies
    steps = ["Python", "Pandas", "YOLO", "OpenCV", "Model Evaluation", "Deployment"]
    diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
    create_tools_flow_diagram(steps, diagram_path)

    # Solution PDF (Project Plan + Diagram)
    solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
    save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)

    # Cover Letter PDF
    cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
    save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
