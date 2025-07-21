import os
from fpdf import FPDF
from groq_utils import get_project_plan, get_cover_letter
import matplotlib.pyplot as plt
from graphviz import Digraph



def create_tools_flow_diagram(steps_dict, output_path):
    fig, ax = plt.subplots(figsize=(6, len(steps_dict)))  # Auto height

    ax.axis('off')

    y_positions = list(range(len(steps_dict)*2, 0, -2))
    for (step, tools), y in zip(steps_dict.items(), y_positions):
        ax.text(0.5, y, f"{step}\nTools: {tools}",
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))

    # Draw arrows downwards
    for i in range(len(y_positions) - 1):
        ax.annotate('', xy=(0.5, y_positions[i+1] + 0.7), xytext=(0.5, y_positions[i] - 0.7),
                    arrowprops=dict(arrowstyle="->", lw=2))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# Save clean PDF for solution with diagram
def save_solution_pdf(job_id, title, description, project_plan, diagram_path, pdf_path):
    if not project_plan:
        project_plan = "No project plan was generated due to API failure."

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




def generate_all_pdfs_for_job(job_id, title, description, skills):
    project_plan, steps_dict = get_project_plan(title, description, skills)
    cover_letter = get_cover_letter(title, description, skills)

    folder = f'outputs/{job_id}'
    os.makedirs(folder, exist_ok=True)

    diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
    create_tools_flow_diagram(steps_dict, diagram_path)

    solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
    save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)

    cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
    save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
