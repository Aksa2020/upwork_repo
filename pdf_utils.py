import os
from fpdf import FPDF
import matplotlib.pyplot as plt


def create_tools_flow_diagram(steps_dict, output_path):
    fig, ax = plt.subplots(figsize=(6, len(steps_dict)))

    ax.axis('off')

    y_positions = list(range(len(steps_dict) * 2, 0, -2))
    for (step, tools), y in zip(steps_dict.items(), y_positions):
        ax.text(0.5, y, f"{step}\nTools: {tools}",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))

    for i in range(len(y_positions) - 1):
        ax.annotate('', xy=(0.5, y_positions[i+1] + 0.7), xytext=(0.5, y_positions[i] - 0.7),
                    arrowprops=dict(arrowstyle="->", lw=2))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_solution_pdf(job_id, title, description, project_plan, diagram_path, pdf_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Solution Flow Document", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Client Request:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, description.strip())
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Project Flow:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in project_plan.strip().split("\n"):
        if line.strip():
            pdf.multi_cell(0, 8, line.strip())
    pdf.add_page()

    pdf.image(diagram_path, x=20, y=40, w=170)
    pdf.output(pdf_path)


def save_cover_letter_pdf(job_id, cover_letter, pdf_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Cover Letter", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, cover_letter.strip())
    pdf.output(pdf_path)


def generate_all_pdfs_for_job(job_id, title, description, skills):
    from groq_utils import get_project_plan, get_cover_letter
    import os

    project_plan, steps_dict = get_project_plan(title, description, skills)
    cover_letter = get_cover_letter(title, description, skills)

    folder = os.path.join("outputs", job_id)
    os.makedirs(folder, exist_ok=True)

    diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
    create_tools_flow_diagram(steps_dict, diagram_path)

    solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
    save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)

    cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
    save_cover_letter_pdf(job_id, cover_letter, cover_letter_pdf_path)
