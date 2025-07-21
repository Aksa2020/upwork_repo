from core.groq_utils import get_project_plan, get_cover_letter
from fpdf import FPDF
from PIL import Image
import os
from graphviz import Digraph


def create_tools_flow_diagram(steps_dict, output_path):
    dot = Digraph(comment='Project Flow Diagram')
    dot.attr(rankdir='TB', size='8,5')

    previous = None
    for step, tool in steps_dict.items():
        label = f"{step}\n({tool})"
        dot.node(step, label)

        if previous:
            dot.edge(previous, step)
        previous = step

    dot.render(output_path, format='png', cleanup=True)


def save_pdf(filename, title, description, project_plan, steps_dict, diagram_path):
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
    pdf.add_page()

    image = Image.open(diagram_path + '.png')
    width, height = image.size
    width_mm = 180
    height_mm = width_mm * height / width
    pdf.image(diagram_path + '.png', x=15, y=40, w=width_mm, h=height_mm)
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

    project_plan, steps_dict = get_project_plan(title, description, skills)
    diagram_path = os.path.join(folder, f"{job_id}_flow_diagram")
    create_tools_flow_diagram(steps_dict, diagram_path)

    solution_pdf = os.path.join(folder, f"{job_id}_solution_flow.pdf")
    save_pdf(solution_pdf, title, description, project_plan, steps_dict, diagram_path)

    cover_letter = get_cover_letter(title, description, skills)
    save_cover_letter_pdf(job_id, cover_letter, folder)
