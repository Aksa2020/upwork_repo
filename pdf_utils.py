import os
from fpdf import FPDF
from groq_utils import get_project_plan, get_cover_letter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import textwrap
import re

def create_tools_flow_diagram(steps, diagram_path):
    """Create vertical flow diagram optimized for PDF"""
    try:
        if not steps:
            steps = ["No steps available"]
        
        # Process and limit steps
        processed_steps = []
        for i, step in enumerate(steps[:15], 1):  # Max 15 steps
            clean_step = step.strip()
            if not clean_step.split('.')[0].isdigit():
                clean_step = f"{i}. {clean_step}"
            
            if len(clean_step) > 50:
                if ':' in clean_step:
                    title, tools = clean_step.split(':', 1)
                    clean_step = f"{title}:\n{tools.strip()}"
                else:
                    clean_step = textwrap.fill(clean_step, width=45)
            processed_steps.append(clean_step)
        
        # Figure dimensions
        num_steps = len(processed_steps)
        fig_width = 10
        step_height = 0.8
        fig_height = max(6, num_steps * step_height + 2)
        
        if fig_height > fig_width * 1.5:
            fig_height = fig_width * 1.4
            step_height = (fig_height - 2) / num_steps
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, num_steps + 1)
        ax.axis('off')
        
        colors = ['#E8F4FD', '#FFF2CC', '#E1F5FE', '#F3E5F5', '#E8F5E8', '#FFE0B2']
        
        for i, step in enumerate(processed_steps):
            y_pos = num_steps - i
            color = colors[i % len(colors)]
            
            box_height = min(0.6, step_height * 0.8)
            box = FancyBboxPatch((1.5, y_pos - box_height/2), 7, box_height,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor='#2196F3', linewidth=1.5)
            ax.add_patch(box)
            
            font_size = max(8, min(10, 120 / len(step)))
            ax.text(5, y_pos, step, ha='center', va='center',
                   fontsize=font_size, fontweight='bold')
            
            if i < len(processed_steps) - 1:
                arrow = patches.FancyArrowPatch((5, y_pos - box_height/2 - 0.1), 
                                              (5, y_pos - step_height + box_height/2 + 0.1),
                                              arrowstyle='->', mutation_scale=15,
                                              color='#2196F3', linewidth=2)
                ax.add_patch(arrow)
        
        ax.text(5, num_steps + 0.3, 'Project Implementation Flow', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='#1976D2', alpha=0.8),
               color='white')
        
        plt.tight_layout(pad=0.5)
        plt.savefig(diagram_path, bbox_inches='tight', dpi=150, format='png', 
                   facecolor='white', pad_inches=0.1)
        plt.close()
        
    except Exception as e:
        print(f"Diagram error: {e}")
        create_fallback_diagram(steps, diagram_path)

def create_fallback_diagram(steps, diagram_path):
    """Simple fallback diagram"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, "Project Flow Diagram", ha='center', va='center', 
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"Total Steps: {len(steps)}", ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        
        step_text = ""
        for i, step in enumerate(steps[:5]):
            step_text += f"{i+1}. {step[:50]}...\n"
        
        ax.text(0.5, 0.3, step_text, ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        
        if len(steps) > 5:
            ax.text(0.5, 0.1, f"... and {len(steps)-5} more steps", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        ax.axis('off')
        plt.savefig(diagram_path, bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Fallback diagram failed: {e}")

def extract_steps_from_project_plan(project_plan):
    """Extract steps from project plan using multiple methods"""
    if not project_plan or "API failure" in project_plan:
        return ["Project plan not available"]
    
    lines = [line.strip() for line in project_plan.strip().split('\n') if line.strip()]
    
    # Method 1: Numbered steps
    numbered_steps = []
    patterns = [r'^\d+\.', r'^\d+\)', r'^Step\s+\d+:', r'^\*\*\d+\.', r'^\d+\s*[-–—]']
    for line in lines:
        if any(re.match(p, line, re.IGNORECASE) for p in patterns):
            numbered_steps.append(line)
    
    if len(numbered_steps) > 2:
        return numbered_steps
    
    # Method 2: Bullet points
    bullet_steps = []
    bullet_patterns = [r'^[-*•]\s*', r'^\*\*[-*•]\s*', r'^→\s*', r'^▪\s*']
    for line in lines:
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                clean_step = re.sub(pattern, '', line).strip()
                if len(clean_step) > 10:
                    bullet_steps.append(clean_step)
                break
    
    if len(bullet_steps) > 2:
        return bullet_steps
    
    # Method 3: Colon-separated
    colon_steps = []
    for line in lines:
        if ':' in line and len(line) > 15:
            if not any(h in line.lower() for h in ['project', 'overview', 'summary', 'title']):
                colon_steps.append(line)
    
    if len(colon_steps) > 2:
        return colon_steps
    
    # Method 4: Paragraphs
    paragraphs = [p.strip() for p in project_plan.split('\n\n') if p.strip()]
    substantial = []
    for para in paragraphs:
        clean_para = ' '.join(para.split('\n')).strip()
        if (len(clean_para) > 20 and 
            not clean_para.lower().startswith(('project', 'overview', 'introduction', 'summary'))):
            substantial.append(clean_para)
    
    return substantial[:15] if substantial else [f"Raw content: {project_plan[:500]}..."]

def save_solution_pdf(job_id, title, description, project_plan, diagram_path, pdf_path):
    """Save project solution with diagram to PDF"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, f"Job Title: {title}", ln=True, align='C')
    pdf.ln(10)
    
    # Description
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Client Request:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, description)
    pdf.ln(5)
    
    # Project flow
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Complete Project Flow:", ln=True)
    pdf.set_font("Arial", size=9)
    
    for line in project_plan.split("\n"):
        clean_line = line.strip().lstrip("-*").strip()
        if clean_line:
            if len(clean_line) > 90:
                pdf.multi_cell(0, 5, clean_line)
            else:
                pdf.cell(0, 5, clean_line, ln=True)
    
    # Add diagram
    if os.path.exists(diagram_path):
        try:
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, "Project Flow Diagram:", ln=True)
            pdf.ln(5)
            
            # Fit diagram to page
            page_width, margin = 210, 10
            available_width = page_width - (2 * margin)
            pdf.image(diagram_path, x=margin, y=25, w=available_width)
            
        except Exception as e:
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Diagram error: {str(e)}", ln=True)
    
    pdf.output(pdf_path)

def save_cover_letter_pdf(job_id, title, cover_letter, pdf_path):
    """Save cover letter to PDF"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, f"Job Title: {title}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Bid Cover Letter:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, cover_letter.strip())
    
    pdf.output(pdf_path)

def generate_all_pdfs_for_job(job_id, title, description, skills):
    """Generate all PDFs for a job"""
    try:
        # Get project plan and cover letter
        try:
            project_plan, _ = get_project_plan(title, description, skills)
        except:
            project_plan = "No project plan was generated due to API failure."
        
        try:
            cover_letter = get_cover_letter(title, description, skills)
        except:
            cover_letter = "Cover letter could not be generated due to API failure."
        
        # Create output folder
        folder = f'outputs/{job_id}'
        os.makedirs(folder, exist_ok=True)
        
        # Extract steps and create diagram
        steps = extract_steps_from_project_plan(project_plan)
        diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
        create_tools_flow_diagram(steps, diagram_path)
        
        # Generate PDFs
        solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
        save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)
        
        cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
        save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
        
        return {
            'solution_pdf': solution_pdf_path,
            'cover_letter_pdf': cover_letter_pdf_path,
            'diagram_path': diagram_path,
            'steps_count': len(steps)
        }
        
    except Exception as e:
        print(f"Error generating PDFs: {e}")
        return None
