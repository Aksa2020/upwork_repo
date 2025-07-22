import os
from fpdf import FPDF
from groq_utils import get_project_plan, get_cover_letter
import matplotlib.pyplot as plt
from graphviz import Digraph

def create_tools_flow_diagram(steps, diagram_path):
    """Create a vertical flow diagram of project steps"""
    try:
        # Ensure we have valid steps
        if not steps or len(steps) == 0:
            steps = ["No steps available"]
        
        # Limit step text length to prevent overflow
        processed_steps = []
        for step in steps:
            if len(step) > 100:
                # Split long text into multiple lines
                step = step[:100] + "..."
            processed_steps.append(step)
        
        # Calculate figure size based on number of steps
        fig_height = max(6, len(processed_steps) * 1.5)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(processed_steps) - 0.5)
        
        for i, step in enumerate(processed_steps):
            y = len(processed_steps) - i - 1
            
            # Create text box for each step
            ax.text(0.5, y, step, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.4", fc="lightblue", ec="black"),
                    fontsize=9, fontweight='normal',
                    wrap=False, multialignment='center')
            
            # Add arrow to next step (except for last step)
            if i < len(processed_steps) - 1:
                ax.annotate('', xy=(0.5, y - 0.4), xytext=(0.5, y - 0.1),
                            arrowprops=dict(arrowstyle="->", lw=2, color='black'))
        
        plt.tight_layout()
        plt.savefig(diagram_path, bbox_inches='tight', dpi=150, format='png')
        plt.close()
        
    except Exception as e:
        print(f"Error creating flow diagram: {e}")
        # Create a simple fallback diagram
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Flow Diagram\nGeneration Error", 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.4", fc="lightcoral", ec="black"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(diagram_path, bbox_inches='tight', dpi=150)
            plt.close()
        except:
            pass  # If even fallback fails, continue without diagram

def save_solution_pdf(job_id, title, description, project_plan, diagram_path, pdf_path):
    """Save project solution with diagram to PDF"""
    if not project_plan:
        project_plan = "No project plan was generated due to API failure."
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, f"Job Title: {title}", ln=True, align='C')
    pdf.ln(10)
    
    # Client Request Section
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Client Request:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, description)
    pdf.ln(5)
    
    # Project Flow Section
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Proposed Project Flow:", ln=True)
    pdf.set_font("Arial", size=10)
    
    # Process project plan lines
    for line in project_plan.split("\n"):
        clean_line = line.strip().lstrip("-").lstrip("*").strip()
        if clean_line:
            # Handle long lines by splitting them
            if len(clean_line) > 80:
                pdf.multi_cell(0, 6, clean_line)
            else:
                pdf.cell(0, 6, clean_line, ln=True)
    
    pdf.ln(10)
    
    # Add diagram if it exists
    if os.path.exists(diagram_path):
        try:
            # Check current position and add new page if needed
            current_y = pdf.get_y()
            if current_y > 200:  # If near bottom of page
                pdf.add_page()
                current_y = pdf.get_y()
            
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, "Project Flow Diagram:", ln=True)
            pdf.ln(5)
            
            # Add image with proper sizing
            pdf.image(diagram_path, x=10, y=pdf.get_y(), w=190)
        except Exception as e:
            print(f"Error adding diagram to PDF: {e}")
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, "Diagram could not be loaded.", ln=True)
    
    pdf.output(pdf_path)

def save_cover_letter_pdf(job_id, title, cover_letter, pdf_path):
    """Save cover letter to PDF"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, f"Job Title: {title}", ln=True, align='C')
    pdf.ln(10)
    
    # Cover Letter Section
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Bid Cover Letter:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, cover_letter.strip())
    
    pdf.output(pdf_path)

def extract_steps_from_project_plan(project_plan):
    """Extract and format steps from project plan"""
    steps = []
    
    try:
        if not project_plan or project_plan.strip() == "No project plan was generated due to API failure.":
            return ["Project plan not available"]
        
        lines = project_plan.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove common prefixes
            line = line.lstrip('-*•').strip()
            
            # Skip very short lines (likely not actual steps)
            if len(line) < 5:
                continue
                
            # Handle lines with colons (step: description format)
            if ':' in line:
                step_title, description = line.split(':', 1)
                # Limit length to prevent display issues
                description = description.strip()
                if len(description) > 50:
                    description = description[:50] + "..."
                formatted_step = f"{step_title.strip()}: {description}"
                steps.append(formatted_step)
            else:
                # Add line as is if it seems like a step
                if any(keyword in line.lower() for keyword in ['step', 'phase', 'stage', 'task', 'implement', 'develop', 'create', 'design']):
                    if len(line) > 80:
                        line = line[:80] + "..."
                    steps.append(line)
        
        # If no steps found, try a different approach
        if not steps:
            for line in lines:
                line = line.strip().lstrip('-*•').strip()
                if line and len(line) > 10:  # Any substantial line
                    if len(line) > 80:
                        line = line[:80] + "..."
                    steps.append(line)
        
        # Limit to reasonable number of steps for diagram readability
        if len(steps) > 6:
            steps = steps[:6] + ["Additional steps in full plan..."]
        
        return steps if steps else ["Project plan structure not recognized"]
        
    except Exception as e:
        print(f"Error extracting steps: {e}")
        return ["Error processing project plan"]

def generate_all_pdfs_for_job(job_id, title, description, skills):
    """Generate all PDFs for a job including flow diagram"""
    try:
        # Validate inputs
        if not job_id or not title:
            raise ValueError("job_id and title are required")
            
        # Get project plan and cover letter
        try:
            project_plan, steps_dict = get_project_plan(title, description, skills)
        except Exception as e:
            print(f"Error getting project plan: {e}")
            project_plan = "No project plan was generated due to API failure."
            steps_dict = {}
            
        if not project_plan:
            project_plan = "No project plan was generated due to API failure."
        
        try:
            cover_letter = get_cover_letter(title, description, skills)
        except Exception as e:
            print(f"Error getting cover letter: {e}")
            cover_letter = "Cover letter could not be generated due to API failure."
        
        # Create output folder
        folder = f'outputs/{job_id}'
        os.makedirs(folder, exist_ok=True)
        
        # Extract steps from project plan
        steps = extract_steps_from_project_plan(project_plan)
        
        # Create flow diagram
        diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
        create_tools_flow_diagram(steps, diagram_path)
        
        # Generate solution PDF with diagram
        solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
        save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)
        
        # Generate cover letter PDF
        cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
        save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
        
        print(f"Successfully generated PDFs for job {job_id}")
        print(f"Solution PDF: {solution_pdf_path}")
        print(f"Cover Letter PDF: {cover_letter_pdf_path}")
        print(f"Flow Diagram: {diagram_path}")
        
        return {
            'solution_pdf': solution_pdf_path,
            'cover_letter_pdf': cover_letter_pdf_path,
            'diagram_path': diagram_path
        }
        
    except Exception as e:
        print(f"Error generating PDFs for job {job_id}: {e}")
        # Still try to create basic PDFs even if flow diagram fails
        try:
            folder = f'outputs/{job_id}'
            os.makedirs(folder, exist_ok=True)
            
            # Create basic PDFs without diagram
            solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
            cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
            
            # Use fallback values if API calls failed
            if 'project_plan' not in locals():
                project_plan = "Project plan generation failed."
            if 'cover_letter' not in locals():
                cover_letter = "Cover letter generation failed."
                
            save_solution_pdf(job_id, title, description, project_plan, "", solution_pdf_path)
            save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
            
            return {
                'solution_pdf': solution_pdf_path,
                'cover_letter_pdf': cover_letter_pdf_path,
                'diagram_path': None
            }
        except Exception as fallback_error:
            print(f"Fallback PDF generation also failed: {fallback_error}")
            return None

# Example usage (if running as main script)
if __name__ == "__main__":
    # Test with sample data
    test_job_id = "test_001"
    test_title = "Sample Project"
    test_description = "This is a test project description"
    test_skills = ["Python", "PDF Generation", "Data Visualization"]
    
    result = generate_all_pdfs_for_job(test_job_id, test_title, test_description, test_skills)
