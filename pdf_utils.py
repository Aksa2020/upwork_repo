import os
from fpdf import FPDF
from groq_utils import get_project_plan, get_cover_letter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import textwrap

def create_tools_flow_diagram(steps, diagram_path):
    """Create a comprehensive vertical flow diagram of ALL project steps - optimized for PDF"""
    try:
        # Ensure we have valid steps
        if not steps or len(steps) == 0:
            steps = ["No steps available"]
        
        print(f"Creating diagram with {len(steps)} steps")  # Debug print
        
        # Process steps - keep them concise but readable
        processed_steps = []
        for i, step in enumerate(steps, 1):
            # Clean up the step text
            clean_step = step.strip()
            
            # If step doesn't start with number, add it
            if not clean_step.split('.')[0].isdigit():
                clean_step = f"{i}. {clean_step}"
            
            # Wrap long text for better display - more aggressive wrapping for PDF
            if len(clean_step) > 50:  # Reduced from 60
                if ':' in clean_step:
                    title, tools = clean_step.split(':', 1)
                    clean_step = f"{title}:\n{tools.strip()}"
                else:
                    wrapped = textwrap.fill(clean_step, width=45)  # Reduced from 50
                    clean_step = wrapped
                    
            processed_steps.append(clean_step)
        
        # Calculate figure dimensions - optimize for PDF (A4 aspect ratio)
        num_steps = len(processed_steps)
        
        # Limit maximum height to prevent PDF overflow
        max_steps_per_diagram = 15
        if num_steps > max_steps_per_diagram:
            print(f"Warning: Too many steps ({num_steps}). Showing first {max_steps_per_diagram}")
            processed_steps = processed_steps[:max_steps_per_diagram]
            processed_steps.append(f"... and {num_steps - max_steps_per_diagram} more steps")
            num_steps = len(processed_steps)
        
        # Optimize dimensions for A4 PDF (portrait)
        # A4 is 210mm x 297mm, aspect ratio ~0.707
        fig_width = 10  # Reduced from 14
        step_height = 0.8  # Reduced from 1.2
        fig_height = max(6, num_steps * step_height + 2)  # More compact
        
        # Ensure reasonable aspect ratio
        if fig_height > fig_width * 1.5:  # If too tall
            fig_height = fig_width * 1.4
            step_height = (fig_height - 2) / num_steps
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, num_steps + 1)
        ax.axis('off')
        
        # Colors for different types of steps
        colors = ['#E8F4FD', '#FFF2CC', '#E1F5FE', '#F3E5F5', '#E8F5E8', '#FFE0B2']
        
        for i, step in enumerate(processed_steps):
            y_pos = num_steps - i
            color = colors[i % len(colors)]
            
            # Create more compact boxes
            box_height = min(0.6, step_height * 0.8)  # Adaptive box height
            box = FancyBboxPatch((1.5, y_pos - box_height/2), 7, box_height,
                               boxstyle="round,pad=0.08",  # Reduced padding
                               facecolor=color,
                               edgecolor='#2196F3',
                               linewidth=1.5)  # Thinner lines
            ax.add_patch(box)
            
            # Add step text with smaller font for compactness
            font_size = max(8, min(10, 120 / len(step)))  # Adaptive font size
            ax.text(5, y_pos, step, ha='center', va='center',
                   fontsize=font_size, fontweight='bold', wrap=False)
            
            # Add smaller arrows to next step
            if i < len(processed_steps) - 1:
                arrow_start_y = y_pos - box_height/2 - 0.1
                arrow_end_y = y_pos - step_height + box_height/2 + 0.1
                arrow = patches.FancyArrowPatch((5, arrow_start_y), (5, arrow_end_y),
                                              arrowstyle='->', 
                                              mutation_scale=15,  # Smaller arrows
                                              color='#2196F3',
                                              linewidth=2)
                ax.add_patch(arrow)
        
        # Add compact title
        ax.text(5, num_steps + 0.3, 'Project Implementation Flow', 
               ha='center', va='center', fontsize=12, fontweight='bold',  # Smaller title
               bbox=dict(boxstyle="round,pad=0.2", facecolor='#1976D2', edgecolor='black', alpha=0.8),
               color='white')
        
        plt.tight_layout(pad=0.5)  # Tighter layout
        
        # Save with higher DPI but optimized for PDF
        plt.savefig(diagram_path, bbox_inches='tight', dpi=150, format='png', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        print(f"PDF-optimized diagram saved successfully to {diagram_path}")
        print(f"Final dimensions: {fig_width}x{fig_height} inches")
        
    except Exception as e:
        print(f"Error creating flow diagram: {e}")
        # Create a simple fallback diagram
        create_fallback_diagram(steps, diagram_path)

def create_fallback_diagram(steps, diagram_path):
    """Create a simple fallback diagram if main creation fails"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, "Project Flow Diagram", ha='center', va='center', 
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.5, 0.5, f"Total Steps: {len(steps)}", ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        
        # List first few steps
        step_text = ""
        for i, step in enumerate(steps[:5]):
            step_text += f"{i+1}. {step[:50]}...\n"
        
        ax.text(0.5, 0.3, step_text, ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        
        if len(steps) > 5:
            ax.text(0.5, 0.1, f"... and {len(steps)-5} more steps", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(diagram_path, bbox_inches='tight', dpi=150)
        plt.close()
        print("Fallback diagram created")
    except Exception as e:
        print(f"Even fallback diagram failed: {e}")

def extract_steps_from_project_plan(project_plan):
    """Extract ALL steps from Groq model response - handles various formats"""
    steps = []
    
    try:
        if not project_plan or project_plan.strip() == "No project plan was generated due to API failure.":
            return ["Project plan not available"]
        
        print("=== DEBUGGING PROJECT PLAN EXTRACTION ===")
        print("Raw project plan from Groq:")
        print(repr(project_plan[:1000]))  # Show first 1000 chars with special characters
        print("=" * 50)
        
        # Clean the project plan text
        clean_plan = project_plan.strip()
        lines = clean_plan.split('\n')
        
        # Method 1: Look for numbered steps (1., 2., etc.)
        numbered_steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match patterns like "1.", "1)", "Step 1:", etc.
            import re
            number_patterns = [
                r'^\d+\.',  # 1., 2., 3.
                r'^\d+\)',  # 1), 2), 3)
                r'^Step\s+\d+:',  # Step 1:, Step 2:
                r'^\*\*\d+\.',  # **1., **2. (markdown bold)
                r'^\d+\s*[-–—]',  # 1 -, 2 -, etc.
            ]
            
            for pattern in number_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    numbered_steps.append(line)
                    break
        
        if numbered_steps and len(numbered_steps) > 2:  # At least 3 steps to be confident
            print(f"Method 1 SUCCESS: Found {len(numbered_steps)} numbered steps")
            for i, step in enumerate(numbered_steps):
                print(f"  Step {i+1}: {step[:100]}...")
            return numbered_steps
        
        # Method 2: Look for bullet points or dashes
        bullet_steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match patterns like "- Step", "* Step", "• Step", etc.
            bullet_patterns = [
                r'^[-*•]\s*',  # -, *, •
                r'^\*\*[-*•]\s*',  # **-, **, **•
                r'^→\s*',  # →
                r'^▪\s*',  # ▪
            ]
            
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    # Remove the bullet point marker
                    clean_step = re.sub(pattern, '', line).strip()
                    if len(clean_step) > 10:  # Must be substantial
                        bullet_steps.append(clean_step)
                    break
        
        if bullet_steps and len(bullet_steps) > 2:
            print(f"Method 2 SUCCESS: Found {len(bullet_steps)} bullet-point steps")
            for i, step in enumerate(bullet_steps):
                print(f"  Step {i+1}: {step[:100]}...")
            return bullet_steps
        
        # Method 3: Look for lines with colons (Phase: Description)
        colon_steps = []
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 15:
                # Skip common headers
                if not any(header in line.lower() for header in ['project', 'overview', 'summary', 'title', 'description']):
                    colon_steps.append(line)
        
        if colon_steps and len(colon_steps) > 2:
            print(f"Method 3 SUCCESS: Found {len(colon_steps)} colon-separated steps")
            for i, step in enumerate(colon_steps):
                print(f"  Step {i+1}: {step[:100]}...")
            return colon_steps
        
        # Method 4: Look for markdown headers (## Step, ### Phase, etc.)
        header_steps = []
        for line in lines:
            line = line.strip()
            if re.match(r'^#{1,4}\s+', line):  # # ## ### ####
                clean_step = re.sub(r'^#{1,4}\s+', '', line).strip()
                if len(clean_step) > 5:
                    header_steps.append(clean_step)
        
        if header_steps and len(header_steps) > 2:
            print(f"Method 4 SUCCESS: Found {len(header_steps)} markdown header steps")
            return header_steps
        
        # Method 5: Split by double newlines (paragraphs) and filter
        paragraphs = [p.strip() for p in clean_plan.split('\n\n') if p.strip()]
        substantial_paragraphs = []
        for para in paragraphs:
            # Remove single newlines within paragraph
            clean_para = ' '.join(para.split('\n')).strip()
            # Must be substantial and not just a header
            if (len(clean_para) > 20 and 
                not clean_para.lower().startswith(('project', 'overview', 'introduction', 'summary', 'conclusion'))):
                substantial_paragraphs.append(clean_para)
        
        if substantial_paragraphs:
            print(f"Method 5 SUCCESS: Found {len(substantial_paragraphs)} paragraph-based steps")
            return substantial_paragraphs[:15]  # Limit to avoid too many
        
        # Method 6: Last resort - split by sentences and take meaningful ones
        sentences = re.split(r'[.!?]+', clean_plan)
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and 
                any(keyword in sentence.lower() for keyword in 
                    ['implement', 'develop', 'create', 'build', 'design', 'deploy', 'configure', 'setup', 'install', 'train', 'test'])):
                meaningful_sentences.append(sentence)
        
        if meaningful_sentences:
            print(f"Method 6 SUCCESS: Found {len(meaningful_sentences)} sentence-based steps")
            return meaningful_sentences[:10]
        
        print("ALL METHODS FAILED - Returning raw project plan")
        return [f"Could not parse project plan. Raw content: {clean_plan[:500]}..."]
        
    except Exception as e:
        print(f"Error extracting steps: {e}")
        import traceback
        traceback.print_exc()
        return [f"Error processing project plan: {str(e)}"]

def get_image_dimensions_mm(image_path):
    """Get image dimensions in millimeters for PDF"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width_px, height_px = img.size
            dpi = img.info.get('dpi', (72, 72))[0]  # Default to 72 DPI if not specified
            width_mm = (width_px / dpi) * 25.4  # Convert inches to mm
            height_mm = (height_px / dpi) * 25.4
            return width_mm, height_mm
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return None, None

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
    pdf.cell(0, 10, "Complete Project Flow:", ln=True)
    pdf.set_font("Arial", size=9)
    
    # Process project plan lines
    for line in project_plan.split("\n"):
        clean_line = line.strip().lstrip("-").lstrip("*").strip()
        if clean_line:
            if len(clean_line) > 90:
                pdf.multi_cell(0, 5, clean_line)
            else:
                pdf.cell(0, 5, clean_line, ln=True)
    
    pdf.ln(5)
    
    # Add diagram if it exists - IMPROVED VERSION
    if os.path.exists(diagram_path):
        try:
            # Always add diagram on a new page
            pdf.add_page()
            
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, "Project Flow Diagram:", ln=True)
            pdf.ln(5)
            
            # PDF page dimensions (A4 in mm)
            page_width = 210
            page_height = 297
            margin = 10
            header_space = 25  # Space for "Project Flow Diagram:" header
            
            available_width = page_width - (2 * margin)  # 190mm
            available_height = page_height - header_space - margin  # ~262mm
            
            # Get actual image dimensions
            img_width_mm, img_height_mm = get_image_dimensions_mm(diagram_path)
            
            if img_width_mm and img_height_mm:
                print(f"Image dimensions: {img_width_mm:.1f}mm x {img_height_mm:.1f}mm")
                print(f"Available space: {available_width}mm x {available_height}mm")
                
                # Calculate scaling factor to fit both width and height
                width_scale = available_width / img_width_mm
                height_scale = available_height / img_height_mm
                scale_factor = min(width_scale, height_scale, 1.0)  # Don't scale up
                
                final_width = img_width_mm * scale_factor
                final_height = img_height_mm * scale_factor
                
                print(f"Scale factor: {scale_factor:.3f}")
                print(f"Final size: {final_width:.1f}mm x {final_height:.1f}mm")
                
                # Center the image horizontally
                x_offset = margin + (available_width - final_width) / 2
                
                # Add image with calculated dimensions
                pdf.image(diagram_path, x=x_offset, y=header_space, w=final_width, h=final_height)
                
                # If image is still too tall, add a note
                if final_height > available_height:
                    pdf.ln(10)
                    pdf.set_font("Arial", size=9, style="I")
                    pdf.cell(0, 5, "Note: Diagram may extend beyond page. Check the PNG file for complete view.", ln=True)
                    
            else:
                # Fallback: try to fit with estimated dimensions
                print("Could not determine image dimensions, using fallback method")
                # Try to fit the image by width first
                pdf.image(diagram_path, x=margin, y=header_space, w=available_width)
            
        except Exception as e:
            print(f"Error adding diagram to PDF: {e}")
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Diagram could not be loaded. Error: {str(e)}", ln=True)
            pdf.cell(0, 10, f"Please check the PNG file: {diagram_path}", ln=True)
    else:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, "Flow diagram was not generated.", ln=True)
    
    pdf.output(pdf_path)
    print(f"Solution PDF saved to {pdf_path}")

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
    """Generate all PDFs for a job including complete flow diagram"""
    try:
        print(f"Starting PDF generation for job: {job_id}")
        
        # Get project plan and cover letter
        try:
            project_plan, steps_dict = get_project_plan(title, description, skills)
            print(f"Got project plan, length: {len(project_plan) if project_plan else 0}")
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
        print(f"Created output folder: {folder}")
        
        # Extract ALL steps from project plan
        steps = extract_steps_from_project_plan(project_plan)
        print(f"Extracted {len(steps)} steps from project plan")
        
        # Create comprehensive flow diagram
        diagram_path = os.path.join(folder, f"{job_id}_flow_diagram.png")
        create_tools_flow_diagram(steps, diagram_path)
        
        # Generate solution PDF with diagram
        solution_pdf_path = os.path.join(folder, f"{job_id}_solution_flow.pdf")
        save_solution_pdf(job_id, title, description, project_plan, diagram_path, solution_pdf_path)
        
        # Generate cover letter PDF
        cover_letter_pdf_path = os.path.join(folder, f"{job_id}_cover_letter.pdf")
        save_cover_letter_pdf(job_id, title, cover_letter, cover_letter_pdf_path)
        
        print(f"Successfully generated all PDFs for job {job_id}")
        print(f"Solution PDF: {solution_pdf_path}")
        print(f"Cover Letter PDF: {cover_letter_pdf_path}")
        print(f"Flow Diagram: {diagram_path}")
        
        return {
            'solution_pdf': solution_pdf_path,
            'cover_letter_pdf': cover_letter_pdf_path,
            'diagram_path': diagram_path,
            'steps_count': len(steps)
        }
        
    except Exception as e:
        print(f"Error in generate_all_pdfs_for_job: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_groq_response(job_id, title, description, skills):
    """Debug function to see exactly what Groq returns"""
    try:
        print("=== DEBUGGING GROQ MODEL RESPONSE ===")
        print(f"Job ID: {job_id}")
        print(f"Title: {title}")
        print(f"Description: {description[:200]}...")
        print(f"Skills: {skills}")
        print("-" * 50)
        
        # Get the actual response from Groq
        project_plan, steps_dict = get_project_plan(title, description, skills)
        
        print("RAW PROJECT PLAN FROM GROQ:")
        print("=" * 60)
        print(repr(project_plan))  # This shows all special characters
        print("=" * 60)
        print("FORMATTED PROJECT PLAN:")
        print(project_plan)
        print("=" * 60)
        
        print(f"STEPS_DICT: {steps_dict}")
        print("-" * 50)
        
        # Test extraction
        extracted_steps = extract_steps_from_project_plan(project_plan)
        print(f"\nEXTRACTED STEPS ({len(extracted_steps)}):")
        for i, step in enumerate(extracted_steps, 1):
            print(f"{i}. {step}")
        
        return project_plan, extracted_steps
        
    except Exception as e:
        print(f"Error in debug function: {e}")
        import traceback
        traceback.print_exc()
        return None, []

# Test function to debug step extraction
def debug_step_extraction(sample_project_plan):
    """Debug function to test step extraction"""
    print("=== DEBUG: Testing Step Extraction ===")
    steps = extract_steps_from_project_plan(sample_project_plan)
    print(f"Extracted {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")
    return steps

# Example usage for testing
if __name__ == "__main__":
    sample_plan = """1. Data Collection: IP Cameras, Raspberry Pi-4 (4GB)
2. Data Annotation: LabelImg, CVAT
3. Data Preprocessing: OpenCV, Pandas
4. Model Training: YOLOv8, PyTorch, ResNet-50
5. Model Optimization: TensorRT, PyTorch
6. Model Deployment: FastAPI, Docker
7. Model Inference: FastAPI, Docker, Azure
8. Real-time Video Processing: OpenCV, FastAPI, Docker, Azure
9. Streamlit Dashboard: Streamlit, FastAPI
10. Cloud Infrastructure: Azure VMs and Instances (Tesla P100, V100, A100 GPUs with 32GB / 64GB RAM)
11. Edge Device Deployment: Raspberry Pi-4 (4GB), Intel Movidius NCS2
12. Model Updates and Maintenance: Azure, Docker, FastAPI"""
    
    debug_step_extraction(sample_plan)
