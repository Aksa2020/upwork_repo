

import streamlit as st
from groq import Groq
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqProjectPlanner:
    """
    Professional AI project planning system using Groq's LLaMA models.
    
    Generates technical project flows and cover letters optimized for
    machine learning and AI development projects.
    """
    
    def __init__(self):
        """Initialize the Groq client with API credentials."""
        try:
            self.client = Groq(api_key=st.secrets["groq"]["api_key"])
            self.model_name = "llama3-70b-8192"
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate_technical_project_flow(
        self, 
        title: str, 
        description: str, 
        skills: str
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate a comprehensive technical project flow with specific tools and technologies.
        
        Args:
            title (str): Project title
            description (str): Detailed project description
            skills (str): Required technical skills
            
        Returns:
            Tuple[str, Dict[str, str]]: Complete project flow and parsed steps dictionary
        """
        
        technical_stack_context = """
        TECHNICAL EXPERTISE & INFRASTRUCTURE:
        
        Core ML/AI Frameworks:
        - PyTorch 2.x, TensorFlow 2.x, Keras, Scikit-learn
        - Hugging Face Transformers, LangChain, LlamaIndex
        - OpenAI GPT-4/GPT-4-Turbo, Anthropic Claude, Cohere
        
        Computer Vision & Deep Learning:
        - YOLOv8/YOLOv11/YOLOv12, Detectron2, MMDetection
        - OpenCV, Pillow, ImageIO, Albumentations
        - CNN architectures: ResNet, EfficientNet, Vision Transformers
        - Object Detection: Faster R-CNN, SSD, RetinaNet, DETR
        - Segmentation: Mask R-CNN, U-Net, DeepLab, Segment Anything
        
        NLP & Generative AI:
        - Transformers, BERT, RoBERTa, T5, GPT variants
        - Stable Diffusion, ControlNet, LoRA fine-tuning
        - Text generation, sentiment analysis, NER, summarization
        - Vector databases: Pinecone, Weaviate, ChromaDB, FAISS
        
        Data Processing & Analytics:
        - Pandas, NumPy, Polars, Dask for large-scale processing
        - Data annotation: LabelImg, CVAT, Label Studio, Roboflow
        - ETL pipelines with Apache Airflow, Prefect
        - Feature stores: Feast, Tecton
        
        MLOps & Deployment:
        - Docker, Kubernetes, Helm charts
        - MLflow, Weights & Biases, Neptune for experiment tracking
        - CI/CD: GitHub Actions, GitLab CI, Jenkins
        - Model serving: FastAPI, Flask, BentoML, Seldon Core
        - Monitoring: Prometheus, Grafana, DataDog
        
        Cloud Infrastructure:
        - AWS: EC2 (p3.xlarge, p4d.24xlarge), SageMaker, Lambda, ECS
        - Google Cloud: Compute Engine (A100, V100 GPUs), Vertex AI, GKE
        - Azure: ML Studio, Container Instances, AKS
        - Edge deployment: Raspberry Pi 4, NVIDIA Jetson, Intel NUC
        
        API Development & Integration:
        - FastAPI, Django REST Framework, GraphQL
        - gRPC for high-performance microservices
        - WebSocket for real-time applications
        - Redis, PostgreSQL, MongoDB for data persistence
        """
        
        system_prompt = """
        You are a Senior AI/ML Solutions Architect creating clean, technical project flows.
        
        Generate concise implementation plans that:
        - Focus on essential tools and technologies only
        - Avoid version numbers, metrics, or hardware specifications
        - Keep each step brief and actionable
        - Use modern, relevant technologies
        - Maintain professional technical accuracy
        
        Create flow diagrams that are clean and easy to read.
        """
        
        user_prompt = f"""
        PROJECT SPECIFICATION:
        Title: {title}
        Description: {description}
        Required Skills: {skills}
        
        {technical_stack_context}
        
        DELIVERABLE REQUIREMENTS:
        Create a clean, technical implementation plan with specific tools and technologies.
        
        Format as numbered steps with this exact structure:
        1. Step Name: Tool1, Tool2, Framework3
        2. Step Name: Tool1, Tool2, Framework3
        
        Requirements:
        - List only the essential tools and technologies for each step
        - NO version numbers, performance metrics, or hardware specifications
        - Keep each step concise and focused on the core technologies
        - Use modern, relevant tools from the technical stack provided
        - Focus on the implementation approach, not detailed configurations
        
        NO markdown formatting. Only numbered steps with colons.
        Keep responses clean and concise for flow diagrams.
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.2,  # Lower temperature for more consistent technical output
                max_tokens=2048,
                top_p=0.9
            )
            
            project_flow = chat_completion.choices[0].message.content.strip()
            parsed_steps = self._parse_project_steps(project_flow)
            
            logger.info(f"Generated project flow with {len(parsed_steps)} steps")
            return project_flow, parsed_steps
            
        except Exception as e:
            logger.error(f"Error generating project flow: {e}")
            raise
    
    def generate_professional_cover_letter(
        self, 
        title: str, 
        description: str, 
        skills: str,
        client_budget: Optional[str] = None,
        timeline: Optional[str] = None
    ) -> str:
       """
Generate a sophisticated, results-oriented cover letter for technical proposals using the CO-STAR prompting technique and respond in the following format:

- Pitch (first line)
- Related experience
- Portfolio mention
- Ask to contact/DM if interested

Args:
    title (str): Project title
    description (str): Project description
    skills (str): Required skills
    client_budget (str, optional): Project budget range
    timeline (str, optional): Expected timeline

Returns:
    str: A professional cover letter with fixed format and tone
"""

system_prompt = """
You are a Senior Technical Lead writing high-converting technical cover letters for Upwork proposals. Use the CO-STAR framework to guide your tone, structure, and relevance.

CO-STAR Breakdown:
- **Context**: You’re applying to a job post on Upwork related to AI/ML or software engineering.
- **Objective**: Generate a concise, effective cover letter that matches the client’s needs with your qualifications.
- **Style**: Clear, professional, and focused on measurable results.
- **Tone**: Friendly, confident, client-centered.
- **Audience**: Clients posting technical jobs (AI, ML, software) on Upwork. May or may not be highly technical.
- **Response Format** (strictly follow):
    1. **Pitch (1-line hook)** – A direct, compelling summary of why you're a fit.
    2. **Related Experience** – Specific achievements aligned with the project.
    3. **Portfolio** – Briefly mention portfolio or past work links.
    4. **CTA** – Invite client to message you for more discussion or next steps.
"""

budget_context = f"Budget Consideration: {client_budget}" if client_budget else ""
timeline_context = f"Timeline Requirement: {timeline}" if timeline else ""

user_prompt = f"""
PROJECT DETAILS:
Title: {title}
Description: {description}
Required Skills: {skills}
{budget_context}
{timeline_context}

TECHNICAL BACKGROUND TO INCLUDE:
- 50+ successful AI/ML deployments
- PyTorch, TensorFlow, Hugging Face
- Cloud-scale deployments (AWS, GCP, Azure with GPUs)
- MLOps (CI/CD, model monitoring, retraining automation)
- Optimization for real-time inference (TensorRT, ONNX, quantization)
- Regulatory compliance: GDPR, HIPAA, SOC2

PROJECT SUCCESS STORIES:
- Object Detection: 99.2% accuracy at 1000+ FPS
- NLP: Custom LLM reduced hallucination by 85%
- MLOps: Pipelines reducing drift by 60%
- Edge AI: On-device inference under 50ms

TASK:
Using the above details and CO-STAR approach, generate a 150–200 word cover letter in the following format:
- Pitch (hook sentence)
- Related experience
- Portfolio mention
- Invitation to contact/DM
"""

        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.4,  # Slightly higher temperature for more natural writing
                max_tokens=1024,
                top_p=0.95
            )
            
            cover_letter = chat_completion.choices[0].message.content.strip()
            logger.info("Generated professional cover letter")
            return cover_letter
            
        except Exception as e:
            logger.error(f"Error generating cover letter: {e}")
            raise
    
    def _parse_project_steps(self, project_flow: str) -> Dict[str, str]:
        """
        Parse the generated project flow into a structured dictionary.
        
        Args:
            project_flow (str): Raw project flow text
            
        Returns:
            Dict[str, str]: Parsed steps with step names as keys and tools as values
        """
        steps_dict = {}
        
        try:
            for line in project_flow.split('\n'):
                line = line.strip()
                if line and ':' in line and any(char.isdigit() for char in line[:5]):
                    # Extract step number and content
                    step_part, tool_part = line.split(':', 1)
                    
                    # Clean step title (remove numbering)
                    step_title = step_part.strip()
                    if '.' in step_title:
                        step_title = step_title.split('.', 1)[-1].strip()
                    
                    tools = tool_part.strip()
                    if step_title and tools:
                        steps_dict[step_title] = tools
                        
            logger.info(f"Parsed {len(steps_dict)} project steps")
            return steps_dict
            
        except Exception as e:
            logger.error(f"Error parsing project steps: {e}")
            return {}

# Global instance for backward compatibility
planner = GroqProjectPlanner()

def get_project_plan(title: str, description: str, skills: str) -> Tuple[str, Dict[str, str]]:
    """Legacy function wrapper for backward compatibility."""
    return planner.generate_technical_project_flow(title, description, skills)

def get_cover_letter(title: str, description: str, skills: str) -> str:
    """Legacy function wrapper for backward compatibility."""
    return planner.generate_professional_cover_letter(title, description, skills)
