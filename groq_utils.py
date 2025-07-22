"""
Enhanced Groq Utilities for AI Project Planning and Proposal Generation

This module provides sophisticated AI-driven project planning and cover letter generation
for technical proposals, with emphasis on modern ML/AI technologies and deployment strategies.
"""

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
        You are a Senior AI/ML Solutions Architect with 10+ years of experience in enterprise AI deployments.
        Your expertise spans computer vision, NLP, MLOps, and scalable AI infrastructure.
        
        Generate highly technical, implementation-ready project flows that demonstrate deep understanding of:
        - Modern AI/ML architecture patterns
        - Production deployment strategies  
        - Performance optimization techniques
        - Scalability and reliability considerations
        
        CRITICAL REQUIREMENTS:
        - Specify exact versions and configurations where relevant
        - Include performance benchmarks and hardware requirements
        - Address data security, model governance, and compliance
        - Mention specific deployment patterns (blue-green, canary, A/B testing)
        - Include monitoring, logging, and observability stack
        """
        
        user_prompt = f"""
        PROJECT SPECIFICATION:
        Title: {title}
        Description: {description}
        Required Skills: {skills}
        
        {technical_stack_context}
        
        DELIVERABLE REQUIREMENTS:
        Create a production-ready technical implementation plan that follows enterprise standards.
        
        Format as clean numbered steps with this exact structure:
        1. Step Name: Specific tools, frameworks, and configurations
        2. Step Name: Specific tools, frameworks, and configurations
        
        Each step must include:
        - Exact tool versions and configurations
        - Hardware/infrastructure specifications
        - Performance metrics and benchmarks
        - Integration points and data flow
        - Quality assurance and testing approaches
        
        Focus on:
        - Scalable architecture design
        - Production deployment readiness
        - Performance optimization
        - Monitoring and observability
        - Security and compliance considerations
        
        NO markdown formatting. Only numbered steps with colons.
        Minimum 8-12 comprehensive technical steps.
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
        Generate a sophisticated, results-oriented cover letter for technical proposals.
        
        Args:
            title (str): Project title
            description (str): Project description
            skills (str): Required skills
            client_budget (str, optional): Project budget range
            timeline (str, optional): Expected timeline
            
        Returns:
            str: Professional cover letter content
        """
        
        system_prompt = """
        You are a Senior Technical Lead writing proposals for enterprise AI/ML projects.
        Your cover letters should demonstrate:
        - Deep technical expertise and proven track record
        - Understanding of business impact and ROI
        - Risk mitigation and quality assurance approaches
        - Clear communication of complex technical concepts
        
        Style guidelines:
        - Professional yet personable tone
        - Specific technical details without overwhelming non-technical readers
        - Quantifiable results and success metrics
        - Clear next steps and engagement process
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
        
        TECHNICAL CREDENTIALS TO HIGHLIGHT:
        - 50+ successful AI/ML deployments in production
        - Expertise in modern ML frameworks (PyTorch, TensorFlow, Hugging Face)
        - Scalable deployment experience (AWS, GCP, Azure with GPU clusters)
        - MLOps implementation with automated CI/CD pipelines
        - Real-time inference optimization (TensorRT, ONNX, quantization)
        - Enterprise security and compliance (GDPR, HIPAA, SOC2)
        
        RECENT SUCCESS CASES:
        - Computer Vision: 99.2% accuracy object detection system processing 1000+ FPS
        - NLP: Custom LLM fine-tuning reducing hallucinations by 85%
        - MLOps: Automated retraining pipelines reducing model drift by 60%
        - Edge AI: Real-time inference on edge devices with <50ms latency
        
        Write a compelling 150-200 word cover letter that:
        1. Demonstrates immediate understanding of project requirements
        2. Highlights relevant technical expertise and past successes
        3. Addresses potential technical challenges proactively
        4. Provides clear next steps for engagement
        5. Maintains professional confidence without overselling
        
        Focus on value delivery and technical excellence.
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
