import streamlit as st
from groq import Groq
from typing import Dict, Tuple, Optional, List
import logging
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchService:
    """
    Web search service to get current information about technologies, tools, and trends.
    """
    
    def __init__(self):
        """Initialize web search service with API credentials."""
        try:
            # You can use multiple search APIs - here's an example with SerpAPI
            # Add your preferred search API key to st.secrets
            self.search_api_key = st.secrets.get("serpapi", {}).get("api_key", "")
            self.search_enabled = bool(self.search_api_key)
            if not self.search_enabled:
                logger.warning("Web search disabled - no API key found")
        except Exception as e:
            logger.error(f"Failed to initialize web search: {e}")
            self.search_enabled = False
    
    def search_latest_tools(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for latest information about tools, frameworks, or technologies.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict]: Search results with titles, snippets, and links
        """
        if not self.search_enabled:
            logger.info("Web search disabled, returning empty results")
            return []
        
        try:
            # Example using SerpAPI (you can replace with your preferred search service)
            search_url = "https://serpapi.com/search"
            params = {
                "api_key": self.search_api_key,
                "engine": "google",
                "q": f"{query} 2024 2025 latest tools frameworks",
                "num": max_results,
                "safe": "active"
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            results = []
            
            if "organic_results" in search_data:
                for result in search_data["organic_results"][:max_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "source": result.get("source", "")
                    })
            
            logger.info(f"Found {len(results)} search results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    def get_trending_ai_tools(self, domain: str = "machine learning") -> str:
        """
        Get trending AI/ML tools and frameworks for a specific domain.
        
        Args:
            domain (str): Domain to search for (e.g., "machine learning", "computer vision")
            
        Returns:
            str: Formatted string of trending tools and technologies
        """
        query = f"best {domain} tools frameworks 2024 2025 trending"
        results = self.search_latest_tools(query, max_results=3)
        
        if not results:
            return ""
        
        trending_info = []
        for result in results:
            if result["snippet"]:
                trending_info.append(f"• {result['snippet'][:200]}...")
        
        return "\n".join(trending_info) if trending_info else ""

class GroqProjectPlanner:
    """
    Professional AI project planning system using Groq's LLaMA models with web search capabilities.
    
    Generates technical project flows and cover letters optimized for
    machine learning and AI development projects using current information.
    """
    
    def __init__(self):
        """Initialize the Groq client and web search service."""
        try:
            self.client = Groq(api_key=st.secrets["groq"]["api_key"])
            self.model_name = "llama3-70b-8192"
            self.web_search = WebSearchService()
            logger.info("Groq client and web search initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def _get_current_tech_context(self, project_type: str, skills: str) -> str:
        """
        Get current technology context by searching the web for latest tools and trends.
        
        Args:
            project_type (str): Type of project (extracted from title/description)
            skills (str): Required skills
            
        Returns:
            str: Current technology context from web search
        """
        if not self.web_search.search_enabled:
            return ""
        
        # Determine search domains based on project type and skills
        search_domains = []
        
        if any(keyword in skills.lower() or keyword in project_type.lower() 
               for keyword in ["computer vision", "cv", "image", "video", "detection", "segmentation"]):
            search_domains.append("computer vision")
        
        if any(keyword in skills.lower() or keyword in project_type.lower() 
               for keyword in ["nlp", "language", "text", "chatbot", "llm", "gpt"]):
            search_domains.append("natural language processing")
        
        if any(keyword in skills.lower() or keyword in project_type.lower() 
               for keyword in ["mlops", "deployment", "cloud", "production"]):
            search_domains.append("MLOps deployment")
        
        if not search_domains:
            search_domains = ["machine learning"]
        
        # Get trending information for each domain
        current_context = []
        for domain in search_domains[:2]:  # Limit to 2 domains to avoid too much content
            trending_info = self.web_search.get_trending_ai_tools(domain)
            if trending_info:
                current_context.append(f"\nCURRENT {domain.upper()} TRENDS (2024-2025):\n{trending_info}")
        
        return "\n".join(current_context)
    
    def generate_technical_project_flow(
        self, 
        title: str, 
        description: str, 
        skills: str,
        use_web_search: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate a comprehensive technical project flow with current tools and technologies.
        
        Args:
            title (str): Project title
            description (str): Detailed project description
            skills (str): Required technical skills
            use_web_search (bool): Whether to include web search for current trends
            
        Returns:
            Tuple[str, Dict[str, str]]: Complete project flow and parsed steps dictionary
        """
        
        technical_stack_context = """
        CORE TECHNICAL EXPERTISE & INFRASTRUCTURE:
        
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
        
        # Get current technology context from web search
        current_tech_context = ""
        if use_web_search:
            try:
                project_type = f"{title} {description}"
                current_tech_context = self._get_current_tech_context(project_type, skills)
                if current_tech_context:
                    logger.info("Added current technology trends from web search")
            except Exception as e:
                logger.error(f"Failed to get current tech context: {e}")
        
        system_prompt = """
        You are a Senior AI/ML Solutions Architect creating clean, technical project flows using the most current tools and technologies.
        
        Generate concise implementation plans that:
        - Prioritize the most current and trending tools when available
        - Focus on essential tools and technologies only
        - Avoid version numbers, metrics, or hardware specifications
        - Keep each step brief and actionable
        - Use modern, relevant technologies from both established and trending sources
        - Maintain professional technical accuracy
        
        When current trends are provided, incorporate the latest tools and approaches while maintaining the core technical foundation.
        
        Create flow diagrams that are clean and easy to read.
        """
        
        user_prompt = f"""
        PROJECT SPECIFICATION:
        Title: {title}
        Description: {description}
        Required Skills: {skills}
        
        {technical_stack_context}
        {current_tech_context}
        
        DELIVERABLE REQUIREMENTS:
        Create a clean, technical implementation plan incorporating both established and current trending tools.
        
        Format as numbered steps with this exact structure:
        1. Step Name: Tool1, Tool2, Framework3
        2. Step Name: Tool1, Tool2, Framework3
        
        Requirements:
        - Prioritize current/trending tools when they're superior to established ones
        - List only the essential tools and technologies for each step
        - NO version numbers, performance metrics, or hardware specifications
        - Keep each step concise and focused on the core technologies
        - Use modern, relevant tools from both the established stack and current trends
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
            
            logger.info(f"Generated project flow with {len(parsed_steps)} steps (web search: {use_web_search})")
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
        timeline: Optional[str] = None,
        use_web_search: bool = True
    ) -> str:
        """
        Generate a sophisticated, results-oriented cover letter with current technology insights.
        
        Args:
            title (str): Project title
            description (str): Project description
            skills (str): Required skills
            client_budget (str, optional): Project budget range
            timeline (str, optional): Expected timeline
            use_web_search (bool): Whether to include current trends in the cover letter
            
        Returns:
            str: A professional cover letter with current technology awareness
        """
        
        # Get current market insights for the cover letter
        market_context = ""
        if use_web_search:
            try:
                project_type = f"{title} {description}"
                current_trends = self._get_current_tech_context(project_type, skills)
                if current_trends:
                    market_context = f"\n\nCURRENT MARKET INSIGHTS:\n{current_trends[:500]}..."
                    logger.info("Added current market insights to cover letter context")
            except Exception as e:
                logger.error(f"Failed to get market context: {e}")
        
        system_prompt = """
        You are a Senior Technical Lead writing high-converting technical cover letters for Upwork proposals. Use the CO-STAR framework and current market awareness.
        
        CO-STAR Breakdown:
        - **Context**: You're applying to a job post on Upwork related to AI/ML or software engineering.
        - **Objective**: Generate a concise, effective cover letter that matches client needs with current expertise.
        - **Style**: Clear, professional, results-focused, and current with industry trends.
        - **Tone**: Friendly, confident, client-centered, and technically current.
        - **Audience**: Clients posting technical jobs (AI, ML, software) on Upwork.
        - **Response Format** (strictly follow):
        1. **Pitch (1-line hook)** – Direct, compelling summary highlighting current expertise.
        2. **Related Experience** – Specific achievements with current/trending technologies.
        3. **Portfolio** – Mention portfolio with current project examples.
        4. **CTA** – Invite client to discuss latest approaches and solutions.
        
        When current market insights are available, subtly incorporate awareness of trending tools and approaches.
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
        {market_context}
        
        TECHNICAL BACKGROUND TO INCLUDE:
        - 50+ successful AI/ML deployments with latest frameworks
        - Current expertise: PyTorch, TensorFlow, Hugging Face, latest LLMs
        - Cloud-scale deployments (AWS, GCP, Azure with latest GPU instances)
        - Modern MLOps (CI/CD, automated monitoring, real-time retraining)
        - Cutting-edge optimization (latest quantization, edge deployment)
        - Compliance expertise: GDPR, HIPAA, SOC2
        
        PROJECT SUCCESS STORIES (emphasize current approaches):
        - Latest Object Detection: 99.2% accuracy with current YOLO variants
        - Advanced NLP: Custom LLM fine-tuning reducing hallucination by 85%
        - Modern MLOps: Automated pipelines reducing model drift by 60%
        - Edge AI: Real-time inference optimization under 50ms
        
        TASK:
        Generate a 150–200 word cover letter incorporating current market awareness when available:
        - Pitch (hook sentence with current expertise)
        - Related experience (highlighting modern approaches)
        - Portfolio mention (current project examples)
        - Invitation to discuss latest solutions
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
            logger.info(f"Generated professional cover letter (web search: {use_web_search})")
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

def get_project_plan(title: str, description: str, skills: str, use_web_search: bool = True) -> Tuple[str, Dict[str, str]]:
    """Legacy function wrapper with web search option."""
    return planner.generate_technical_project_flow(title, description, skills, use_web_search)

def get_cover_letter(title: str, description: str, skills: str, use_web_search: bool = True) -> str:
    """Legacy function wrapper with web search option."""
    return planner.generate_professional_cover_letter(title, description, skills, use_web_search=use_web_search)
