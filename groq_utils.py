import streamlit as st
from groq import Groq
from typing import Dict, Tuple, Optional
import logging
import requests
import json

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GroqProjectPlanner:
    """AI project planning system using Groq's LLaMA models."""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["groq"]["api_key"])
            self.model_name = "llama3-70b-8192"
            logger.info("Groq client initialized")
        except Exception as e:
            logger.error(f"Groq client init failed: {e}")
            raise

    def search_web(self, query: str, num_results: int = 5) -> str:
        """Search web using DuckDuckGo Instant Answer API."""
        st.info(f"🔍 Searching the web using DuckDuckGo for: '{query}'")
        logger.info(f"Performing DuckDuckGo search for query: {query}")

        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            search_context = ""

            if data.get("AbstractText"):
                search_context += f"- {data.get('Heading', 'Info')}: {data['AbstractText']}\n"

            related = data.get("RelatedTopics", [])
            for item in related:
                if isinstance(item, dict) and 'Text' in item:
                    search_context += f"- {item['Text']}\n"
                    if len(search_context.splitlines()) >= num_results:
                        break

            return search_context.strip() if search_context else "No relevant results found"

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Search failed: {str(e)}"

    def generate_technical_project_flow(self, title: str, description: str, skills: str) -> Tuple[str, Dict[str, str]]:
        """Generate technical project flow with tools and technologies."""
        search_query = f"latest {skills} tools frameworks 2024 2025"
        web_context = self.search_web(search_query, 3)

        tech_stack = """
        ML/AI: PyTorch, TensorFlow, Scikit-learn, Hugging Face, LangChain
        Computer Vision: YOLOv8, OpenCV, ResNet, EfficientNet, Mask R-CNN
        NLP: BERT, GPT, Stable Diffusion, Vector databases (Pinecone, ChromaDB)
        Data: Pandas, NumPy, Airflow, MLflow, Label Studio
        Deployment: Docker, Kubernetes, FastAPI, AWS/GCP/Azure, Redis
        """

        system_prompt = """Create clean technical project flows with:
        - Latest and most relevant tools from web search results
        - Essential tools only
        - Brief actionable steps
        - Modern technologies
        - No version numbers or specs"""

        user_prompt = f"""
        PROJECT: {title}
        DESCRIPTION: {description}
        SKILLS: {skills}

        LATEST TECH TRENDS (from web search):
        {web_context}

        CORE TECH STACK: {tech_stack}

        Create numbered steps: "1. Step Name: Tool1, Tool2, Tool3"
        Prioritize latest tools from web search when relevant.
        Keep concise. No markdown. Focus on implementation approach.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.2,
                max_tokens=2048
            )
            project_flow = response.choices[0].message.content.strip()
            parsed_steps = self._parse_steps(project_flow)
            logger.info(f"Generated {len(parsed_steps)} steps")
            return project_flow, parsed_steps

        except Exception as e:
            logger.error(f"Project flow error: {e}")
            raise

    def generate_professional_cover_letter(self, title: str, description: str, skills: str, 
                                           client_budget: Optional[str] = None) -> str:
        """Generate professional cover letter for technical proposals."""
        search_query = f"{title} {skills} industry trends requirements 2024"
        st.info(f"🔍 Searching industry context using DuckDuckGo for: '{search_query}'")
        industry_context = self.search_web(search_query, 2)
        logger.info(f"DuckDuckGo context: {industry_context}")

        system_prompt = """Write professional AI/ML project proposals showing:
        - Technical expertise and track record
        - Understanding of current industry trends
        - Business impact understanding  
        - Clear next steps"""

        budget_text = f"Budget: {client_budget}" if client_budget else ""

        user_prompt = f"""
        PROJECT: {title}
        DESCRIPTION: {description}
        SKILLS: {skills}
        {budget_text}

        CURRENT INDUSTRY CONTEXT (from web search):
        {industry_context}

        CREDENTIALS:
        - 50+ AI/ML deployments in production
        - Modern ML frameworks expertise
        - Cloud deployment experience (AWS/GCP/Azure)
        - MLOps and automated pipelines
        - 99%+ accuracy systems, <50ms latency optimization

        Write 150-200 word cover letter demonstrating:
        1. Pitch for the Project
        2. Relevant expertise aligned with current trends
        3. Project understanding with industry awareness
        4. Technical approach
        5. Clear next steps
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.4,
                max_tokens=1024
            )
            cover_letter = response.choices[0].message.content.strip()
            logger.info("✅ Cover letter generated successfully")
            return cover_letter

        except Exception as e:
            logger.error(f"Cover letter generation error: {e}")
            raise

    def _parse_steps(self, project_flow: str) -> Dict[str, str]:
        """Parse project flow into structured dictionary."""
        steps_dict = {}
        try:
            for line in project_flow.split('\n'):
                line = line.strip()
                if line and ':' in line and any(c.isdigit() for c in line[:5]):
                    step_part, tool_part = line.split(':', 1)
                    step_title = step_part.split('.', 1)[-1].strip()
                    tools = tool_part.strip()
                    if step_title and tools:
                        steps_dict[step_title] = tools
            logger.info(f"Parsed {len(steps_dict)} steps")
            return steps_dict
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {}

# Global instance and wrapper functions
planner = GroqProjectPlanner()

def get_project_plan(title: str, description: str, skills: str) -> Tuple[str, Dict[str, str]]:
    return planner.generate_technical_project_flow(title, description, skills)

def get_cover_letter(title: str, description: str, skills: str) -> str:
    return planner.generate_professional_cover_letter(title, description, skills)
