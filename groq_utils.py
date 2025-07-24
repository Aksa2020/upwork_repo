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
    Web search service supporting multiple APIs: Tavily, Brave, Google Custom Search, and Bing.
    """
    
    def __init__(self):
        """Initialize web search service with multiple API options."""
        try:
            # Multiple API options - will use the first available
            self.apis = {
                'tavily': {
                    'key': st.secrets.get("tavily", {}).get("api_key", ""),
                    'url': "https://api.tavily.com/search",
                    'description': "AI-optimized search API"
                },
                'brave': {
                    'key': st.secrets.get("brave", {}).get("api_key", ""),
                    'url': "https://api.search.brave.com/res/v1/web/search",
                    'description': "Independent search index"
                },
                'google': {
                    'key': st.secrets.get("google", {}).get("api_key", ""),
                    'cx': st.secrets.get("google", {}).get("cx", ""),  # Custom Search Engine ID
                    'url': "https://www.googleapis.com/customsearch/v1",
                    'description': "Google Custom Search"
                },
                'bing': {
                    'key': st.secrets.get("bing", {}).get("api_key", ""),
                    'url': "https://api.bing.microsoft.com/v7.0/search",
                    'description': "Microsoft Bing Search"
                },
                'serpapi': {
                    'key': st.secrets.get("serpapi", {}).get("api_key", ""),
                    'url': "https://serpapi.com/search",
                    'description': "SerpAPI (fallback)"
                }
            }
            
            # Find first available API
            self.active_api = None
            for api_name, api_config in self.apis.items():
                if api_config['key'] and (api_name != 'google' or api_config.get('cx')):
                    self.active_api = api_name
                    logger.info(f"Using {api_config['description']} for web search")
                    break
            
            self.search_enabled = bool(self.active_api)
            if not self.search_enabled:
                logger.warning("Web search disabled - no API keys found")
                
        except Exception as e:
            logger.error(f"Failed to initialize web search: {e}")
            self.search_enabled = False
            self.active_api = None
    
    def search_latest_tools(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for latest information using the available API.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict]: Search results with titles, snippets, and links
        """
        if not self.search_enabled or not self.active_api:
            logger.info("Web search disabled, returning empty results")
            return []
        
        try:
            if self.active_api == 'tavily':
                return self._search_tavily(query, max_results)
            elif self.active_api == 'brave':
                return self._search_brave(query, max_results)
            elif self.active_api == 'google':
                return self._search_google(query, max_results)
            elif self.active_api == 'bing':
                return self._search_bing(query, max_results)
            elif self.active_api == 'serpapi':
                return self._search_serpapi(query, max_results)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    def _search_tavily(self, query: str, max_results: int) -> List[Dict]:
        """Search using Tavily API (AI-optimized)"""
        api_config = self.apis['tavily']
        
        payload = {
            "api_key": api_config['key'],
            "query": f"{query} 2024 2025 latest tools frameworks",
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        response = requests.post(api_config['url'], json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "results" in data:
            for result in data["results"][:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("content", "")[:300],
                    "link": result.get("url", ""),
                    "source": "Tavily"
                })
        
        return results
    
    def _search_brave(self, query: str, max_results: int) -> List[Dict]:
        """Search using Brave Search API"""
        api_config = self.apis['brave']
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_config['key']
        }
        
        params = {
            "q": f"{query} 2024 2025 latest tools frameworks",
            "count": max_results,
            "safesearch": "moderate"
        }
        
        response = requests.get(api_config['url'], headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"][:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("description", ""),
                    "link": result.get("url", ""),
                    "source": "Brave"
                })
        
        return results
    
    def _search_google(self, query: str, max_results: int) -> List[Dict]:
        """Search using Google Custom Search API"""
        api_config = self.apis['google']
        
        params = {
            "key": api_config['key'],
            "cx": api_config['cx'],
            "q": f"{query} 2024 2025 latest tools frameworks",
            "num": min(max_results, 10),  # Google Custom Search max is 10
            "safe": "active"
        }
        
        response = requests.get(api_config['url'], params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "items" in data:
            for result in data["items"][:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "source": "Google"
                })
        
        return results
    
    def _search_bing(self, query: str, max_results: int) -> List[Dict]:
        """Search using Bing Search API"""
        api_config = self.apis['bing']
        
        headers = {
            "Ocp-Apim-Subscription-Key": api_config['key']
        }
        
        params = {
            "q": f"{query} 2024 2025 latest tools frameworks",
            "count": max_results,
            "safeSearch": "Moderate"
        }
        
        response = requests.get(api_config['url'], headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "webPages" in data and "value" in data["webPages"]:
            for result in data["webPages"]["value"][:max_results]:
                results.append({
                    "title": result.get("name", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("url", ""),
                    "source": "Bing"
                })
        
        return results
    
    def _search_serpapi(self, query: str, max_results: int) -> List[Dict]:
        """Search using SerpAPI (fallback)"""
        api_config = self.apis['serpapi']
        
        params = {
            "api_key": api_config['key'],
            "engine": "google",
            "q": f"{query} 2024 2025 latest tools frameworks",
            "num": max_results,
            "safe": "active"
        }
        
        response = requests.get(api_config['url'], params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "organic_results" in data:
            for result in data["organic_results"][:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "source": "SerpAPI"
                })
        
        return results
    
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
        Get current technology context by searching specifically for the client's mentioned skills.
        
        Args:
            project_type (str): Type of project (extracted from title/description)
            skills (str): Required skills mentioned by client
            
        Returns:
            str: Current technology context focused on client's specific skills
        """
        if not self.web_search.search_enabled:
            return ""
        
        # Extract and prioritize specific skills mentioned by client
        skill_keywords = [skill.strip().lower() for skill in skills.split(',')]
        
        # Search specifically for each mentioned skill
        current_context = []
        processed_skills = set()
        
        for skill in skill_keywords[:3]:  # Focus on top 3 mentioned skills
            if skill and skill not in processed_skills:
                processed_skills.add(skill)
                
                # Create targeted search query for this specific skill
                search_query = f"{skill} best practices tools frameworks 2024 2025"
                trending_info = self.web_search.get_trending_ai_tools(search_query)
                
                if trending_info:
                    current_context.append(f"\nCURRENT {skill.upper()} BEST PRACTICES & TOOLS:\n{trending_info}")
        
        # If no specific skills found, fall back to general project context
        if not current_context and project_type:
            fallback_query = f"{project_type} latest tools"
            trending_info = self.web_search.get_trending_ai_tools(fallback_query)
            if trending_info:
                current_context.append(f"\nCURRENT PROJECT-RELEVANT TOOLS:\n{trending_info}")
        
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
        You are a Senior AI/ML Solutions Architect creating technical project flows that STRICTLY FOCUS on the client's specified skills.
        
        Generate implementation plans that:
        - PRIORITIZE the exact skills and technologies mentioned by the client
        - Build the entire flow around the client's specified skill requirements
        - Only include tools and frameworks that directly relate to the mentioned skills
        - Avoid generic or unrelated technologies not specified by the client
        - Keep each step focused on implementing the client's specific skill requirements
        - Use the client's mentioned skills as the primary foundation for all technical decisions
        
        CRITICAL: The client's specified skills should be the main focus and foundation of every step in the project flow.
        
        Create flows that demonstrate deep expertise in the client's exact requirements.
        """
        
        user_prompt = f"""
        CLIENT'S SPECIFIC REQUIREMENTS:
        Title: {title}
        Description: {description}
        REQUIRED SKILLS (PRIORITY FOCUS): {skills}
        
        BASE TECHNICAL KNOWLEDGE:
        {technical_stack_context}
        {current_tech_context}
        
        DELIVERABLE REQUIREMENTS:
        Create a technical implementation plan that is LASER-FOCUSED on the client's specified skills: "{skills}"
        
        Format as numbered steps with this exact structure:
        1. Step Name: Tool1, Tool2, Framework3
        2. Step Name: Tool1, Tool2, Framework3
        
        STRICT REQUIREMENTS:
        - Every step must directly utilize or implement the client's specified skills: "{skills}"
        - Prioritize tools and frameworks that are specifically mentioned or directly related to: "{skills}"
        - If the client mentions specific technologies (e.g., "PyTorch", "React", "AWS"), make them central to the flow
        - Avoid generic tools unless they directly support the client's specified skill requirements
        - Each step should demonstrate expertise in the exact skills the client is looking for
        - NO version numbers, performance metrics, or hardware specifications
        - Focus on the client's skill requirements, not general best practices
        
        The project flow should read like a demonstration of mastery in the client's specific skill requirements.
        
        NO markdown formatting. Only numbered steps with colons.
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
        
        # Get current market insights focused on client's specific skills
        market_context = ""
        if use_web_search:
            try:
                # Search specifically for the client's mentioned skills
                skill_specific_trends = self._get_current_tech_context(title, skills)
                if skill_specific_trends:
                    market_context = f"\n\nCLIENT'S SKILL-SPECIFIC MARKET INSIGHTS:\n{skill_specific_trends[:600]}..."
                    logger.info("Added skill-specific market insights to cover letter context")
            except Exception as e:
                logger.error(f"Failed to get skill-specific market context: {e}")
        
        system_prompt = """
        You are a Senior Technical Lead writing high-converting cover letters that DIRECTLY ADDRESS the client's specific skill requirements.
        
        CO-STAR Framework with Skill-Focus:
        - **Context**: Applying to a technical job with specific skill requirements on Upwork.
        - **Objective**: Demonstrate EXACT expertise in the client's specified skills.
        - **Style**: Skill-focused, results-driven, directly addressing client's technical needs.
        - **Tone**: Confident expertise in the client's specific skill areas.
        - **Audience**: Clients looking for specific technical skills and expertise.
        - **Response Format** (strictly follow):
        1. **Pitch** – Hook that directly mentions client's specified skills.
        2. **Related Experience** – Achievements using the EXACT skills the client mentioned.
        3. **Portfolio** – Examples specifically showcasing the client's required skills.
        4. **CTA** – Invite discussion about their specific skill requirements.
        
        CRITICAL: Every sentence should demonstrate expertise in the client's specified skills.
        """
        
        user_prompt = f"""
        CLIENT'S SPECIFIC SKILL REQUIREMENTS:
        Title: {title}
        Description: {description}
        REQUIRED SKILLS (PRIMARY FOCUS): {skills}
        {budget_context}
        {timeline_context}
        {market_context}
        
        SKILL-SPECIFIC EXPERIENCE TO HIGHLIGHT:
        Based on the client's required skills "{skills}", emphasize relevant experience such as:
        - Direct projects using the client's specified technologies/skills
        - Quantifiable results achieved with the exact skills they mentioned
        - Advanced implementations in their specified skill areas
        - Problem-solving expertise in their particular technical domain
        - Current best practices in their specified skill requirements
        
        SUCCESS STORIES (ADAPT TO CLIENT'S SKILLS):
        - Select examples that directly showcase expertise in: "{skills}"
        - Emphasize achievements using the client's specified technologies
        - Highlight results that demonstrate mastery of their required skills
        - Focus on outcomes that matter for their specific technical needs
        
        TASK:
        Generate a 150–200 word cover letter that is LASER-FOCUSED on the client's skill requirements:
        - Pitch: Directly mention and demonstrate expertise in "{skills}"
        - Experience: Show specific achievements using the skills they mentioned
        - Portfolio: Reference projects that showcase their required skills
        - CTA: Invite discussion about solving their specific technical challenges
        
        Every sentence should reinforce your expertise in the client's specified skills: "{skills}"
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
