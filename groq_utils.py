import streamlit as st
from groq import Groq
from typing import Dict, Tuple, Optional, List
import logging
import requests
import json
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalDomainMapper:
    """
    Maps client-mentioned skills to deep technical domain specifications.
    Expands broad domains into comprehensive technical stacks and methodologies.
    """
    
    def __init__(self):
        """Initialize domain mapping with comprehensive technical specifications."""
        self.domain_mappings = {
            'data_science': {
                'keywords': ['data science', 'data scientist', 'analytics', 'statistical analysis', 'predictive modeling'],
                'deep_stack': {
                    'core_frameworks': ['Pandas', 'NumPy', 'Scikit-learn', 'SciPy', 'Statsmodels'],
                    'visualization': ['Matplotlib', 'Seaborn', 'Plotly', 'Bokeh', 'Altair'],
                    'ml_algorithms': ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Linear/Logistic Regression'],
                    'statistical_methods': ['Hypothesis Testing', 'A/B Testing', 'Time Series Analysis', 'Bayesian Statistics'],
                    'data_processing': ['Feature Engineering', 'Data Cleaning', 'ETL Pipelines', 'Data Validation'],
                    'deployment': ['Jupyter Notebooks', 'Streamlit', 'Flask', 'Docker', 'AWS SageMaker']
                },
                'methodologies': ['CRISP-DM', 'KDD Process', 'Cross-validation', 'Feature Selection', 'Model Interpretability']
            },
            
            'computer_vision': {
                'keywords': ['computer vision', 'cv', 'image processing', 'object detection', 'image recognition', 'opencv'],
                'deep_stack': {
                    'core_frameworks': ['OpenCV', 'PIL/Pillow', 'ImageIO', 'Albumentations', 'imgaug'],
                    'deep_learning': ['PyTorch', 'TensorFlow', 'Keras', 'torchvision', 'tf.keras.applications'],
                    'architectures': ['CNN', 'ResNet', 'EfficientNet', 'Vision Transformers', 'MobileNet'],
                    'detection_models': ['YOLO (v5/v8/v11)', 'Faster R-CNN', 'SSD', 'RetinaNet', 'DETR'],
                    'segmentation': ['U-Net', 'Mask R-CNN', 'DeepLab', 'Segment Anything Model (SAM)', 'FCN'],
                    'preprocessing': ['Image Augmentation', 'Normalization', 'Geometric Transformations', 'Color Space Conversion'],
                    'evaluation': ['mAP', 'IoU', 'Precision/Recall', 'F1-Score', 'Confusion Matrix']
                },
                'methodologies': ['Transfer Learning', 'Data Augmentation', 'Multi-scale Training', 'Test Time Augmentation']
            },
            
            'machine_learning': {
                'keywords': ['machine learning', 'ml', 'predictive modeling', 'classification', 'regression', 'clustering'],
                'deep_stack': {
                    'supervised_learning': ['Classification', 'Regression', 'Ensemble Methods', 'Support Vector Machines'],
                    'unsupervised_learning': ['K-Means', 'Hierarchical Clustering', 'DBSCAN', 'PCA', 't-SNE'],
                    'frameworks': ['Scikit-learn', 'XGBoost', 'LightGBM', 'CatBoost', 'Optuna'],
                    'deep_learning': ['Neural Networks', 'Backpropagation', 'Gradient Descent', 'Regularization'],
                    'model_selection': ['Cross-validation', 'Grid Search', 'Random Search', 'Bayesian Optimization'],
                    'evaluation': ['Accuracy', 'Precision/Recall', 'ROC-AUC', 'RMSE', 'MAE'],
                    'deployment': ['Model Serialization', 'API Development', 'Model Monitoring', 'A/B Testing']
                },
                'methodologies': ['Feature Engineering', 'Hyperparameter Tuning', 'Model Interpretability', 'Bias Detection']
            },
            
            'natural_language_processing': {
                'keywords': ['nlp', 'natural language processing', 'text analysis', 'sentiment analysis', 'text mining'],
                'deep_stack': {
                    'preprocessing': ['Tokenization', 'Stemming', 'Lemmatization', 'Stop Words', 'Text Cleaning'],
                    'traditional_methods': ['TF-IDF', 'N-grams', 'Bag of Words', 'Word2Vec', 'GloVe'],
                    'modern_nlp': ['Transformers', 'BERT', 'RoBERTa', 'GPT', 'T5', 'DistilBERT'],
                    'frameworks': ['NLTK', 'spaCy', 'Transformers (Hugging Face)', 'Gensim', 'TextBlob'],
                    'tasks': ['Named Entity Recognition', 'Part-of-Speech Tagging', 'Dependency Parsing', 'Coreference Resolution'],
                    'applications': ['Sentiment Analysis', 'Text Classification', 'Question Answering', 'Text Summarization'],
                    'embeddings': ['Word Embeddings', 'Sentence Embeddings', 'Document Embeddings', 'Contextual Embeddings']
                },
                'methodologies': ['Transfer Learning', 'Fine-tuning', 'Prompt Engineering', 'Few-shot Learning']
            },
            
            'deep_learning': {
                'keywords': ['deep learning', 'neural networks', 'dl', 'artificial neural networks', 'cnn', 'rnn', 'lstm'],
                'deep_stack': {
                    'frameworks': ['PyTorch', 'TensorFlow', 'Keras', 'JAX', 'Lightning'],
                    'architectures': ['CNN', 'RNN', 'LSTM', 'GRU', 'Transformer', 'Autoencoder'],
                    'optimization': ['Adam', 'SGD', 'RMSprop', 'Learning Rate Scheduling', 'Gradient Clipping'],
                    'regularization': ['Dropout', 'Batch Normalization', 'Layer Normalization', 'Weight Decay'],
                    'advanced_techniques': ['Transfer Learning', 'Multi-task Learning', 'Meta-learning', 'Self-supervised Learning'],
                    'hardware_acceleration': ['GPU Computing', 'CUDA', 'Mixed Precision Training', 'Distributed Training'],
                    'model_optimization': ['Quantization', 'Pruning', 'Knowledge Distillation', 'ONNX Conversion']
                },
                'methodologies': ['Gradient Descent', 'Backpropagation', 'Hyperparameter Tuning', 'Model Architecture Search']
            },
            
            'data_engineering': {
                'keywords': ['data engineering', 'etl', 'data pipeline', 'big data', 'data infrastructure'],
                'deep_stack': {
                    'data_processing': ['Apache Spark', 'Pandas', 'Dask', 'Polars', 'Apache Beam'],
                    'databases': ['PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'InfluxDB'],
                    'cloud_platforms': ['AWS', 'Google Cloud', 'Azure', 'Snowflake', 'Databricks'],
                    'orchestration': ['Apache Airflow', 'Prefect', 'Dagster', 'Luigi', 'Kubeflow'],
                    'streaming': ['Apache Kafka', 'Apache Storm', 'Apache Flink', 'Amazon Kinesis'],
                    'containerization': ['Docker', 'Kubernetes', 'Apache Spark on K8s', 'Helm Charts'],
                    'monitoring': ['Prometheus', 'Grafana', 'ELK Stack', 'DataDog', 'New Relic']
                },
                'methodologies': ['Data Modeling', 'Schema Design', 'Data Quality', 'Data Governance', 'CDC (Change Data Capture)']
            },
            
            'web_development': {
                'keywords': ['web development', 'frontend', 'backend', 'full stack', 'react', 'django', 'flask'],
                'deep_stack': {
                    'frontend': ['React', 'Vue.js', 'Angular', 'JavaScript', 'TypeScript', 'HTML5', 'CSS3'],
                    'backend': ['Django', 'Flask', 'FastAPI', 'Node.js', 'Express.js', 'Spring Boot'],
                    'databases': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'SQLite'],
                    'deployment': ['Docker', 'Kubernetes', 'AWS', 'Heroku', 'Vercel', 'Netlify'],
                    'testing': ['Jest', 'Pytest', 'Selenium', 'Cypress', 'Unit Testing', 'Integration Testing'],
                    'api_development': ['REST APIs', 'GraphQL', 'WebSockets', 'gRPC', 'API Documentation'],
                    'state_management': ['Redux', 'Vuex', 'Context API', 'MobX', 'Zustand']
                },
                'methodologies': ['Agile Development', 'Test-Driven Development', 'CI/CD', 'Version Control (Git)', 'Code Review']
            }
        }
    
    def identify_domains(self, skills_text: str) -> List[str]:
        """
        Identify technical domains from client's skill requirements.
        
        Args:
            skills_text (str): Client's mentioned skills
            
        Returns:
            List[str]: List of identified domain keys
        """
        skills_lower = skills_text.lower()
        identified_domains = []
        
        for domain_key, domain_config in self.domain_mappings.items():
            for keyword in domain_config['keywords']:
                if keyword in skills_lower:
                    identified_domains.append(domain_key)
                    break
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(identified_domains))
    
    def get_domain_deep_stack(self, domain_keys: List[str]) -> Dict[str, any]:
        """
        Get comprehensive technical stack for identified domains.
        
        Args:
            domain_keys (List[str]): List of domain keys
            
        Returns:
            Dict[str, any]: Comprehensive technical specifications
        """
        combined_stack = {
            'frameworks': set(),
            'tools': set(),
            'methodologies': set(),
            'domain_specific': {}
        }
        
        for domain_key in domain_keys:
            if domain_key in self.domain_mappings:
                domain_config = self.domain_mappings[domain_key]
                
                # Add domain-specific stack
                combined_stack['domain_specific'][domain_key] = domain_config['deep_stack']
                
                # Collect all frameworks and tools
                for category, items in domain_config['deep_stack'].items():
                    if isinstance(items, list):
                        combined_stack['frameworks'].update(items)
                
                # Add methodologies
                combined_stack['methodologies'].update(domain_config.get('methodologies', []))
        
        # Convert sets back to lists for JSON serialization
        combined_stack['frameworks'] = list(combined_stack['frameworks'])
        combined_stack['tools'] = list(combined_stack['tools'])
        combined_stack['methodologies'] = list(combined_stack['methodologies'])
        
        return combined_stack

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
    
    def search_domain_specific_trends(self, domain: str, max_results: int = 5) -> List[Dict]:
        """
        Search for domain-specific trends and best practices.
        
        Args:
            domain (str): Technical domain (e.g., "computer vision", "data science")
            max_results (int): Maximum number of results
            
        Returns:
            List[Dict]: Search results with domain-specific insights
        """
        if not self.search_enabled or not self.active_api:
            return []
        
        query = f"{domain} best practices tools frameworks 2024 2025 latest trends"
        return self.search_latest_tools(query, max_results)
    
    def search_latest_tools(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for latest information using the available API."""
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
            "query": query,
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
            "q": query,
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
            "q": query,
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
            "q": query,
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
            "q": query,
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

class GroqProjectPlanner:
    """
    Enhanced AI project planning system with deep domain expertise.
    Automatically expands broad skills into comprehensive technical implementations.
    """
    
    def __init__(self):
        """Initialize the Groq client, web search service, and domain mapper."""
        try:
            self.client = Groq(api_key=st.secrets["groq"]["api_key"])
            self.model_name = "llama3-70b-8192"
            self.web_search = WebSearchService()
            self.domain_mapper = TechnicalDomainMapper()
            logger.info("Enhanced Groq client with domain mapping initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def _build_domain_specific_context(self, skills: str, use_web_search: bool = True) -> str:
        """
        Build comprehensive domain-specific technical context.
        
        Args:
            skills (str): Client's mentioned skills
            use_web_search (bool): Whether to include current trends
            
        Returns:
            str: Domain-specific technical context
        """
        # Identify technical domains from skills
        identified_domains = self.domain_mapper.identify_domains(skills)
        
        if not identified_domains:
            return ""
        
        # Get deep technical stack for identified domains
        domain_stack = self.domain_mapper.get_domain_deep_stack(identified_domains)
        
        context_parts = []
        
        # Add domain-specific technical stacks
        for domain_key, domain_config in domain_stack['domain_specific'].items():
            domain_name = domain_key.replace('_', ' ').title()
            context_parts.append(f"\n=== {domain_name.upper()} DEEP TECHNICAL STACK ===")
            
            for category, tools in domain_config.items():
                if isinstance(tools, list) and tools:
                    category_name = category.replace('_', ' ').title()
                    context_parts.append(f"\n{category_name}:")
                    context_parts.append(f"- {', '.join(tools[:8])}")  # Limit to first 8 items
        
        # Add current trends from web search
        if use_web_search and self.web_search.search_enabled:
            try:
                for domain_key in identified_domains[:2]:  # Limit to first 2 domains
                    domain_name = domain_key.replace('_', ' ')
                    trends = self.web_search.search_domain_specific_trends(domain_name, 3)
                    
                    if trends:
                        context_parts.append(f"\n=== CURRENT {domain_name.upper()} TRENDS 2024-2025 ===")
                        for trend in trends:
                            if trend.get('snippet'):
                                context_parts.append(f"• {trend['snippet'][:200]}...")
                        
            except Exception as e:
                logger.error(f"Failed to get domain trends: {e}")
        
        return "\n".join(context_parts)
    
    def generate_technical_project_flow(
        self, 
        title: str, 
        description: str, 
        skills: str,
        use_web_search: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate a comprehensive domain-specific technical project flow.
        
        Args:
            title (str): Project title
            description (str): Detailed project description
            skills (str): Required technical skills
            use_web_search (bool): Whether to include web search for current trends
            
        Returns:
            Tuple[str, Dict[str, str]]: Complete project flow and parsed steps dictionary
        """
        
        # Build domain-specific context
        domain_context = self._build_domain_specific_context(skills, use_web_search)
        
        # Identify domains for specialized prompting
        identified_domains = self.domain_mapper.identify_domains(skills)
        domain_focus = ", ".join([d.replace('_', ' ').title() for d in identified_domains])
        
        system_prompt = f"""
        You are a Senior Technical Architect specializing in {domain_focus if domain_focus else 'Advanced Technical Solutions'}.
        
        Create DOMAIN-SPECIFIC project flows that:
        - Dive DEEP into the technical domains mentioned by the client
        - Use advanced, domain-specific tools and frameworks
        - Follow industry best practices for the identified domains
        - Include specialized methodologies and techniques
        - Demonstrate expert-level understanding of the technical domain
        - Progress from foundational setup to advanced implementation
        
        CRITICAL: When a client mentions broad domains like "data science", "computer vision", or "machine learning",
        create flows that showcase DEEP expertise in those specific domains using the most relevant and advanced tools.
        
        Focus Areas: {domain_focus if domain_focus else 'General Technical Implementation'}
        """
        
        user_prompt = f"""
        CLIENT PROJECT REQUIREMENTS:
        Title: {title}
        Description: {description}
        Required Skills: {skills}
        
        DOMAIN-SPECIFIC TECHNICAL CONTEXT:
        {domain_context}
        
        TASK: Create a deep, domain-specific technical implementation flow that demonstrates expert-level knowledge in the client's mentioned domains.
        
        REQUIREMENTS:
        - Each step should use advanced, domain-specific tools and techniques
        - Progress from setup through advanced implementation
        - Include domain-specific best practices and methodologies
        - Use the most current and relevant tools for each identified domain
        - Demonstrate deep technical expertise beyond basic implementations
        
        Format: Numbered steps with tools/frameworks:
        1. Step Name: Tool1, Tool2, Framework3
        2. Step Name: Tool1, Tool2, Framework3
        
        Create 6-10 comprehensive steps that show mastery of the technical domains.
        NO markdown formatting. Only numbered steps with colons.
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            project_flow = chat_completion.choices[0].message.content.strip()
            parsed_steps = self._parse_project_steps(project_flow)
            
            logger.info(f"Generated domain-specific project flow with {len(parsed_steps)} steps for domains: {domain_focus}")
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
        Generate a domain-specific professional cover letter.
        """
        
        # Get domain-specific context
        domain_context = self._build_domain_specific_context(skills, use_web_search)
        identified_domains = self.domain_mapper.identify_domains(skills)
        domain_focus = ", ".join([d.replace('_', ' ').title() for d in identified_domains])
        
        budget_context = f"Budget: {client_budget}" if client_budget else ""
        timeline_context = f"Timeline: {timeline}" if timeline else ""
        
        system_prompt = f"""
        You are a Senior Technical Lead with deep expertise in {domain_focus if domain_focus else 'Advanced Technical Solutions'}.
        
        Write cover letters that demonstrate DEEP domain expertise by:
        - Mentioning specific, advanced tools and techniques from the client's domain
        - Showing understanding of domain-specific challenges and solutions
        - Using technical terminology that proves expertise in the identified domains
        - Highlighting relevant experience with domain-specific implementations
        - Addressing the technical complexity of the client's specific domain needs
        
        Target Domains: {domain_focus if domain_focus else 'General Technical'}
        """
        
        user_prompt = f"""
        CLIENT REQUIREMENTS:
        Title: {title}
        Description: {description}
        Required Skills: {skills}
        {budget_context}
        {timeline_context}
        
        DOMAIN-SPECIFIC CONTEXT:
        {domain_context}
        
        TASK: Write a 150-200 word cover letter that demonstrates DEEP expertise in the client's specific technical domains.
        
        FOCUS:
        - Use domain-specific terminology and advanced concepts
        - Mention relevant tools, frameworks, and methodologies from the domain context
        - Show understanding of domain-specific challenges
        - Highlight experience with similar domain-specific projects
        - Demonstrate thought leadership in the client's technical areas
        
        Structure:
        1. Hook with domain expertise
        2. Specific domain experience and achievements
        3. Relevant portfolio examples
        4. Technical discussion invitation
        
        Write as an expert in: {domain_focus if domain_focus else 'advanced technical solutions'}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.4,
                max_tokens=1024,
                top_p=0.95
            )
            
            cover_letter = chat_completion.choices[0].message.content.strip()
            logger.info(f"Generated domain-specific cover letter for: {domain_focus}")
            return cover_letter
            
        except Exception as e:
            logger.error(f"Error generating cover letter: {e}")
            raise
    
    def _parse_project_steps(self, project_flow: str) -> Dict[str, str]:
        """Parse the generated project flow into a structured dictionary."""
        steps_dict = {}
        
        try:
            for line in project_flow.split('\n'):
                line = line.strip()
                if line and ':' in line
                
    
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
