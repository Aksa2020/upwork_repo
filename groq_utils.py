import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["groq"]["api_key"])

def get_project_plan(title, description, skills):
    prompt = f"""
You are an AI engineer writing project plans for clients.

Job Title: {title}
Client Description: {description}
Required Skills: {skills}

Instead of generic steps, list the tools, technologies, and platforms you will use.
Structure it as a clear numbered list with the heading 'Proposed Project Flow with Technologies:'.
Example tools: LabelImg, CVAT, YOLOv8, PyTorch, TensorRT, Docker, Azure, Streamlit.

Use clean text, no markdown.

Respond in this clean format ONLY:

1. Step: Tools or Technologies
2. Step: Tools or Technologies
   ...

NO headings, no markdown, no bullet points. Only numbered steps with colon.
Be realistic with AI/ML tools.
Use tools like YOLOv8, PyTorch, FastAPI, Docker, Azure, etc., where appropriate.

We have completed projects using the following technologies, tools, and infrastructure: Python (Python3), OpenCV, Pandas, PyTorch, TensorFlow Object Detection API, Dlib, FaceRecognition API, CSRT Trackers, YOLOv8 / YOLOv11 / YOLOv12, FastAPI, Streamlit, LangChain, Pinecone, OpenAI GPT-4. Our projects have involved machine learning models and techniques such as CNN, UNet, Mask R-CNN, Faster R-CNN, ResNet-50, ResNet-101, SSD MobileNet V1, SSD, EAST Detector, Pose Estimation (ResNet-50), Conditional GANs, GANs, StyleGAN2, FaceSwap Models, LipGAN, Wav2Lip, ISR models, and TTS Transformers.

We have also integrated AI services and technologies including Text-to-Speech APIs (TTS), AI Text-to-Speech (TTS), AI Speech-to-Text (STT), AI Text-to-Image, AI-Generated Voice-Over, Stable Diffusion, Avatar Generation, Chatbot Development, and REST APIs.

For infrastructure, we have utilized Google Cloud VMs and Instances (Tesla P100, V100, A100 GPUs with 32GB / 64GB RAM), AWS EC2 (64GB RAM, 8× Tesla V100 GPUs), Raspberry Pi-4 (4GB), and Intel Movidius NCS2. Our deployments often involve real-time video processing and the integration of AI pipelines on cloud and edge devices.
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Write detailed technical AI project pipelines using real tools."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.3
    )
    project_plan = chat_completion.choices[0].message.content.strip()

    steps_dict = {}
    for line in project_plan.split('\n'):
        if line.strip() and ':' in line:
            step_part, tool_part = line.split(':', 1)
            step_title = step_part.strip().split('.', 1)[-1].strip()
            tools = tool_part.strip()
            steps_dict[step_title] = tools
    return project_plan, steps_dict


def get_cover_letter(title, description, skills):
    prompt = f"""Job Title: {title}
Client Description: {description}
Required Skills: {skills}

Write a short, professional Upwork cover letter. Mention experience with these tools.
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Write clean, professional Upwork cover letters."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.3
    )
    return chat_completion.choices[0].message.content.strip()
