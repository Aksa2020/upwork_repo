import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["groq"]["api_key"])

def get_project_plan(title, description, skills):
    prompt = f"""
You are a senior AI Consultant.
Your task is to design a detailed technical project pipeline for this job.

Job Title: {title}
Client Description: {description}
Required Skills: {skills}

Respond in this clean format ONLY:
1. Step: Tools or Technologies
2. Step: Tools or Technologies
...

NO headings, no markdown, no bullet points. Only numbered steps with colon.
Be realistic with AI/ML tools. 
Use tools like YOLOv8, PyTorch, FastAPI, Docker, Azure, etc., where appropriate.
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
