from groq import Groq

import streamlit as st

# Read securely from Streamlit Secrets
api_key = st.secrets["groq"]["api_key"]
client = Groq(api_key=api_key)


def get_project_plan(title, description, skills):
    prompt = f"""Job Title: {title}
Client Description: {description}
Required Skills: {skills}

Give me a step-by-step technical plan with real tools like YOLOv8, PyTorch, Docker, etc.
Format like this: '1. Task: Tools'

Heading: Proposed Project Flow with Technologies:
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Write detailed AI project plans mentioning tools clearly."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.3
    )
    project_plan = chat_completion.choices[0].message.content.strip()

    steps_dict = {}
    capture = False
    for line in project_plan.split('\n'):
        if 'Proposed Project Flow' in line:
            capture = True
            continue
        if capture:
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and '.' in line:
                split_line = line.split(':', 1)
                if len(split_line) == 2:
                    step_title = split_line[0].split('.', 1)[1].strip()
                    tools = split_line[1].strip()
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
