import json
import os

def load_processed_jobs(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = f.read().strip()
            return json.loads(data) if data else []
    except json.JSONDecodeError:
        return []

def save_processed_jobs(job_list, filepath):
    with open(filepath, "w") as f:
        json.dump(job_list, f)

def filter_jobs(df):
    return df[
        (~df['Client Location'].str.strip().str.lower().isin(["pakistan", "india"])) &
        (df['Payment Verified'].str.strip().str.lower() == "yes") &
        (df['Client Rating'] >= 4.0)
    ]
