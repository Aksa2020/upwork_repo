import json

def load_processed_jobs(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def save_processed_jobs(job_list, filepath):
    with open(filepath, "w") as f:
        json.dump(job_list, f)


def filter_jobs(df):
    df = df[
        (df['Client Location'].str.strip().str.lower() != "pakistan") &
        (df['Client Location'].str.strip().str.lower() != "india") &
        (df['Payment Verified'].str.strip().str.lower() == "yes") &
        (df['Client Rating'] >= 4.0)
    ]
    return df
