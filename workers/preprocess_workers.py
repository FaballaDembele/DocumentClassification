import os, re
from orkes.conductor import OrkesClient, OrkesWorker

client = OrkesClient(
    key_id=os.getenv("ORKES_API_KEY"),
    key_secret=os.getenv("ORKES_API_SECRET"),
    server_url=os.getenv("ORKES_SERVER_URL", "https://play.orkes.io/api")
)

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[\t\r]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

def preprocess_task(task: dict):
    text = task.get("inputData", {}).get("text", "")
    return {"text": clean_text(text)}

if __name__ == "__main__":
    worker = OrkesWorker(client, task_def_name="preprocess_worker", execute_function=preprocess_task)
    worker.start()
