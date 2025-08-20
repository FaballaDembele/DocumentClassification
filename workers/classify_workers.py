import os, torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from orkes.conductor import OrkesClient, OrkesWorker

MODEL_DIR = os.getenv("MODEL_DIR", "models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
labels_map = {0: "identity", 1: "invoice", 2: "mail", 3: "other"}

client = OrkesClient(
    key_id=os.getenv("ORKES_API_KEY"),
    key_secret=os.getenv("ORKES_API_SECRET"),
    server_url=os.getenv("ORKES_SERVER_URL", "https://play.orkes.io/api")
)

def classify_task(task: dict):
    text = task.get("inputData", {}).get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
    label_id = int(probs.argmax())
    return {
        "label": labels_map[label_id],
        "probs": {labels_map[i]: float(p) for i, p in enumerate(probs)},
        "chars": len(text)
    }

if __name__ == "__main__":
    worker = OrkesWorker(client, task_def_name="classification_worker", execute_function=classify_task)
    worker.start()
