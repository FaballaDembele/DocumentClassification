import os, io, base64
from typing import List
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from orkes.conductor import OrkesClient, OrkesWorker

OCR_LANG = os.getenv("OCR_LANG", "eng+fra")

client = OrkesClient(
    key_id=os.getenv("ORKES_API_KEY"),
    key_secret=os.getenv("ORKES_API_SECRET"),
    server_url=os.getenv("ORKES_SERVER_URL", "https://play.orkes.io/api")
)

def _images_from_base64(b64: str, mime: str) -> List[Image.Image]:
    raw = base64.b64decode(b64)
    if mime.lower() == "application/pdf":
        pages = convert_from_bytes(raw, dpi=200, first_page=1, last_page=3)
        return pages
    else:
        return [Image.open(io.BytesIO(raw))]

def _ocr_images(imgs: List[Image.Image]) -> str:
    texts = []
    for im in imgs:
        g = im.convert("L")
        texts.append(pytesseract.image_to_string(g, lang=OCR_LANG) or "")
    return "\n".join(texts)

def ocr_task(task: dict):
    inp = task.get("inputData", {})
    b64 = inp["file_base64"]
    mime = inp.get("mime_type", "application/octet-stream")
    imgs = _images_from_base64(b64, mime)
    text = _ocr_images(imgs)
    return {"text": text, "chars": len(text)}

if __name__ == "__main__":
    worker = OrkesWorker(client, task_def_name="ocr_worker", execute_function=ocr_task)
    worker.start()
