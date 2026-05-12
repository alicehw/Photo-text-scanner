from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
from PIL import Image
import io

app = FastAPI(title="EasyOCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache reader theo ngôn ngữ
readers = {}

def get_reader(lang: str):
    if lang not in readers:
        lang_list = lang.split("+")
        readers[lang] = easyocr.Reader(lang_list, gpu=False)
    return readers[lang]

@app.get("/")
def root():
    return {"status": "ok", "message": "EasyOCR API đang chạy"}

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form(default="vi")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        reader = get_reader(lang)
        results = reader.readtext(img_array)

        lines = []
        total_conf = 0
        for (_, text, conf) in results:
            lines.append({"text": text, "confidence": round(conf * 100, 1)})
            total_conf += conf

        full_text = " ".join([r["text"] for r in lines])
        avg_conf = round((total_conf / len(results)) * 100, 1) if results else 0

        return {
            "success": True,
            "text": full_text,
            "lines": lines,
            "confidence": avg_conf,
            "lang": lang
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
