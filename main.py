from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
from PIL import Image
import io
import gc

app = FastAPI(title="EasyOCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chỉ giữ 1 reader tại một thời điểm để tiết kiệm RAM
current_reader = None
current_lang = None

def get_reader(lang: str):
    global current_reader, current_lang
    if current_lang != lang:
        # Giải phóng reader cũ trước
        current_reader = None
        gc.collect()
        lang_list = lang.split("+")
        current_reader = easyocr.Reader(lang_list, gpu=False, verbose=False)
        current_lang = lang
    return current_reader

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
        # Resize ảnh nếu quá lớn để tiết kiệm RAM
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        max_size = 1920
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)

        img_array = np.array(image)

        reader = get_reader(lang)
        results = reader.readtext(img_array)

        lines = []
        total_conf = 0
        for (_, text, conf) in results:
            lines.append({"text": text, "confidence": round(conf * 100, 1)})
            total_conf += conf

        full_text = "\n".join([r["text"] for r in lines])
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
