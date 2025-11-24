from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import typing
import traceback

app = FastAPI(title="Handwritten OCR + RAG API")

# Allow all origins for UI development; lock this down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (mounted at /static) and index at /
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", response_class=FileResponse)
async def root_index():
    return FileResponse("frontend/index.html")

# Optional dependencies and fallbacks
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    canvas = None
    letter = (612, 792)
    REPORTLAB_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    convert_from_bytes = None
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    fitz = None
    PYMUPDF_AVAILABLE = False


# Utility functions copied/adapted from test2.py
def truncate_for_context(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[- (max_chars // 2) :]
    return head + "\n\n...TRUNCATED...\n\n" + tail


def text_to_pdf_bytes(text: str) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not available; install with: python -m pip install reportlab")
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    textobject = c.beginText()
    textobject.setTextOrigin(40, height - 40)
    for line in text.split("\n"):
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def pdf_to_images(pdf_bytes: bytes) -> typing.List[Image.Image]:
    if PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes)
            return images
        except Exception:
            pass

    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            for page in doc:
                pix = page.get_pixmap()
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                images.append(img)
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to render PDF pages: {e}")

    raise RuntimeError(
        "No PDF rendering backend available. Install `pdf2image` + poppler or `PyMuPDF` (pip install pymupdf)."
    )


def ocr_image_bytes(img_bytes: bytes) -> str:
    # Accept image bytes (PIL-ready), convert to PNG bytes for Ollama input
    img = Image.open(BytesIO(img_bytes))
    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama client not available. Install and configure Ollama or mock this in development.")

    # Call Ollama chat API similarly to test2.py
    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {"role": "user", "content": "Extract ALL text from this handwriting. Return only text.", "images": [png_bytes]}
        ],
    )
    # Best-effort extraction of content
    return response.get("message", {}).get("content")


# Request/response models
class RAGRequest(BaseModel):
    text: str
    question: str
    use_full: bool = False


@app.get("/health")
async def health():
    return {"status": "ok", "ollama": OLLAMA_AVAILABLE, "reportlab": REPORTLAB_AVAILABLE}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        data = await file.read()
        is_pdf = file.content_type == "application/pdf" or file.filename.lower().endswith('.pdf')
        images = []
        if is_pdf:
            try:
                images = pdf_to_images(data)
            except Exception as e:
                print(e)
        else:
            # single image
            images = [Image.open(BytesIO(data))]

        texts = []
        for i, img in enumerate(images, start=1):
            # convert PIL image to bytes and call ocr_image_bytes
            buf = BytesIO()
            img.save(buf, format="PNG")
            page_bytes = buf.getvalue()
            try:
                page_text = ocr_image_bytes(page_bytes)
            except Exception as e:
                # include page-specific error but continue
                page_text = f""  # leave blank on fail
            texts.append(page_text)

        full_text = "\n\n".join(texts)
        return JSONResponse({"text": full_text, "pages": len(images)})
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)


@app.post("/rag")
async def rag_endpoint(req: RAGRequest):
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=500, detail="ollama client not available on server")
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required for RAG")
    ctx = req.text if req.use_full else truncate_for_context(req.text, max_chars=5000)
    system_msg = (
        "You are an assistant answering questions using the user's extracted handwritten notes as context. "
        "Use only the provided notes to answer and cite nothing outside them. If the answer is not in the notes, say you don't know.\n\nNotes:\n"
        + ctx
    )
    try:
        response = ollama.chat(
            model="gemma3:12b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": req.question},
            ],
        )
        answer = response.get("message", {}).get("content")
        return JSONResponse({"answer": answer})
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)


class PDFRequest(BaseModel):
    text: str


@app.post("/generate_pdf")
async def generate_pdf(req: PDFRequest):
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=500, detail="reportlab not installed. Install with: python -m pip install reportlab")
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required to generate PDF")

    try:
        pdf_bytes = text_to_pdf_bytes(req.text)
        return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=notes.pdf"})
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)


# If run directly, start uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
