import streamlit as st
from PIL import Image
import math
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    canvas = None
    # default Letter size in points (width, height)
    letter = (612, 792)
    REPORTLAB_AVAILABLE = False
from io import BytesIO
import ollama

# Optional PDF -> images backends
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

def ocr_image(image: Image.Image):
    # Convert PIL to bytes for Ollama
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {"role": "user", "content": "Extract ALL text from this handwriting. Return only text.", 
             "images": [img_bytes]}
        ]
    )
    return response["message"]["content"]


def pdf_to_images(pdf_bytes: bytes):
    """Return list of PIL.Image for each page in the PDF.
    Tries `pdf2image` first, then `PyMuPDF` as fallback. Raises RuntimeError with install hints if neither is available.
    """
    if PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes)
            return images
        except Exception as e:
            # fall through to try pymupdf
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
        "No PDF rendering backend available. Install `pdf2image` and ensure poppler is installed, or install `PyMuPDF` (pip install pymupdf)."
    )


def truncate_for_context(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    # keep start and end for context
    head = text[: max_chars // 2]
    tail = text[- (max_chars // 2) :]
    return head + "\n\n...TRUNCATED...\n\n" + tail

def text_to_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    # create text object and set origin (safer API usage)
    textobject = c.beginText()
    textobject.setTextOrigin(40, height - 40)
    for line in text.split("\n"):
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("📝 Handwritten Notes → PDF Converter (Free, Local)")

    uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type

        images = []
        if file_type == "application/pdf" or uploaded_file.name.lower().endswith('.pdf'):
            try:
                with st.spinner("Converting PDF pages to images…"):
                    images = pdf_to_images(file_bytes)
                st.success(f"Converted PDF to {len(images)} image(s)")
                # show first page thumbnail
                if images:
                    st.image(images[0], caption="First page", use_column_width=True)
            except Exception as e:
                st.error(str(e))
                return
        else:
            try:
                image = Image.open(BytesIO(file_bytes))
                images = [image]
                st.image(images[0], caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to open uploaded image: {e}")
                return

        if st.button("Run OCR"):
            with st.spinner("Reading handwriting…"):
                texts = []
                for i, img in enumerate(images, start=1):
                    try:
                        page_text = ocr_image(img)
                    except Exception as e:
                        st.error(f"OCR failed on page {i}: {e}")
                        page_text = ""
                    texts.append(page_text)
                full_text = "\n\n".join(texts)
                st.session_state["extracted_text"] = full_text
            st.success("OCR Complete!")
            st.text_area("Extracted Text", full_text, height=300)

        # RAG QA section
        st.markdown("---")
        st.subheader("Ask questions about your notes (RAG)")
        question = st.text_input("Ask a question about the extracted notes:")
        use_full = st.checkbox("Use full notes as context (may be long)", value=False)
        if st.button("Ask"):
            if not question:
                st.warning("Please type a question first.")
            else:
                context = st.session_state.get("extracted_text", "")
                if not context:
                    st.warning("No extracted notes available. Run OCR first.")
                else:
                    with st.spinner("Querying model with RAG context…"):
                        try:
                            ctx = context if use_full else truncate_for_context(context, max_chars=5000)
                            system_msg = "You are an assistant answering questions using the user's extracted handwritten notes as context. Use only the provided notes to answer and cite nothing outside them. If the answer is not in the notes, say you don't know.\n\nNotes:\n" + ctx
                            response = ollama.chat(
                                model="gemma3:12b",
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": question},
                                ],
                            )
                            answer = response.get("message", {}).get("content")
                        except Exception as e:
                            st.error(f"RAG query failed: {e}")
                            answer = None
                    if answer:
                        st.write("**Answer:**")
                        st.write(answer)

        # PDF generation
        if st.button("Generate PDF"):
            try:
                if not REPORTLAB_AVAILABLE:
                    raise RuntimeError("reportlab is not installed. Install with: python -m pip install reportlab")
                text_for_pdf = st.session_state.get("extracted_text", "")
                if not text_for_pdf:
                    st.warning("No extracted text to generate PDF from. Run OCR first.")
                else:
                    pdf_buffer = text_to_pdf(text_for_pdf)
                    st.download_button(label="Download PDF", data=pdf_buffer.getvalue(), file_name="notes.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")

if __name__ == "__main__":
    main()
