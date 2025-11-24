# app.py
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# If needed, set path to tesseract executable:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r"E:/Tesseract-OCR/tesseract.exe"

def preprocess_image(image: Image.Image) -> Image.Image:
    # convert to grayscale, threshold
    img = np.array(image.convert('L'))
    # maybe resize if small
    # threshold: convert to binary
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # convert back to PIL image
    return Image.fromarray(img_bin)

def image_to_text(image: Image.Image, lang='eng') -> str:
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def text_to_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    # simple layout: start at top
    textobject = c.beginText(40, height - 40)
    text_lines = text.split('\n')
    for line in text_lines:
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("Handwritten Notes → PDF Converter (Free)")
    uploaded_file = st.file_uploader("Upload image of handwritten notes", type=["jpg","jpeg","png","tif","tiff"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Preprocessing image …")
        proc_img = preprocess_image(image)
        st.image(proc_img, caption="Pre-processed Image", use_column_width=True)
        st.write("Running OCR …")
        text = image_to_text(proc_img)
        st.text_area("Extracted Text", text, height=200)
        if st.button("Generate PDF"):
            pdf_buffer = text_to_pdf(text)
            st.download_button(label="Download PDF", data=pdf_buffer, file_name="notes.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
