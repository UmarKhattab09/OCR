from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image,ImageOps

processor = DonutProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/layoutlmv3-base")

image = Image.open("E:/ocr/handwrittennotes.jpg").convert("RGB")
image = ImageOps.autocontrast(image)     # improve clarity
image = image.resize((384, 384))         # resize to stable shape

# preprocess
inputs = processor(images=image, return_tensors="pt")

# generate text
outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(text)
