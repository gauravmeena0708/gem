from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import ImageDraw, ImageFont
img = Image.new('RGB', (200, 60), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10, 10), "Hello World!", fill=(0, 0, 0))
img.show()
text = pytesseract.image_to_string(img)

print("Extracted Text:")
print(text)
