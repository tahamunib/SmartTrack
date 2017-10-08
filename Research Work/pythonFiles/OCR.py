from pytesser3 import *
from PIL import Image

image_file = 'carNumber.jpg'
im = Image.open(image_file)
im.load()
text = image_to_string(im)
#text = image_file_to_string(image_file)
#text = image_file_to_string(image_file, graceful_errors=True)
print "=====output=======\n"
print text
