from picamera2 import Picamera2
import numpy
from PIL import Image

lsize = (320, 240)
picam = Picamera2()
picam.create_still_configuration(lores={"size": lsize})
with Image.open(picam.start() as image:
    (left, upper, right, lower) = (20, 20, 100, 100)
    crop = image.crop((left, upper, right, lower))
Image.fromarray(crop).save("image.jpg","jpg")

#lsize = (320, 240))
#picam.start()
#cur = picam.capture_buffer()
#w, h = 320, 240
#cur = cur[:w * h].reshape(h, w)

#print(cur)
