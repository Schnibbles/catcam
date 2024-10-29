from picamera2 import Picamera2

picam = Picamera2()
cur = picam.capture_buffer()
lsize = (320, 240)
w, h = 320, 240
cur = cur[:w * h].reshape(h, w)
picam.start()
print(cur)
