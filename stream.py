from picamera2 import Picamera2


lsize = (320, 240)
picam = Picamera2()
picam.create_still_configuration(lores={"size": lsize})
picam.start()
cur = picam.capture_buffer()
w, h = 320, 240
cur = cur[:w * h].reshape(h, w)

print(cur)
