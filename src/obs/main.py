from mss import mss

with mss() as sct:
    mon = sct.monitors[2]
    screenShot = sct.grab(mon)
    print(screenShot.width, screenShot.height)
