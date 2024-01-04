import gc
import libcamera
import picamera2

def print_referrers(obj):
    """Print all objects that refer to an object, except for the top namespace"""
    print("Referrers: ", end="")
    for r in gc.get_referrers(obj):
        try:
            assert r["__name__"] == "__main__"
            print("__main__", end=", ")
            continue
        except:
            pass
        print(r, end=", ")
    print()

print("Making singleton (should see debug output below)")
m = libcamera.CameraManager.singleton()
print_referrers(m)

print("Making Picamera2")
cam = picamera2.Picamera2()
print_referrers(m)

print("cam.close()")
cam.close()
print_referrers(m)

del cam
del m
print("deleted cam, manager -- should start afresh from here.")

print("Making singleton (should see debug output below)")
m = libcamera.CameraManager.singleton()
print_referrers(m)

print("Making Picamera2")
cam = picamera2.Picamera2()
print_referrers(m)

print("cam.configure()")
cam.configure()
print_referrers(m)

print("cam.close()")
cam.close()
print_referrers(m)

del cam
del m
print("deleted cam, manager -- should start afresh from here.")

print("Making singleton (should see debug output below)")
m = libcamera.CameraManager.singleton()
print_referrers(m)
