from labthings_picamera2.thing import StreamingPiCamera2
from labthings_fastapi.thing_settings import ThingSettings
import json
import time

class StubPortal:
    def start_task_soon(*args, **kwargs):
        pass

settings = ThingSettings("/var/openflexure/settings/camera/settings.json")
portal = StubPortal()
cam = StreamingPiCamera2()
cam._labthings_thing_settings = settings
cam._labthings_blocking_portal = portal
#print(
#    "Tuning algorithms: "
#    ", ".join([list(a.keys())[0] for a in cam.thing_settings["tuning"]["algorithms"]])
#)
print(f"Tuning file version {cam.thing_settings['tuning']['version']}")
print("checking modes/tuning")
cam.populate_default_tuning()
print("initialising tuning")
cam.initialise_tuning()
print("initialising picamera")
cam.initialise_picamera()
print("setting persistent controls")
cam.settings_to_persistent_controls()
print("setting properties from settings")
cam.settings_to_properties()
print("starting stream")
cam.start_streaming()
print("streaming")
time.sleep(1)
print("stopping stream")
cam.stop_streaming()
print("closing camera")
with cam.picamera() as cam:
    cam.close()
del cam._picamera
print("closed")

