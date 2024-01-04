from picamera2 import Picamera2
from labthings_picamera2.recalibrate_utils import load_default_tuning
import json
import time

if __name__ == "__main__":
    print("Loading settings")
    with open("/var/openflexure/settings/camera/settings.json") as f:
        settings = json.load(f)
    #print(
    #    "Tuning algorithms: "
    #    ", ".join([list(a.keys())[0] for a in settings["tuning"]["algorithms"]])
    #)
    print("setting up picamera with no tuning")
    with Picamera2() as cam:
        print("opened")
        #sensor_modes = cam.sensor_modes
        #print(f"Enumerated {len(sensor_modes)} sensor modes")
        default_tuning = load_default_tuning(cam)
        time.sleep(1)
    print("closed")
    print("Setting up picamera")
    with Picamera2(camera_num=0, tuning=settings["tuning"]) as cam:
        print("Set up, generating coonfiguration")
        stream_config = cam.create_video_configuration(
            main={"size": (1640, 1232)},
            lores={"size": (320, 240), "format": "YUV420"},
            controls=settings["persistent_controls"],
        )
        print("Generated config, ")
        cam.configure(stream_config)
        cam.start()
        print("started")
        time.sleep(1)
    print("stopped")
