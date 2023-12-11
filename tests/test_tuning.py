from __future__ import annotations
from fastapi.testclient import TestClient
from labthings_fastapi.thing_server import ThingServer
from labthings_fastapi.thing_settings import ThingSettings
from labthings_picamera2.thing import StreamingPiCamera2
import anyio
import pytest

thing_server = ThingServer()
cam = StreamingPiCamera2()
thing_server.add_thing(cam, "/camera")

def test_tuning_updates(capsys: pytest.CaptureFixture):
    print("Making StreamingPiCamera2")
    cam = StreamingPiCamera2()
    print("Making blocking portal")
    with anyio.from_thread.start_blocking_portal() as portal:
        cam._labthings_blocking_portal = portal
        cam._labthings_thing_settings = ThingSettings("temp_settings.json")
        print("Entering camera object")
        with cam:
            print("Entered")
            captured = capsys.readouterr()
            print(f"Setting up the camera printed {len(captured.err.splitlines())} lines")
            cam.initialise_picamera()
            captured = capsys.readouterr()

            print(f"Re-initialising the camera printed {len(captured.err.splitlines())} lines")
    assert False

# def test_tuning_updates():
#     # TestClient is a convenient way to run the Thing
#     with TestClient(thing_server.app):
#         oldversion = cam.tuning["version"]
#         cam.tuning["version"] = 999  # deliberately cause an error
#         with pytest.raises(IndexError):
#             # If we reload the tuning, this will fail
#             cam.initialise_picamera()
#         # Reset it before the test closes - or there will be errors
#         # in __exit__
#         #cam.tuning["version"] = oldversion
#         #cam.initialise_picamera()  # This hangs - can't recover from libcamera error.

if __name__ == "__main__":
    class FakeOutput:
        err = ""
        out = ""
    class FakeCapsys:
        def readouterr(self):
            return FakeOutput()
    test_tuning_updates(FakeCapsys())