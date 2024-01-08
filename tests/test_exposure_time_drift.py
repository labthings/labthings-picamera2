import logging
import time

from fastapi.testclient import TestClient

from labthings_fastapi.thing_server import ThingServer
from labthings_fastapi.client import ThingClient
from labthings_picamera2.thing import StreamingPiCamera2

logging.basicConfig(level=logging.DEBUG)


def test_exposure_time_drift():
    cam = StreamingPiCamera2()
    server = ThingServer()
    server.add_thing(cam, "/camera/")

    with TestClient(server.app) as test_client:
        client = ThingClient.from_url("/camera/", client=test_client)
        client.exposure_time = 50000
        time.sleep(0.1)
        initial_et = client.exposure_time
        print(f"Before capture, et is {client.exposure_time}")
        for i in range(10):
            client.capture_jpeg(resolution="full")
            print(f"After capture, et is {client.exposure_time}")
        final_et = client.exposure_time
        assert initial_et == final_et


if __name__ == "__main__":
    test_exposure_time_drift()
