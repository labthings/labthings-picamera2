import logging

from fastapi.testclient import TestClient
import numpy as np

from labthings_fastapi.server import ThingServer
from labthings_fastapi.client import ThingClient
from labthings_picamera2.thing import StreamingPiCamera2

logging.basicConfig(level=logging.DEBUG)


def test_sensor_mode():
    cam = StreamingPiCamera2()
    server = ThingServer()
    server.add_thing(cam, "/camera/")

    with TestClient(server.app) as test_client:
        client = ThingClient.from_url("/camera/", client=test_client)
        for size in [(3280, 2464), (1640, 1232)]:
            client.sensor_mode = {"output_size": size, "bit_depth": 10}
            arr = np.array(client.capture_array(stream_name="raw"))
            assert arr.shape[0] == size[1]


if __name__ == "__main__":
    test_sensor_mode()
