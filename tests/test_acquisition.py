from labthings_picamera2 import StreamingPiCamera2
from labthings_fastapi.server import ThingServer
from labthings_fastapi.client import ThingClient
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from pytest import fixture

@fixture(scope="module")
def client():
    server = ThingServer()
    server.add_thing(StreamingPiCamera2(), "/camera/")
    with TestClient(server.app) as test_client:
        client = ThingClient.from_url("/camera/", client=test_client)
        yield client

def test_calibration(client):
    client.full_auto_calibrate()


def test_jpeg_and_array(client):
    blob = client.grab_jpeg()
    mjpeg_frame = Image.open(blob.open())
    assert mjpeg_frame
    blob = client.capture_jpeg(resolution="main")
    jpeg_capture = Image.open(blob.open())
    assert jpeg_capture
    arrlist = client.capture_array(stream_name="main")
    array_main = np.array(arrlist)
    assert mjpeg_frame.size == jpeg_capture.size
    assert array_main.shape[1::-1] == jpeg_capture.size

def test_raw_and_processed(client):
    #params = client.prepare_image_normalisation()
    raw = client.capture_raw()
    blob = client.raw_to_png(raw=raw) #, parameters=params)
    img = Image.open(blob.open())
    print(img.size)
