from __future__ import annotations
import logging
import time

from pydantic import BaseModel, BeforeValidator

from labthings_fastapi.descriptors.property import PropertyDescriptor
from labthings_fastapi.thing import Thing
from labthings_fastapi.decorators import thing_action, thing_property
from labthings_fastapi.file_manager import FileManager
from typing import Annotated, Any, Iterator, Optional, Tuple
from contextlib import contextmanager
from anyio.from_thread import BlockingPortal
from threading import RLock
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, Quality
from picamera2.outputs import Output
from picamera2 import controls
from labthings_fastapi.outputs.mjpeg_stream import MJPEGStreamDescriptor, MJPEGStream
from labthings_fastapi.utilities import get_blocking_portal
from .recalibrate_utils import adjust_shutter_and_gain_from_raw, adjust_white_balance_from_raw



class PicameraControl(PropertyDescriptor):
    def __init__(
            self,
            control_name: str,
            model: type=float,
            description: Optional[str]=None
        ):
        """A property descriptor controlling a picamera control"""
        PropertyDescriptor.__init__(
            self, model, observable=False, description=description
        )
        self.control_name = control_name
        self._getter

    def _getter(self, obj: StreamingPiCamera2):
        print(f"getting {self.control_name} from {obj}")
        with obj.picamera() as cam:
            ret = cam.capture_metadata()[self.control_name]
            print(f"Trying to return camera control {self.control_name} as `{ret}`")
            return ret
        
    def _setter(self, obj: StreamingPiCamera2, value: Any):
        with obj.picamera() as cam:
            cam.set_controls({self.control_name: value})
    

class PicameraStreamOutput(Output):
    """An Output class that sends frames to a stream"""
    def __init__(self, stream: MJPEGStream, portal: BlockingPortal):
        """Create an output that puts frames in an MJPEGStream
        
        We need to pass the stream object, and also the blocking portal, because
        new frame notifications happen in the anyio event loop and frames are
        sent from a thread. The blocking portal enables thread-to-async 
        communication.
        """
        Output.__init__(self)
        self.stream = stream
        self.portal = portal

    def outputframe(self, frame, _keyframe=True, _timestamp=None):
        """Add a frame to the stream's ringbuffer"""
        self.stream.add_frame(frame, self.portal)


class SensorMode(BaseModel):
    unpacked: str
    bit_depth: int
    size: tuple[int, int]
    fps: float
    crop_limits: tuple[int, int, int, int]
    exposure_limits: tuple[Optional[int], Optional[int], Optional[int]]
    format: Annotated[str, BeforeValidator(repr)]


class StreamingPiCamera2(Thing):
    """A Thing that represents an OpenCV camera"""
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.camera_configs: dict[str, dict] = {}
        # NB persistent controls will be updated with settings, in __enter__.
        self.persistent_controls = {
            "AeEnable": False,
            "AnalogueGain": 1.0,
            "AwbEnable": False,
            "Brightness": 0,
            "ColourGains": (1,1),
            "Contrast": 1,
            "ExposureTime": 0,
            "Saturation": 1,
            "Sharpness": 1,
        }

    def update_persistent_controls(self):
        with self.picamera() as cam:
            for k, v in cam.capture_metadata().items():
                if k in self.persistent_controls:
                    print(f"Updating persistent control {k} to {v}")
                    self.persistent_controls[k] = v
        self.thing_settings.update(self.persistent_controls)

    stream_resolution = PropertyDescriptor(
        tuple[int, int],
        initial_value=(1640, 1232),
        description="Resolution to use for the MJPEG stream",
    )
    image_resolution = PropertyDescriptor(
        tuple[int, int],
        initial_value=(3280, 2464),
        description="Resolution to use for still images (by default)",
    )
    mjpeg_bitrate = PropertyDescriptor(
        int, 
        initial_value=0, 
        description="Bitrate for MJPEG stream (best left at 0)"
    )
    stream_active = PropertyDescriptor(
        bool, 
        initial_value=False,
        description="Whether the MJPEG stream is active",
        observable=True,
        readonly=True,
    )
    mjpeg_stream = MJPEGStreamDescriptor()
    analogue_gain = PicameraControl(
        "AnalogueGain",
        float
    )
    colour_gains = PicameraControl(
        "ColourGains",
        tuple[float, float]
    )
    colour_correction_matrix = PicameraControl(
        "ColourCorrectionMatrix",
        tuple[float, float, float, float, float, float, float, float, float]
    )
    exposure_time = PicameraControl(
        "ExposureTime", 
        int, 
        description="The exposure time in microseconds"
    )
    exposure_time = PicameraControl(
        "ExposureTime", 
        int, 
        description="The exposure time in microseconds"
    )
    sensor_modes = PropertyDescriptor(list[SensorMode], readonly=True)
    
    def __enter__(self):
        self._picamera = picamera2.Picamera2(camera_num=self.device_index)
        self._picamera_lock = RLock()
        self.populate_sensor_modes()
        for k in self.persistent_controls:
            if k in self.thing_settings:
                self.persistent_controls[k] = self.thing_settings[k]
        self.start_streaming()
        return self
    
    @contextmanager
    def picamera(self, pause_stream=False) -> Iterator[Picamera2]:
        with self._picamera_lock:
            if pause_stream:
                self.update_persistent_controls()
                self.stop_streaming(stop_web_stream=False)
            try:
                yield self._picamera
            finally:
                if pause_stream:
                    self.start_streaming()

    def populate_sensor_modes(self):
        with self.picamera() as cam:
            self.sensor_modes = cam.sensor_modes

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_streaming()
        with self.picamera() as cam:
            cam.close()
        del self._picamera

    def start_streaming(self) -> None:
        """
        Start the MJPEG stream
        
        Sets the camera resolution to the video/stream resolution, and starts recording
        if the stream should be active.
        """
        with self.picamera() as picam:
            #TODO: Filip: can we use the lores output to keep preview stream going
            #while recording? According to picamera2 docs 4.2.1.6 this should work
            try:
                if picam.started:
                    picam.stop()
                if picam.encoder is not None and picam.encoder.running:
                    picam.encoder.stop()
                stream_config = picam.create_video_configuration(
                    main={"size": self.stream_resolution},
                    controls=self.persistent_controls,
                    #colour_space=ColorSpace.Rec709(),
                )
                picam.configure(stream_config)
                logging.info("Starting picamera MJPEG stream...")
                picam.start_recording(
                    MJPEGEncoder(
                        self.mjpeg_bitrate if self.mjpeg_bitrate > 0 else None,
                    ),
                    PicameraStreamOutput(
                        self.mjpeg_stream, 
                        get_blocking_portal(self),
                    ),
                    Quality.HIGH #TODO: use provided quality
                )
            except Exception as e:
                logging.info("Error while starting preview:")
                logging.exception(e)
            else:
                self.stream_active = True
                logging.debug(
                    "Started MJPEG stream at %s on port %s", self.stream_resolution, 1
                )

    def stop_streaming(self, stop_web_stream=True) -> None:
        """
        Stop the MJPEG stream
        """
        with self.picamera() as picam:
            try:
                picam.stop_recording()
            except Exception as e:
                logging.info("Stopping recording failed")
                logging.exception(e)
            else:
                self.stream_active = False
                if stop_web_stream:
                    self.mjpeg_stream.stop()
                logging.info(
                    f"Stopped MJPEG stream. Switching to {self.image_resolution}."
                )

            # Increase the resolution for taking an image
            time.sleep(
                0.2
            )  # Sprinkled a sleep to prevent camera getting confused by rapid commands

    @thing_action
    def snap_image(self, file_manager: FileManager) -> str:
        """Acquire one image from the camera.

        This action cannot run if the camera is in use by a background thread, for
        example if a preview stream is running.
        """
        raise NotImplementedError
    
    @thing_property
    def exposure(self) -> float:
        """An alias for `exposure_time` to fit the micromanager API"""
        return self.exposure_time
    @exposure.setter  # type: ignore
    def exposure(self, value):
        self.exposure_time = value
    
    @thing_action
    def auto_expose_from_minimum(self):
        with self.picamera(pause_stream=True) as cam:
            adjust_shutter_and_gain_from_raw(cam)
            self.update_persistent_controls()

    @thing_action
    def auto_white_balance(self):
        with self.picamera(pause_stream=True) as cam:
            adjust_white_balance_from_raw(cam, percentile=99)
            self.update_persistent_controls()
