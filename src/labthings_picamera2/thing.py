from __future__ import annotations
import logging
import time

from pydantic import BaseModel, BeforeValidator

from labthings_fastapi.descriptors.property import PropertyDescriptor
from labthings_fastapi.thing import Thing
from labthings_fastapi.decorators import thing_action, thing_property
from labthings_fastapi.file_manager import FileManagerDep
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
import numpy as np
from . import recalibrate_utils


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
    def __init__(self, camera_num: int = 0):
        self.camera_num = camera_num
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

    def update_persistent_controls(self, discard_frames: int=1):
        """Update the persistent controls dict from the camera"""
        with self.picamera() as cam:
            for i in range(discard_frames):
                # Discard frames, so we know our data is fresh
                cam.capture_metadata()
            for k, v in cam.capture_metadata().items():
                if k in self.persistent_controls:
                    print(f"Updating persistent control {k} to {v}")
                    self.persistent_controls[k] = v
        self.thing_settings.update(self.persistent_controls)

    def settings_to_persistent_controls(self):
        """Update the persistent controls dict from the settings dict
        
        NB this must be called **after** self.thing_settings is initialised,
        i.e. during or after `__enter__`.
        """
        try:
            pc = self.thing_settings["persistent_controls"]
        except KeyError:
            return  # If there are no saved settings, use defaults
        for k in self.persistent_controls:
            try:
                self.persistent_controls[k] = pc[k]
            except KeyError:
                pass  # If controls are missing, leave at default

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

    tuning = PropertyDescriptor(Optional[dict], None, readonly=True)

    def initialise_tuning(self):
        """Read the tuning from the settings, or load default tuning
        
        NB this relies on `self.thing_settings` and `self.default_tuning`
        so will fail if it's run before those are populated in `__enter__`.
        """
        if "tuning" in self.thing_settings:
            # TODO: should this be a separate file?
            self.tuning = self.thing_settings["tuning"].dict
        else:
            logging.info("Did not find tuning in settings, reading from camera...")
            self.tuning = self.default_tuning
    
    def initialise_picamera(self):
        """Acquire the picamera device and store it as `self._picamera`"""
        if hasattr(self, "_picamera_lock"):
            # Don't close the camera if it's in use
            self._picamera_lock.acquire()
        if hasattr(self, "_picamera") and self._picamera:
            logging.info("Camera object already exists, closing for reinitialisation")
            self._picamera.close()
        self._picamera = picamera2.Picamera2(
            camera_num=self.camera_num,
            tuning=self.tuning
        )
        self._picamera_lock = RLock()

    def __enter__(self):
        self.populate_sensor_modes_and_default_tuning()
        self.initialise_tuning()
        self.initialise_picamera()
        self.settings_to_persistent_controls()
        self.start_streaming()
        return self
    
    @contextmanager
    def picamera(self, pause_stream=False) -> Iterator[Picamera2]:
        """Return the underlying `Picamera2` instance, optionally pausing the stream.
        
        If pause_stream is True (default is False), we will stop the MJPEG stream
        before yielding control of the camera, and restart afterwards. If you make
        changes to the camera settings, these may be ignored when the stream is
        restarted: you may nened to call `update_persistent_controls()` to ensure
        your changes persist after the stream restarts.
        """
        with self._picamera_lock:
            if pause_stream:
                self.update_persistent_controls()
                self.stop_streaming(stop_web_stream=False)
            try:
                yield self._picamera
            finally:
                if pause_stream:
                    self.start_streaming()

    def populate_sensor_modes_and_default_tuning(self):
        """Sensor modes are enumerated and stored, once, on start-up (`__enter__`).
        
        This opens and closes the camera - must be run before the camera is 
        initialised.
        """
        logging.info("Starting & reconfiguring camera to populate sensor_modes.")
        with Picamera2(camera_num=self.camera_num) as cam:
            self.sensor_modes = cam.sensor_modes
            self.default_tuning = recalibrate_utils.load_default_tuning(cam)
        logging.info("Done reading sensor modes & default tuning.")

    def __exit__(self, exc_type, exc_value, traceback):
        # Allow key controls to persist across restarts
        self.update_persistent_controls()
        self.thing_settings["persistent_controls"] = self.persistent_controls
        self.thing_settings["tuning"] = self.tuning
        self.thing_settings.write_to_file() 
        # Shut down the camera
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
    def snap_image(self, file_manager: FileManagerDep) -> str:
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

    @thing_property
    def capture_metadata(self) -> dict:
        """Return the metadata from the camera"""
        with self.picamera() as cam:
            return cam.capture_metadata()
    
    @thing_action
    def auto_expose_from_minimum(
        self, 
        target_white_level: int=700,
        percentile: float=99.9,
    ):
        """Adjust exposure to hit the target white level
        
        Starting from the minimum exposure, we gradually increase exposure until
        we hit the specified white level. We use a percentile rather than the
        maximum, in order to be robust to a small number of noisy/bright pixels.
        """
        with self.picamera(pause_stream=True) as cam:
            recalibrate_utils.adjust_shutter_and_gain_from_raw(
                cam,
                target_white_level=target_white_level,
                percentile=percentile,
            )
            self.update_persistent_controls()

    @thing_action
    def calibrate_white_balance(self):
        """Correct the white balance of the image
        
        This method requires a neutral image, such that the 99th centile of
        each colour channel should correspond to white. We calculate the 
        centiles and use this to set the colour gains.
        """
        with self.picamera(pause_stream=True) as cam:
            recalibrate_utils.adjust_white_balance_from_raw(cam, percentile=99)
            self.update_persistent_controls()

    @thing_action
    def calibrate_lens_shading(self):
        """Take an image and use it for flat-field correction.

        This method requires an empty (i.e. bright) field of view. It will take
        a raw image and effectively divide every subsequent image by the current
        one. This uses the camera's "tuning" file to correct the preview and
        the processed images. It should not affect raw images.
        """
        with self.picamera(pause_stream=True) as cam:
            L, Cr, Cb = recalibrate_utils.lst_from_camera(cam)
            gain_r, gain_b = self.persistent_controls["ColourGains"]
            print(f"Colour gains currently {gain_r}, {gain_b}")
            gain_r *= np.min(Cr)
            Cr /= np.min(Cr)
            gain_b *= np.min(Cb)
            Cb /= np.min(Cb)
            self.persistent_controls["ColourGains"] = (gain_r, gain_b)
            print(f"Colour gains now {gain_r}, {gain_b}")
            recalibrate_utils.set_static_lst(self.tuning, L, Cr, Cb)
            self.initialise_picamera()

    @thing_action
    def flat_lens_shading(self):
        """Disable flat-field correction
        
        This method will set a completely flat lens shading table. It is not the
        same as the default behaviour, which is to use an adaptive lens shading
        table.
        """
        with self.picamera(pause_stream=True) as cam:
            f = np.ones((12, 16))
            recalibrate_utils.set_static_lst(self.tuning, f, f, f)
            self.initialise_picamera()

    @thing_action
    def flat_lens_shading_chrominance(self):
        """Disable flat-field correction
        
        This method will set a completely flat lens shading table. It is not the
        same as the default behaviour, which is to use an adaptive lens shading
        table.
        """
        with self.picamera(pause_stream=True) as cam:
            alsc = Picamera2.find_tuning_algo(self.tuning, "rpi.alsc")
            luminance = alsc["luminance_lut"]
            flat = np.ones((12, 16))
            recalibrate_utils.set_static_lst(self.tuning, luminance, flat, flat)
            self.initialise_picamera()
    
    @thing_action
    def reset_lens_shading(self):
        """Revert to default lens shading settings
        
        This method will restore the default "adaptive" lens shading method used
        by the Raspberry Pi camera.
        """
        with self.picamera(pause_stream=True) as cam:
            recalibrate_utils.copy_alsc_section(self.default_tuning, self.tuning)
            self.initialise_picamera()

