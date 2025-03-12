from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import io
import json
import logging
import os
import tempfile
import time
from tempfile import TemporaryDirectory
import uuid

from pydantic import BaseModel, BeforeValidator, RootModel

from labthings_fastapi.descriptors.property import PropertyDescriptor
from labthings_fastapi.thing import Thing
from labthings_fastapi.decorators import thing_action, thing_property
from labthings_fastapi.outputs.mjpeg_stream import MJPEGStreamDescriptor, MJPEGStream
from labthings_fastapi.utilities import get_blocking_portal
from labthings_fastapi.types.numpy import NDArray
from labthings_fastapi.dependencies.metadata import GetThingStates
from labthings_fastapi.dependencies.blocking_portal import BlockingPortal
from labthings_fastapi.outputs.blob import Blob, BlobBytes
from typing import Annotated, Any, Iterator, Literal, Mapping, Optional, Self
from contextlib import contextmanager
import piexif
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from PIL import Image
from threading import RLock
import picamera2
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import Output
import numpy as np
from . import recalibrate_utils


class JPEGBlob(Blob):
    media_type: str = "image/jpeg"


class PNGBlob(Blob):
    media_type: str = "image/png"


class RawBlob(Blob):
    media_type: str = "image/raw"


class RawImageModel(BaseModel):
    image_data: RawBlob
    thing_states: Optional[Mapping[str, Mapping]]
    metadata: Optional[Mapping[str, Mapping]]
    processing_inputs: Optional[ImageProcessingInputs] = None
    size: tuple[int, int]
    stride: int
    format: str

class PicameraControl(PropertyDescriptor):
    def __init__(
        self, control_name: str, model: type = float, description: Optional[str] = None
    ):
        """A property descriptor controlling a picamera control"""
        PropertyDescriptor.__init__(
            self, model, observable=False, description=description
        )
        self.control_name = control_name

    def _getter(self, obj: StreamingPiCamera2):
        with obj.picamera() as cam:
            ret = cam.capture_metadata()[self.control_name]
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

    def outputframe(self, frame, _keyframe=True, _timestamp=None, _packet=None, _audio=False):
        """Add a frame to the stream's ringbuffer"""
        self.stream.add_frame(frame, self.portal)


class ArrayModel(RootModel):
    root: NDArray


class SensorMode(BaseModel):
    unpacked: str
    bit_depth: int
    size: tuple[int, int]
    fps: float
    crop_limits: tuple[int, int, int, int]
    exposure_limits: tuple[Optional[int], Optional[int], Optional[int]]
    format: Annotated[str, BeforeValidator(repr)]


class SensorModeSelector(BaseModel):
    output_size: tuple[int, int]
    bit_depth: int


class LensShading(BaseModel):
    luminance: list[list[float]]
    Cr: list[list[float]]
    Cb: list[list[float]]


class ImageProcessingInputs(BaseModel):
    lens_shading: LensShading
    colour_gains: tuple[float, float]
    white_norm_lores: NDArray
    raw_size: tuple[int, int]
    colour_correction_matrix: tuple[float, float, float, float, float, float, float, float, float]
    gamma: NDArray


@dataclass
class ImageProcessingCache:
    white_norm: np.ndarray
    gamma: interp1d
    ccm: np.ndarray


class BlobNumpyDict(BlobBytes): 
    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self._arrays = arrays
        self._bytesio: Optional[io.BytesIO] = None
        self.media_type = "application/npz"

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        return self._arrays

    @property
    def _bytes(self) -> bytes: #noqa mypy: override
        """Generate binary content on-the-fly from numpy data"""
        if not self._bytesio:
            out = io.BytesIO()
            np.savez(out, **self.arrays)
            self._bytes_cache = out.getvalue()
        return self._bytes_cache


class NumpyBlob(Blob):
    media_type: str = "application/npz"

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> Self:
        return cls.model_construct(  # type: ignore[return-value]
            href="blob://local",
            _data=BlobNumpyDict(
                arrays,
                media_type=cls.default_media_type()
            ),
        )



def raw2rggb(raw: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Convert packed 10 bit raw to RGGB 8 bit"""
    raw = np.asarray(raw)  # ensure it's an array
    output_shape = (size[1]//2, size[0]//2, 4)
    rggb = np.empty(output_shape, dtype=np.uint8)
    raw_w = rggb.shape[1] // 2 * 5
    for plane, offset in enumerate([(1, 1), (0, 1), (1, 0), (0, 0)]):
        rggb[:, ::2, plane] = raw[offset[0] :: 2, offset[1] : raw_w + offset[1] : 5]
        rggb[:, 1::2, plane] = raw[
            offset[0] :: 2, offset[1] + 2 : raw_w + offset[1] + 2 : 5
        ]
    return rggb


def rggb2rgb(rggb: np.ndarray) -> np.ndarray:
    """Convert rggb to rgb by averaging green channels"""
    return np.stack(
        [rggb[..., 0], rggb[..., 1] // 2 + rggb[..., 2] // 2, rggb[..., 3]], axis=2
    )


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
            "ColourGains": (1, 1),
            "Contrast": 1,
            "ExposureTime": 0,
            "Saturation": 1,
            "Sharpness": 1,
        }
        self.persistent_control_tolerances = {
            "ExposureTime": 30,
        }

    def update_persistent_controls(self, discard_frames: int = 1):
        """Update the persistent controls dict from the camera

        Query the camera and update the value of `persistent_controls` to
        match the current state of the camera.

        There is a work-around here, that will suppress small updates. There
        appears to be a bug in the camera code that causes a slight drift in
        `ExposureTime` each time the camera is reinitialised: this can
        add up over time, particularly if the camera is reconfigured many
        times. To get around this, we look in `self.persistent_control_tolerances`
        and only update `self.persistent_controls` if the change is greater than
        this tolerance.
        """
        with self.picamera() as cam:
            for i in range(discard_frames):
                # Discard frames, so we know our data is fresh
                cam.capture_metadata()
            for k, v in cam.capture_metadata().items():
                if k in self.persistent_controls:
                    if k in self.persistent_control_tolerances:
                        if (
                            np.abs(self.persistent_controls[k] - v)
                            < self.persistent_control_tolerances[k]
                        ):
                            logging.debug(
                                f"Ignoring a small change in persistent control {k}"
                                f"from {self.persistent_controls[k]} to {v}"
                                "while updating persistent controls."
                            )
                            continue  # Ignore small changes, to avoid drift
                    self.persistent_controls[k] = v
        self.thing_settings.update(
            self.persistent_controls
        )  # TODO: Is this saving to the wrong place?

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
        initial_value=(820, 616),
        description="Resolution to use for the MJPEG stream",
    )
    mjpeg_bitrate = PropertyDescriptor(
        Optional[int],
        initial_value=100000000,
        description="Bitrate for MJPEG stream (None for default)",
    )

    @mjpeg_bitrate.setter
    def mjpeg_bitrate(self, value: Optional[int]):
        """Restart the stream when we set the bitrate"""
        with self.picamera(pause_stream=True):
            pass  # just pausing and restarting the stream is enough.

    stream_active = PropertyDescriptor(
        bool,
        initial_value=False,
        description="Whether the MJPEG stream is active",
        observable=True,
        readonly=True,
    )
    mjpeg_stream = MJPEGStreamDescriptor()
    lores_mjpeg_stream = MJPEGStreamDescriptor()
    analogue_gain = PicameraControl("AnalogueGain", float)
    colour_gains = PicameraControl("ColourGains", tuple[float, float])

    exposure_time = PicameraControl(
        "ExposureTime", int, description="The exposure time in microseconds"
    )

    _sensor_modes = None

    @thing_property
    def sensor_modes(self) -> list[SensorMode]:
        """All the available modes the current sensor supports"""
        if not self._sensor_modes:
            with self.picamera() as cam:
                self._sensor_modes = cam.sensor_modes
        return self._sensor_modes

    @thing_property
    def sensor_mode(self) -> Optional[SensorModeSelector]:
        """The intended sensor mode of the camera"""
        return self.thing_settings["sensor_mode"]

    @sensor_mode.setter
    def sensor_mode(self, new_mode: Optional[SensorModeSelector]):
        """Change the sensor mode used"""
        if isinstance(new_mode, SensorModeSelector):
            new_mode = new_mode.model_dump()
        with self.picamera(pause_stream=True):
            self.thing_settings["sensor_mode"] = new_mode

    @thing_property
    def sensor_resolution(self) -> tuple[int, int]:
        """The native resolution of the camera's sensor"""
        with self.picamera() as cam:
            return cam.sensor_resolution

    tuning = PropertyDescriptor(Optional[dict], None, readonly=True)

    def settings_to_properties(self):
        """Set the values of properties based on the settings dict"""
        try:
            props = self.thing_settings["properties"]
        except KeyError:
            return
        for k, v in props.items():
            setattr(self, k, v)

    def properties_to_settings(self):
        """Save certain properties to the settings dictionary"""
        props = {}
        for k in ["mjpeg_bitrate", "stream_resolution"]:
            props[k] = getattr(self, k)
        self.thing_settings["properties"] = props

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
        with tempfile.NamedTemporaryFile("w") as tuning_file:
            # This duplicates logic in `Picamera2.__init__` to provide a tuning file
            # that will be read when the camera system initialises.
            # This is a necessary work-around until `picamera2` better supports
            # reinitialisation of the camera with new tuning.
            json.dump(self.tuning, tuning_file)
            tuning_file.flush()  # but leave it open as closing it will delete it
            os.environ["LIBCAMERA_RPI_TUNING_FILE"] = tuning_file.name
            # NB even though we've put the tuning file in the environment, we will
            # need to specify the filename in the `Picamera2` initialiser as otherwise
            # it will be overwritten with None.
            if hasattr(self, "_picamera") and self._picamera:
                print("Closing picamera object for reinitialisation")
                logging.info(
                    "Camera object already exists, closing for reinitialisation"
                )
                self._picamera.close()
                print("closed, deleting picamera")
                del self._picamera
                recalibrate_utils.recreate_camera_manager()
            print("[re]creating Picamera2 object")
            self._picamera = picamera2.Picamera2(
                camera_num=self.camera_num,
                tuning=self.tuning,
            )
        self._picamera_lock = RLock()

    def __enter__(self):
        self.populate_default_tuning()
        self.initialise_tuning()
        self.initialise_picamera()
        self.sensor_modes
        self.settings_to_persistent_controls()
        self.settings_to_properties()
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
        already_streaming = self.stream_active
        with self._picamera_lock:
            if pause_stream and already_streaming:
                self.update_persistent_controls()
                self.stop_streaming(stop_web_stream=False)
            try:
                yield self._picamera
            finally:
                if pause_stream and already_streaming:
                    self.start_streaming()

    def populate_default_tuning(self):
        """Sensor modes are enumerated and stored, once, on start-up (`__enter__`).

        This opens and closes the camera - must be run before the camera is
        initialised.
        """
        logging.info("Starting & reconfiguring camera to populate sensor_modes.")
        with Picamera2(camera_num=self.camera_num) as cam:
            self.default_tuning = recalibrate_utils.load_default_tuning(cam)
        logging.info("Done reading sensor modes & default tuning.")

    def __exit__(self, exc_type, exc_value, traceback):
        # Allow key controls to persist across restarts
        self.update_persistent_controls()
        self.thing_settings["persistent_controls"] = self.persistent_controls
        self.thing_settings["tuning"] = self.tuning
        self.properties_to_settings()
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
            # TODO: Filip: can we use the lores output to keep preview stream going
            # while recording? According to picamera2 docs 4.2.1.6 this should work
            try:
                if picam.started:
                    picam.stop()
                    picam.stop_encoder()  # make sure there are no other encoders going
                stream_config = picam.create_video_configuration(
                    main={"size": self.stream_resolution},
                    lores={"size": (320, 240), "format": "YUV420"},
                    sensor=self.thing_settings.get("sensor_mode", None),
                    controls=self.persistent_controls,
                )
                picam.configure(stream_config)
                logging.info("Starting picamera MJPEG stream...")
                picam.start_recording(
                    MJPEGEncoder(self.mjpeg_bitrate),
                    PicameraStreamOutput(
                        self.mjpeg_stream,
                        get_blocking_portal(self),
                    ),
                )
                picam.start_encoder(
                    MJPEGEncoder(100000000),
                    PicameraStreamOutput(
                        self.lores_mjpeg_stream,
                        get_blocking_portal(self),
                    ),
                    name="lores",
                )
            except Exception as e:
                logging.exception("Error while starting preview: {e}")
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
                picam.stop_recording()  # This should also stop the extra lores encoder
            except Exception as e:
                logging.info("Stopping recording failed")
                logging.exception(e)
            else:
                self.stream_active = False
                if stop_web_stream:
                    self.mjpeg_stream.stop()
                    self.lores_mjpeg_stream.stop()
                logging.info("Stopped MJPEG stream.")

            # Increase the resolution for taking an image
            time.sleep(
                0.2
            )  # Sprinkled a sleep to prevent camera getting confused by rapid commands

    @thing_action
    def snap_image(self) -> ArrayModel:
        """Acquire one image from the camera.

        This action cannot run if the camera is in use by a background thread, for
        example if a preview stream is running.
        """
        return self.capture_array()

    @thing_action
    def capture_array(
        self,
        stream_name: Literal["main", "lores", "raw"] = "main",
        wait: Optional[float] = None,
    ) -> ArrayModel:
        """Acquire one image from the camera and return as an array

        This function will produce a nested list containing an uncompressed RGB image.
        It's likely to be highly inefficient - raw and/or uncompressed captures using
        binary image formats will be added in due course.

        stream_name: (Optional) The PiCamera2 stream to use, should be one of ["main", "lores", "raw"]. Default = "main"
        wait: (Optional, float) Set a timeout in seconds. A TimeoutError is raised if this time is exceeded during capture. Default = None
        """
        with self.picamera() as cam:
            return cam.capture_array(stream_name, wait = wait)

    @thing_action
    def capture_raw(
        self,
        states_getter: GetThingStates,
        get_states: bool=True,
        get_processing_inputs: bool=True,
    ) -> RawImageModel:
        """Capture a raw image
        
        This function is intended to be as fast as possible, and will return
        as soon as an image has been captured. The output format is not intended
        to be useful, except as input to `raw_to_png`. 
        
        When used via the HTTP interface, this function returns the data as a
        `Blob` object, meaning it can be passed to another action without
        transferring it over the network.
        """
        with self.picamera() as cam:
            (buffer, ), parameters = cam.capture_buffers(["raw"])
            configuration = cam.camera_configuration()
        return RawImageModel(
            image_data = RawBlob.from_bytes(buffer.tobytes()),
            thing_states = states_getter() if get_states else None,
            metadata = { "parameters": parameters, "sensor": configuration["sensor"], "tuning": self.tuning },
            processing_inputs = (
                self.image_processing_inputs if get_processing_inputs else None
            ),
            size = configuration["raw"]["size"],
            format = configuration["raw"]["format"],
            stride = configuration["raw"]["stride"],
        )

    @thing_property
    def image_processing_inputs(self) -> ImageProcessingInputs:
        """The information needed to turn raw images into processed ones"""
        lst = self.lens_shading_tables
        lum = np.array(lst.luminance)
        Cr = np.array(lst.Cr)
        Cb = np.array(lst.Cb)
        gr, gb = self.colour_gains
        G = 1 / lum
        R = (
            G / Cr / gr * np.min(Cr)
        )  # The extra /np.max(Cr) emulates the quirky handling of Cr in
        B = G / Cb / gb * np.min(Cb)  # the picamera2 pipeline
        white_norm_lores = np.stack([R, G, B], axis=2)

        with self.picamera() as cam:
            size: tuple[int, int] = cam.camera_configuration()["raw"]["size"]

        contrast_algorithm = Picamera2.find_tuning_algo(self.tuning, "rpi.contrast")
        gamma = np.array(contrast_algorithm["gamma_curve"]).reshape((-1, 2))

        return ImageProcessingInputs(
            lens_shading=lst,
            colour_gains=(gr, gb),
            colour_correction_matrix=self.colour_correction_matrix,
            white_norm_lores=white_norm_lores,
            raw_size=size,
            gamma=gamma,
        )

    @staticmethod
    def generate_image_processing_cache(
        p: ImageProcessingInputs,
    ) -> ImageProcessingCache:
        """Prepare to process raw images
        
        This is a static method to ensure its outputs depend only on its
        inputs."""
        zoom_factors = [
            i / 2 / n for i, n in zip(p.raw_size[::-1], p.white_norm_lores.shape[:2])
        ] + [1]
        white_norm = zoom(p.white_norm_lores, zoom_factors, order=1)[
            : (p.raw_size[1]//2), : (p.raw_size[0]//2), :
        ]
        ccm = np.array(p.colour_correction_matrix).reshape((3,3))
        gamma = interp1d(p.gamma[:, 0] / 255, p.gamma[:, 1] / 255)
        return ImageProcessingCache(
            white_norm=white_norm,
            ccm = ccm,
            gamma = gamma,
        )

    _image_processing_cache: ImageProcessingCache | None = None
    @thing_action
    def prepare_image_normalisation(
        self,
        inputs: ImageProcessingInputs | None = None
    ) -> ImageProcessingInputs:
        """The parameters used to convert raw image data into processed images
        
        NB this method uses only information from `inputs` or 
        `self.image_processing_inputs`, to ensure repeatability
        """
        p = inputs or self.image_processing_inputs
        self._image_processing_cache = self.generate_image_processing_cache(p)
        return p

    @thing_action
    def process_raw_array(
        self,
        raw: RawImageModel,
        use_cache: bool = False,
    )->NDArray:
        """Convert a raw image to a processed array"""
        if not use_cache:
            if raw.processing_inputs is None:
                raise ValueError(
                    "The raw image does not contain processing inputs, "
                    "and we are not using the cache. This may be solved by "
                    "capturing with `get_processing_inputs=True`."
                )
            self.prepare_image_normalisation(
                raw.processing_inputs
            )
        p = self._image_processing_cache
        assert p is not None
        assert raw.format == "SBGGR10_CSI2P"
        buffer = np.frombuffer(raw.image_data.content, dtype=np.uint8)
        packed = buffer.reshape((-1, raw.stride))
        rgb = rggb2rgb(raw2rggb(packed, raw.size))
        normed = rgb / p.white_norm
        corrected = np.dot(
            p.ccm, normed.reshape((-1, 3)).T
        ).T.reshape(normed.shape)
        corrected[corrected < 0] = 0
        corrected[corrected > 255] = 255
        processed_image = p.gamma(corrected)
        return processed_image.astype(np.uint8)

    @thing_action
    def raw_to_png(self, raw: RawImageModel, use_cache: bool = False)->PNGBlob:
        """Process a raw image to a PNG"""
        arr = self.process_raw_array(raw=raw, use_cache=use_cache)
        image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        out = io.BytesIO()
        image.save(out, format="png")
        return PNGBlob.from_bytes(out.getvalue())

    @thing_property
    def camera_configuration(self) -> Mapping:
        """The "configuration" dictionary of the picamera2 object

        The "configuration" sets the resolution and format of the camera's streams.
        Together with the "tuning" it determines how the sensor is configured and
        how the data is processed.

        Note that the configuration may be modified when taking still images, and
        this property refers to whatever configuration is currently in force -
        usually the one used for the preview stream.
        """
        with self.picamera() as cam:
            return cam.camera_configuration()

    @thing_action
    def capture_jpeg(
        self,
        metadata_getter: GetThingStates,
        resolution: Literal["lores", "main", "full"] = "main",
    ) -> JPEGBlob:
        """Acquire one image from the camera as a JPEG

        The JPEG will be acquired using `Picamera2.capture_file`. If the
        `resolution` parameter is `main` or `lores`, it will be captured
        from the main preview stream, or the low-res preview stream,
        respectively. This means the camera won't be reconfigured, and
        the stream will not pause (though it may miss one frame).

        If `full` resolution is requested, we will briefly pause the
        MJPEG stream and reconfigure the camera to capture a full
        resolution image.

        Note that this always uses the image processing pipeline - to
        bypass this, you must use a raw capture.
        """
        fname = datetime.now().strftime("%Y-%m-%d-%H%M%S.jpeg")
        folder = TemporaryDirectory()
        path = os.path.join(folder.name, fname)
        config = self.camera_configuration
        # Low-res and main streams are running already - so we don't need
        # to reconfigure for these
        if resolution in ("lores", "main") and config[resolution]:
            with self.picamera() as cam:
                cam.capture_file(path, name=resolution, format="jpeg")
        else:
            if resolution != "full":
                logging.warning(
                    f"There was no {resolution} stream, capturing full resolution"
                )
            with self.picamera(pause_stream=True) as cam:
                logging.info("Reconfiguring camera for full resolution capture")
                cam.configure(cam.create_still_configuration())
                cam.start()
                logging.info("capturing")
                cam.capture_file(path, name="main", format="jpeg")
                logging.info("done")
        # After the file is written, add metadata about the current Things
        exif_dict = piexif.load(path)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = json.dumps(
            metadata_getter()
        ).encode("utf-8")
        piexif.insert(piexif.dump(exif_dict), path)
        return JPEGBlob.from_temporary_directory(folder, fname)

    @thing_action
    def grab_jpeg(
        self,
        portal: BlockingPortal,
        stream_name: Literal["main", "lores"] = "main",
    ) -> JPEGBlob:
        """Acquire one image from the preview stream and return as an array

        This differs from `capture_jpeg` in that it does not pause the MJPEG
        preview stream. Instead, we simply return the next frame from that
        stream (either "main" for the preview stream, or "lores" for the low
        resolution preview). No metadata is returned.
        """
        logging.debug(
            f"StreamingPiCamera2.grab_jpeg(stream_name={stream_name}) starting"
        )
        stream = (
            self.lores_mjpeg_stream if stream_name == "lores" else self.mjpeg_stream
        )
        frame = portal.call(stream.grab_frame)
        logging.debug(
            f"StreamingPiCamera2.grab_jpeg(stream_name={stream_name}) got frame"
        )
        return JPEGBlob.from_bytes(frame)

    @thing_action
    def grab_jpeg_size(
        self,
        portal: BlockingPortal,
        stream_name: Literal["main", "lores"] = "main",
    ) -> int:
        """Acquire one image from the preview stream and return its size"""
        stream = (
            self.lores_mjpeg_stream if stream_name == "lores" else self.mjpeg_stream
        )
        return portal.call(stream.next_frame_size)

    # @thing_action
    # def capture_to_scan(
    #     self,
    #     scan_manager: ScanManager,
    #     format: Literal["jpeg"] = "jpeg",
    # ) -> None:
    #     with scan_manager.new_jpeg() as output, self.picamera() as cam:
    #         cam.capture_file(output, format="jpeg")

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
        target_white_level: int = 700,
        percentile: float = 99.9,
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
    def calibrate_white_balance(
        self,
        method: Literal["percentile", "centre"] = "centre",
        luminance_power: float = 1.0,
    ):
        """Correct the white balance of the image

        This calibration requires a neutral image, such that the 99th centile
        of each colour channel should correspond to white. We calculate the
        centiles and use this to set the colour gains. This is done on the raw
        image with the lens shading correction applied, which should mean
        that the image is uniform, rather than weighted towards the centre.

        If `method` is `"centre"`, we will correct the mean of the central 10%
        of the image.
        """
        with self.picamera(pause_stream=True) as cam:
            if self.lens_shading_is_static:
                lst: LensShading = self.lens_shading_tables
                recalibrate_utils.adjust_white_balance_from_raw(
                    cam,
                    percentile=99,
                    luminance=lst.luminance,
                    Cr=lst.Cr,
                    Cb=lst.Cb,
                    luminance_power=luminance_power,
                    method=method,
                )
            else:
                recalibrate_utils.adjust_white_balance_from_raw(
                    cam, percentile=99, method=method
                )
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
            recalibrate_utils.set_static_lst(self.tuning, L, Cr, Cb)
            self.initialise_picamera()

    @thing_property
    def colour_correction_matrix(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """An alias for `colour_correction_matrix` to fit the micromanager API"""
        return self.thing_settings.get(
            "colour_correction_matrix",
            tuple(recalibrate_utils.get_static_ccm(self.tuning)[0]["ccm"]),
        )

    @colour_correction_matrix.setter  # type: ignore
    def colour_correction_matrix(self, value):
        self.thing_settings["colour_correction_matrix"] = value
        self.calibrate_colour_correction(value)

    @thing_action
    def reset_ccm(self):
        """Overwrite the colour correction matrix in camera tuning with default values from the documentation"""
        c = [
            1.80439,
            -0.73699,
            -0.06739,
            -0.36073,
            1.83327,
            -0.47255,
            -0.08378,
            -0.56403,
            1.64781,
        ]
        self.colour_correction_matrix = c

    @thing_action
    def calibrate_colour_correction(self, c: tuple):
        """Overwrite the colour correction matrix in camera tuning"""
        with self.picamera(pause_stream=True):
            recalibrate_utils.set_static_ccm(self.tuning, c)
            self.initialise_picamera()

    @thing_action
    def set_static_green_equalisation(self, offset: int = 65535):
        """Set the green equalisation to a static value.

        Green equalisation avoids the debayering algorithm becoming confused
        by the two green channels having different values, which is a problem
        when the chief ray angle isn't what the sensor was designed for, and
        that's the case in e.g. a microscope using camera module v2.

        A value of 0 here does nothing, a value of 65535 is maximum correction.
        """
        with self.picamera(pause_stream=True):
            recalibrate_utils.set_static_geq(self.tuning, offset)
            self.initialise_picamera()

    @thing_action
    def full_auto_calibrate(self):
        """Perform a full auto-calibration

        This function will call the other calibration actions in sequence:

        * `flat_lens_shading` to disable flat-field
        * `auto_expose_from_minimum`
        * `set_static_green_equalisation` to set geq offset to max
        * `calibrate_lens_shading`
        * `calibrate_white_balance`
        """
        self.flat_lens_shading()
        self.auto_expose_from_minimum()
        self.set_static_green_equalisation()
        self.calibrate_lens_shading()
        self.calibrate_white_balance()

    @thing_action
    def flat_lens_shading(self):
        """Disable flat-field correction

        This method will set a completely flat lens shading table. It is not the
        same as the default behaviour, which is to use an adaptive lens shading
        table.
        """
        with self.picamera(pause_stream=True):
            f = np.ones((12, 16))
            recalibrate_utils.set_static_lst(self.tuning, f, f, f)
            self.initialise_picamera()

    @thing_property
    def lens_shading_tables(self) -> Optional[LensShading]:
        """The current lens shading (i.e. flat-field correction)

        This returns the current lens shading correction, as three 2D lists
        each with dimensions 16x12. This assumes that we are using a static
        lens shading table - if adaptive control is enabled, or if there
        are multiple LSTs in use for different colour temperatures,
        we return a null value to avoid confusion.
        """
        if not self.lens_shading_is_static:
            return None
        alsc = Picamera2.find_tuning_algo(self.tuning, "rpi.alsc")
        if any(len(alsc[f"calibrations_C{c}"]) != 1 for c in ("r", "b")):
            return None

        def reshape_lst(lin: list[float]) -> list[list[float]]:
            w, h = 16, 12
            return [lin[w * i : w * (i + 1)] for i in range(h)]

        return LensShading(
            luminance=reshape_lst(alsc["luminance_lut"]),
            Cr=reshape_lst(alsc["calibrations_Cr"][0]["table"]),
            Cb=reshape_lst(alsc["calibrations_Cb"][0]["table"]),
        )

    @lens_shading_tables.setter
    def lens_shading_tables(self, lst: LensShading) -> None:
        """Set the lens shading tables"""
        with self.picamera(pause_stream=True):
            recalibrate_utils.set_static_lst(
                self.tuning,
                luminance=lst.luminance,
                cr=lst.Cr,
                cb=lst.Cb,
            )
            self.initialise_picamera()

    def correct_colour_gains_for_lens_shading(
        self, colour_gains: tuple[float, float]
    ) -> tuple[float, float]:
        """Correct white balance gains for the effect of lens shading

        The white balance algorithm we use assumes the brightest pixels
        should be white, and that the only thing affecting the colour of
        said pixels is the `colour_gains`.

        The lens shading correction is normalised such that the *minimum*
        gain in the `Cr` and `Cb` channels is 1. The white balance
        assumption above requires that the gain for the brightest pixels
        is 1. The solution might be that, when calibrating, we note which
        pixels are brightest (usually the centre) and explicitly use
        the LST values for there. However, for now I will assume that we
        need to normalise by the **maximum** of the `Cr` and `Cb`
        channels, which is correct the majority of the time.
        """
        if not self.lens_shading_is_static:
            return colour_gains
        lst = self.lens_shading_tables
        # The Cr and Cb corrections are normalised to have a minimum of 1,
        # but the white balance algorithm normalises the brightest pixels
        # to be white, assuming the brightest pixels have equal gain from
        # the LST.
        gain_r, gain_b = colour_gains
        return (
            float(gain_r / np.max(lst.Cr)),
            float(gain_b / np.max(lst.Cb)),
        )

    @thing_action
    def flat_lens_shading_chrominance(self):
        """Disable flat-field correction

        This method will set the chrominance of the lens shading table to be
        flat, i.e. we'll correct vignetting of intensity, but not any change in
        colour across the image.
        """
        with self.picamera(pause_stream=True):
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
        with self.picamera(pause_stream=True):
            recalibrate_utils.copy_alsc_section(self.default_tuning, self.tuning)
            self.initialise_picamera()

    @thing_property
    def lens_shading_is_static(self) -> bool:
        """Whether the lens shading is static

        This property is true if the lens shading correction has been set to use
        a static table (i.e. the number of automatic correction iterations is zero).
        The default LST is not static, but all the calibration controls will set it
        to be static (except "reset")
        """
        return recalibrate_utils.lst_is_static(self.tuning)
