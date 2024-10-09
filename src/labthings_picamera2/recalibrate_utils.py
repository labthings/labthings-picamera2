"""
Functions to set up a Raspberry Pi Camera v2 for scientific use

This module provides slower, simpler functions to set the
gain, exposure, and white balance of a Raspberry Pi camera, using
`picamerax` (a fork of `picamera`) to get as-manual-as-possible
control over the camera.  It's mostly used by the OpenFlexure
Microscope, though it deliberately has no hard dependencies on
said software, so that it's useful on its own.

There are three main calibration steps:

* Setting exposure time and gain to get a reasonably bright
  image.
* Fixing the white balance to get a neutral image
* Taking a uniform white image and using it to calibrate
  the Lens Shading Table

The most reliable way to do this, avoiding any issues relating
to "memory" or nonlinearities in the camera's image processing
pipeline, is to use raw images.  This is quite slow, but very
reliable.  The three steps above can be accomplished by:

```
picamera = picamerax.PiCamera()

adjust_shutter_and_gain_from_raw(picamera)
adjust_white_balance_from_raw(picamera)
lst = lst_from_camera(picamera)
picamera.lens_shading_table = lst
```
"""
from __future__ import annotations
import logging
import time
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel
import numpy as np
from scipy.ndimage import zoom

from picamera2 import Picamera2


def load_default_tuning(cam: Picamera2) -> dict:
    """Load the default tuning file for the camera

    This will open and close the camera to determine its model. If you are
    using a model that's supported by `picamera2` it should have a tuning
    file built in. If not, this will probably crash with an error.

    Error handling for unsupported cameras is not something we are likely
    to test in the short term.
    """
    cp = cam.camera_properties
    fname = f"{cp['Model']}.json"
    try:
        return cam.load_tuning_file(fname)
    except RuntimeError:
        dir = "/usr/share/libcamera/ipa/raspberrypi"  # from picamera2 v0.3.9
        # The directory above has been removed from the search path, which I
        # find odd - as that's where the files currently are on a default
        # Raspbian image. This may need updating if the files have moved
        # in future updates to the system libcamera package
        return cam.load_tuning_file(fname, dir=dir)


def set_minimum_exposure(camera: Picamera2):
    """Enable manual exposure, with low gain and shutter speed

    We set exposure mode to manual, analog and digital gain
    to 1, and shutter speed to the minimum (8us for Pi Camera v2)
    NB ISO is left at auto, because this is needed for the gains
    to be set correctly.
    """
    camera.set_controls({"AeEnable": False, "AnalogueGain": 1, "ExposureTime": 1})
    # camera.iso = 0  # We must set ISO=0 (auto) or we can't set gain
    #  camera.analog_gain = 1
    # camera.digital_gain = 1 (not configurable)
    # Setting the shutter speed to 1us will result in it being set
    # to the minimum possible, which is probably 8us for PiCamera v2
    # camera.shutter_speed = 1
    time.sleep(0.5)


class ExposureTest(BaseModel):
    """Record the results of testing the camera's current exposure settings"""

    level: int
    exposure_time: int
    analog_gain: float


def test_exposure_settings(camera: Picamera2, percentile: float) -> ExposureTest:
    """Evaluate current exposure settings using a raw image

    CAMERA SHOULD BE STARTED!

    We will acquire a raw image and calculate the given percentile
    of the pixel values.  We return a dictionary containing the
    percentile (which will be compared to the target), as well as
    the camera's shutter and gain values.
    """
    camera.capture_array("raw")  # controls might not be updated for the first frame?
    max_brightness = np.percentile(
        channels_from_bayer_array(camera.capture_array("raw")),
        percentile,
    )
    # The reported brightness can, theoretically, be negative or zero
    # because of black level compensation.  The line below forces a
    # minimum value of 1 which will keep things well-behaved!
    if max_brightness < 1:
        logging.warning(
            f"Measured brightness of {max_brightness}. "
            "This should normally be >= 1, and may indicate the "
            "camera's black level compensation has gone wrong."
        )
        max_brightness = 1
    metadata = camera.capture_metadata()
    result = ExposureTest(
        level=max_brightness,
        exposure_time=int(metadata["ExposureTime"]),
        analog_gain=float(metadata["AnalogueGain"]),
    )
    logging.info(f"{result.model_dump()}")
    return result


def check_convergence(test: ExposureTest, target: int, tolerance: float):
    """Check whether the brightness is within the specified target range"""
    converged = abs(test.level - target) < target * tolerance
    return converged


def adjust_shutter_and_gain_from_raw(
    camera: Picamera2,
    target_white_level: int = 700,
    max_iterations: int = 20,
    tolerance: float = 0.05,
    percentile: float = 99.9,
) -> float:
    """Adjust exposure and analog gain based on raw images.

    This routine is slow but effective.  It uses raw images, so we
    are not affected by white balance or digital gain.


    Arguments:
        target_white_level:
            The raw, 10-bit value we aim for.  The brightest pixels
            should be approximately this bright.  Maximum possible
            is about 900, 700 is reasonable.
        max_iterations:
            We will terminate once we perform this many iterations,
            whether or not we converge.  More than 10 shouldn't happen.
        tolerance:
            How close to the target value we consider "done".  Expressed
            as a fraction of the ``target_white_level`` so 0.05 means
            +/- 5%
        percentile:
            Rather then use the maximum value for each channel, we
            calculate a percentile.  This makes us robust to single
            pixels that are bright/noisy.  99.9% still picks the top
            of the brightness range, but seems much more reliable
            than just ``np.max()``.
    """
    # TODO: read black level and bit depth from camera?
    if target_white_level * (tolerance + 1) >= 959:
        raise ValueError(
            "The target level is too high - a saturated image would be "
            "considered successful.  target_white_level * (tolerance + 1) "
            "must be less than 959."
        )

    config = camera.create_still_configuration(raw={"format": "SBGGR10"})
    camera.configure(config)
    camera.start()
    set_minimum_exposure(camera)

    # We start with very low exposure settings and work up
    # until either the brightness is high enough, or we can't increase the
    # shutter speed any more.
    iterations = 0
    while iterations < max_iterations:
        test = test_exposure_settings(camera, percentile)
        if check_convergence(test, target_white_level, tolerance):
            break
        iterations += 1

        # Adjust shutter speed so that the brightness approximates the target
        # NB we put a maximum of 8 on this, to stop it increasing too quickly.
        new_time = int(test.exposure_time * min(target_white_level / test.level, 8))
        camera.controls.ExposureTime = new_time
        camera.controls.AeEnable = False
        time.sleep(0.5)

        # Check whether the shutter speed is still going up - if not, we've hit a maximum
        if camera.capture_metadata()["ExposureTime"] == test.exposure_time:
            logging.info(f"Shutter speed has maxed out at {test.exposure_time}")
            break

    # Now, if we've not converged, increase gain until we converge or run out of options.
    while iterations < max_iterations:
        test = test_exposure_settings(camera, percentile)
        if check_convergence(test, target_white_level, tolerance):
            break
        iterations += 1

        # Adjust gain to make the white level hit the target, again with a maximum
        camera.controls.AnalogueGain = test.analog_gain * min(
            target_white_level / test.level, 2
        )
        time.sleep(0.5)

        # Check the gain is still changing - if not, we have probably hit the maximum
        if camera.capture_metadata()["AnalogueGain"] == test.analog_gain:
            logging.info(f"Gain has maxed out. at {test.analog_gain}")
            break

    if check_convergence(test, target_white_level, tolerance):
        logging.info(f"Brightness has converged to within {tolerance * 100 :.0f}%.")
    else:
        logging.warning(
            f"Failed to reach target brightness of {target_white_level}."
            f"Brightness reached {test.level} after {iterations} iterations."
        )

    return test.level


def adjust_white_balance_from_raw(
    camera: Picamera2,
    percentile: float = 99,
    luminance: Optional[np.ndarray] = None,
    Cr: Optional[np.ndarray] = None,
    Cb: Optional[np.ndarray] = None,
    luminance_power: float = 1.0,
    method: Literal["percentile", "centre"] = "centre",
) -> Tuple[float, float]:
    """Adjust the white balance in a single shot, based on the raw image.

    NB if ``channels_from_raw_image`` is broken, this will go haywire.
    We should probably have better logic to verify the channels really
    are BGGR...
    """
    config = camera.create_still_configuration(raw={"format": "SBGGR10"})
    camera.configure(config)
    camera.start()
    channels = channels_from_bayer_array(camera.capture_array("raw"))
    #logging.info(f"White balance: channels were retrieved with shape {channels.shape}.")
    if luminance is not None and Cr is not None and Cb is not None:
        # Reconstruct a low-resolution image from the lens shading tables
        # and use it to normalise the raw image, to compensate for
        # the brightest pixels in each channel not coinciding.
        grids = grids_from_lst(np.array(luminance)**luminance_power, Cr, Cb)
        channel_gains = 1/grids
        if channel_gains.shape[1:] != channels.shape[1:]:
            channel_gains = upsample_channels(channel_gains, channels.shape[1:])
        logging.info(f"Before gains, channel maxima are {np.max(channels, axis=(1,2))}")
        channels = channels * channel_gains
        logging.info(f"After gains, channel maxima are {np.max(channels, axis=(1,2))}")
    if method == "centre":
        _, h, w = channels.shape
        blue, g1, g2, red = np.mean(
            channels[:, 9*h//20:11*h//20, 9*w//20:11*w//20],
            axis=(1,2),
        ) - 64
    else:
        # TODO: read black level from camera rather than hard-coding 64
        blue, g1, g2, red = np.percentile(channels, percentile, axis=(1, 2)) - 64
    green = (g1 + g2) / 2.0
    new_awb_gains = (green / red, green / blue)
    if Cr is not None and Cb is not None:
        # The LST algorithm normalises Cr and Cb by their minimum. 
        # The lens shading correction only ever boosts the red and blue values.
        # Here, we decrease the gains by the minimum value of Cr and Cb.
        new_awb_gains = (green / red * np.min(Cr), green / blue * np.min(Cb))

    logging.info(
        f"Raw white point is R: {red} G: {green} B: {blue}, "
        f"setting AWB gains to ({new_awb_gains[0]:.2f}, "
        f"{new_awb_gains[1]:.2f})."
    )
    camera.controls.AwbEnable = False
    camera.controls.ColourGains = new_awb_gains
    time.sleep(0.2)
    m = camera.capture_metadata()
    print(f"Camera confirms gains are now {m['ColourGains']}")
    return new_awb_gains


def channels_from_bayer_array(bayer_array: np.ndarray) -> np.ndarray:
    """Given the 'array' from a PiBayerArray, return the 4 channels."""
    bayer_pattern: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bayer_array = bayer_array.view(np.uint16)
    channels_shape: Tuple[int, int, int] = (
        4,
        bayer_array.shape[0] // 2,
        bayer_array.shape[1] // 2,
    )
    channels: np.ndarray = np.zeros(channels_shape, dtype=bayer_array.dtype)
    for i, offset in enumerate(bayer_pattern):
        # We simplify life by dealing with only one channel at a time.
        channels[i, :, :] = bayer_array[offset[0] :: 2, offset[1] :: 2]

    return channels


LensShadingTables = tuple[np.ndarray, np.ndarray, np.ndarray]


def get_16x12_grid(chan: np.ndarray, dx: int, dy: int):
    """Compresses channel down to a 16x12 grid - from libcamera

    This is taken from https://git.linuxtv.org/libcamera.git/tree/utils/raspberrypi/ctt/ctt_alsc.py
    for consistency.
    """
    grid = []
    """
    since left and bottom border will not necessarily have rectangles of
    dimension dx x dy, the 32nd iteration has to be handled separately.
    """
    for i in range(11):
        for j in range(15):
            grid.append(np.mean(chan[dy*i:dy*(1+i), dx*j:dx*(1+j)]))
        grid.append(np.mean(chan[dy*i:dy*(1+i), 15*dx:]))
    for j in range(15):
        grid.append(np.mean(chan[11*dy:, dx*j:dx*(1+j)]))
    grid.append(np.mean(chan[11*dy:, 15*dx:]))
    """
    return as np.array, ready for further manipulation
    """
    return np.reshape(np.array(grid), (12, 16))

def upsample_channels(grids: np.ndarray, shape: tuple[int]):
    """Zoom an image in the last two dimensions

    This is effectively the inverse operation of `get_16x12_grid`
    """
    zoom_factors = [1,] + list(np.ceil(np.array(shape)/np.array(grids.shape[1:])))
    return zoom(grids, zoom_factors, order=1)[:, :shape[0], :shape[1]]

def downsampled_channels(channels: np.ndarray, blacklevel=64) -> list[np.ndarray]:
    """Generate a downsampled, un-normalised image from which to calculate the LST

    TODO: blacklevel probably ought to be determined from the camera...
    """
    channel_shape = np.array(channels.shape[1:])
    lst_shape = np.array([12,16])
    step = np.ceil(channel_shape/lst_shape).astype(int)
    return np.stack(
        [
            get_16x12_grid(channels[i, ...].astype(float) - blacklevel, step[1], step[0])
            for i in range(channels.shape[0])
        ],
        axis=0,
    )

def lst_from_channels(channels: np.ndarray) -> LensShadingTables:
    """Given the 4 Bayer colour channels from a white image, generate a LST.

    Internally, is just calls `downsampled_channels` and `lst_from_grids`.
    """
    grids = downsampled_channels(channels)
    return lst_from_grids(grids)


def lst_from_grids(grids: np.ndarray) -> LensShadingTables:
    """Given 4 downsampled grids, generate the luminance and chrominance tables

    The LST format has changed with `picamera2` and now uses a fixed resolution,
    and is in luminance, Cr, Cb format. This function returns three ndarrays of
    luminance, Cr, Cb, each with shape (12, 16).

    # TODO: make consistent with
    https://git.linuxtv.org/libcamera.git/tree/utils/raspberrypi/ctt/ctt_alsc.py
    """
    r: np.ndarray = grids[3, ...] 
    g: np.ndarray = np.mean(grids[1:3, ...], axis=0)
    b: np.ndarray = grids[0, ...]

    # What we actually want to calculate is the gains needed to compensate for the
    # lens shading - that's 1/lens_shading_table_float as we currently have it.
    luminance_gains: np.ndarray = np.max(g) / g  # Minimum luminance gain is 1
    cr_gains: np.ndarray = g / r
    #cr_gains /= cr_gains[5, 7]  # Normalise so the central colour doesn't change
    cb_gains: np.ndarray = g / b
    #cb_gains /= cb_gains[5, 7]
    return luminance_gains, cr_gains, cb_gains

def grids_from_lst(lum: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """Convert form luminance/chrominance dict to four RGGB channels
    
    Note that these will be normalised - the maximum green value is always 1.
    Also, note that the channels are BGGR, to be consistent with the
    `channels_from_raw_image` function. This should probably change in the
    future.
    """
    G = 1/np.array(lum)
    R = G/np.array(Cr)
    B = G/np.array(Cb)
    return np.stack([B, G, G, R], axis=0)

def set_static_lst(
    tuning: dict,
    luminance: np.ndarray,
    cr: np.ndarray,
    cb: np.ndarray,
) -> None:
    """Update the `rpi.alsc` section of a camera tuning dict to use a static correcton.

    `tuning` will be updated in-place to set its shading to static, and disable any
    adaptive tweaking by the algorithm.
    """
    for table in luminance, cr, cb:
        assert np.array(table).shape == (12, 16), "Lens shading tables must be 12x16!"
    alsc = Picamera2.find_tuning_algo(tuning, "rpi.alsc")
    alsc["n_iter"] = 0  # disable the adaptive part
    alsc["luminance_strength"] = 1.0
    alsc["calibrations_Cr"] = [
        {"ct": 4500, "table": np.reshape(cr, (-1)).round(3).tolist()}
    ]
    alsc["calibrations_Cb"] = [
        {"ct": 4500, "table": np.reshape(cb, (-1)).round(3).tolist()}
    ]
    alsc["luminance_lut"] = np.reshape(luminance, (-1)).round(3).tolist()

def set_static_ccm(
    tuning: dict,
    c: list
) -> None:
    """Update the `rpi.alsc` section of a camera tuning dict to use a static correcton.

    `tuning` will be updated in-place to set its shading to static, and disable any
    adaptive tweaking by the algorithm.
    """
    ccm = Picamera2.find_tuning_algo(tuning, "rpi.ccm")
    ccm["ccms"] = [{
        "ct": 2860,
        "ccm": c
        }
    ]

def get_static_ccm(
    tuning: dict
) -> None:
    """Get the `rpi.ccm` section of a camera tuning dict
    """
    ccm = Picamera2.find_tuning_algo(tuning, "rpi.ccm")
    return ccm["ccms"]

def lst_is_static(tuning: dict) -> bool:
    """Whether the lens shading table is set to static"""
    alsc = Picamera2.find_tuning_algo(tuning, "rpi.alsc")
    return alsc["n_iter"] == 0


def index_of_algorithm(algorithms: list[dict], algorithm: str):
    """Find the index of an algorithm's section in the tuning file"""
    for i, a in enumerate(algorithms):
        if algorithm in a:
            return i


def copy_alsc_section(from_tuning: dict, to_tuning: dict):
    """Copy the `rpi.alsc` algorithm from one tuning to another.

    This is done in-place, i.e. modifying to_tuning.
    """
    from_i = index_of_algorithm(from_tuning["algorithms"], "rpi.alsc")
    to_i = index_of_algorithm(to_tuning["algorithms"], "rpi.alsc")
    # Please excuse the clumsy update-and-delete - this lets us use
    # the nice Picamera2 function to find the relevant sub-dict.
    to_tuning["algorithms"][to_i] = from_tuning["algorithms"][from_i]


def lst_from_camera(camera: Picamera2) -> LensShadingTables:
    """Acquire a raw image and use it to calculate a lens shading table."""
    channels = raw_channels_from_camera(camera)
    return lst_from_channels(channels)

def raw_channels_from_camera(camera: Picamera2) -> LensShadingTables:
    """Acquire a raw image and return a 4xNxM array of the colour channels."""
    if camera.started:
        camera.stop_recording()
    # We will acquire a raw image with unpacked pixels, which is what the
    # format below requests. Bit depth and Bayer order may be overwritten.
    # TODO: don't assume 10-bit - the high quality camera uses 12.
    # TODO: what's the best mode to use here?
    config = camera.create_still_configuration(raw={"format": "SBGGR10"})
    camera.configure(config)
    camera.start()
    raw_image = camera.capture_array("raw")
    camera.stop()
    # Now we need to calculate a lens shading table that would make this flat.
    # raw_image is a 3D array, with full resolution and 3 colour channels.  No
    # de-mosaicing has been done, so 2/3 of the values are zero (3/4 for R and B
    # channels, 1/2 for green because there's twice as many green pixels).
    format = camera.camera_configuration()["raw"]["format"]
    print(f"Acquired a raw image in format {format}")
    return channels_from_bayer_array(raw_image)


if __name__ == "__main__":
    """This block is untested but has been updated."""
    with Picamera2() as cam:
        tuning = load_default_tuning(cam)
    f = np.ones((12, 16))
    set_static_lst(tuning, f, f, f)
    with Picamera2(tuning=tuning) as cam:
        cam.start_preview()
        time.sleep(3)
        logging.info("Recalibrating...")
        adjust_shutter_and_gain_from_raw(cam)
        adjust_white_balance_from_raw(cam)
        lst = lst_from_camera(cam)
        set_static_lst(tuning, *lst)
        logging.info("Done.")
    with Picamera2(tuning=tuning) as cam:
        cam.start_preview()
        time.sleep(2)
