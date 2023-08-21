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
from typing import List, NamedTuple, Optional, Tuple
from pydantic import BaseModel
import numpy as np

from libcamera import controls
from picamera2 import Picamera2


def rgb_image(
    camera: Picamera2, resize: Optional[Tuple[int, int]] = None, **kwargs
) -> np.array:
    """Capture an image and return an RGB numpy array"""
    return camera.capture_array()

def flat_lens_shading_table(camera: Picamera2, scamera: PiCamera2Streamer) -> np.ndarray:
    """Return a flat (i.e. unity gain) lens shading table.

    This is mostly useful because it makes it easy to get the size
    of the array correct.  NB if you are not using the forked picamera
    library (with lens shading table support) it will raise an error.
    """
    set_lst_values(np.ones((4,16,12)), scamera)
    return np.ones((4,16,12))

def adjust_exposure_to_setpoint(camera: Picamera2, setpoint: int):
    """Adjust the camera's exposure time until the maximum pixel value is <setpoint>.

    NB this method uses RGB images (i.e. processed ones) not raw images.
    """

    logging.info(f"Adjusting shutter speed to hit setpoint {setpoint}")
    for _ in range(3):
        camera.controls.ExposureTime = int(
            camera.controls.ExposureTime * setpoint / np.max(rgb_image(camera))
        )
        time.sleep(1)


def set_minimum_exposure(camera: Picamera2):
    """Enable manual exposure, with low gain and shutter speed

    We set exposure mode to manual, analog and digital gain
    to 1, and shutter speed to the minimum (8us for Pi Camera v2)
    NB ISO is left at auto, because this is needed for the gains
    to be set correctly.
    """
    camera.set_controls({
        "AeEnable": False,
        "AnalogueGain": 1,
        "ExposureTime": 1
    })
   # camera.iso = 0  # We must set ISO=0 (auto) or we can't set gain
  #  camera.analog_gain = 1
   # camera.digital_gain = 1 (not configurable)
    # Setting the shutter speed to 1us will result in it being set
    # to the minimum possible, which is probably 8us for PiCamera v2
    #camera.shutter_speed = 1
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
    camera.capture_array("raw") #controls might not be updated for the first frame?
    max_brightness = np.max(get_channel_percentiles(camera, percentile,))
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
        new_time = int(
            test.exposure_time * min(target_white_level / test.level, 8)
        )
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
        camera.controls.AnalogueGain = test.analog_gain * min(target_white_level / test.level, 2)
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
    camera: Picamera2, percentile: float = 99
) -> Tuple[float, float]:
    """Adjust the white balance in a single shot, based on the raw image.

    NB if ``channels_from_raw_image`` is broken, this will go haywire.
    We should probably have better logic to verify the channels really
    are BGGR...
    """
    config = camera.create_still_configuration(raw={"format": "SBGGR10"})
    camera.configure(config)
    camera.start()
    blue, g1, g2, red = get_channel_percentiles(camera, percentile)
    green = (g1 + g2) / 2.0
    new_awb_gains = (green / red, green / blue)
    logging.info(
        f"Raw white point is R: {red} G: {green} B: {blue}, "
        f"setting AWB gains to ({new_awb_gains[0]:.2f}, "
        f"{new_awb_gains[1]:.2f})."
    )
    camera.controls.AwbEnable = False
    camera.controls.ColourGains = new_awb_gains
    m = camera.capture_metadata()
    print(f"Gains are now {m['ColourGains']}")
    time.sleep(0.2)
    print(f"Gains are now {m['ColourGains']}")
    time.sleep(0.2)
    print(f"Gains are now {m['ColourGains']}")
    time.sleep(0.2)
    print(f"Gains are now {m['ColourGains']}")
    return new_awb_gains


def channels_from_bayer_array(bayer_array: np.ndarray) -> np.ndarray:
    """Given the 'array' from a PiBayerArray, return the 4 channels."""
    bayer_pattern: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bayer_array = bayer_array.view(np.uint16)
    channels_shape: Tuple[int, ...] = (
        4,
        bayer_array.shape[0] // 2,
        bayer_array.shape[1] // 2,
    )
    channels: np.ndarray = np.zeros(channels_shape, dtype=bayer_array.dtype)
    for i, offset in enumerate(bayer_pattern):
        # We simplify life by dealing with only one channel at a time.
        channels[i, :, :] = bayer_array[offset[0] :: 2, offset[1] :: 2]

    return channels


def get_channel_percentiles(camera: Picamera2, percentile: float, reconfigure = True) -> np.ndarray:
    """Calculate the brightness percentile of the pixels in each channel

    Camera should be started and configured for raw frames

    This is a number between -64 and 959 for each channel, because the
    camera takes 10-bit images (maximum=1023) and its zero level is set
    at 64 for denoising purposes (there's black level compensation built
    in, and to avoid skewing the noise, the black level is set as 64 to
    leave some room for negative values.
    """

    channels = channels_from_bayer_array(camera.capture_array("raw"))

    return np.percentile(channels, percentile, axis=(1, 2)) - 64


def lst_from_channels(channels: np.ndarray) -> np.ndarray:
    """Given the 4 Bayer colour channels from a white image, generate a LST."""
    full_resolution: np.ndarray = np.array(
        channels.shape[1:]
    ) * 2  # channels have been binned

    # NOTE: the size of the LST is 1/64th of the image, but rounded UP.
    #lst_resolution: List[int] = [(r // 64) + 1 for r in full_resolution]
    lst_resolution = np.array([16, 12])
    ratios = np.ceil(full_resolution/lst_resolution).astype(int) #pixels per section
    logging.info("Generating a lens shading table at %sx%s", *lst_resolution)
    lens_shading: np.ndarray = np.zeros(
        [channels.shape[0]] + list(lst_resolution), dtype=float
    )
    for i in range(lens_shading.shape[0]):
        image_channel: np.ndarray = channels[i, :, :]
        iw: int
        ih: int
        iw, ih = image_channel.shape
        ls_channel: np.ndarray = lens_shading[i, :, :]
        lw: int
        lh: int
        lw, lh = ls_channel.shape
        # The lens shading table is rounded **up** in size to 1/64th of the size of
        # the image.  Rather than handle edge images separately, I'm just going to
        # pad the image by copying edge pixels, so that it is exactly 32 times the
        # size of the lens shading table (NB 32 not 64 because each channel is only
        # half the size of the full image - remember the Bayer pattern...  This
        # should give results very close to 6by9's solution, albeit considerably
        # less computationally efficient!
        padded_image_channel: np.ndarray = np.pad(
            image_channel, [(0, lw * ratios[0] - iw), (0, lh * ratios[1] - ih)], mode="edge"
        )  # Pad image to the right and bottom
        logging.info(
            "Channel shape: %sx%s, shading table shape: %sx%s, after padding %s",
            iw,
            ih,
            lw * ratios[0],
            lh * ratios[1],
            padded_image_channel.shape,
        )
        # Next, fill the shading table (except edge pixels).  Please excuse the
        # for loop - I know it's not fast but this code needn't be!
        box: int = 9  # We average together a square of this side length for each pixel.
        # NB this isn't quite what 6by9's program does - it averages 3 pixels
        # horizontally, but not vertically.
        for dx in np.arange(box) - box // 2:
            for dy in np.arange(box) - box // 2:
                ls_channel[:, :] += (
                    padded_image_channel[ratios[0]//2 + dx :: ratios[0], ratios[1]//2 + dy :: ratios[1]] - 64
                    #TODO why -64?
                )
        ls_channel /= box ** 2
        # The original C code written by 6by9 normalises to the central 64 pixels in each channel.
        # ls_channel /= np.mean(image_channel[iw//2-4:iw//2+4, ih//2-4:ih//2+4])
        # I have had better results just normalising to the maximum:
        ls_channel /= np.max(ls_channel)
        # NB the central pixel should now be *approximately* 1.0 (may not be exactly
        # due to different averaging widths between the normalisation & shading table)
        # For most sensible lenses I'd expect that 1.0 is the maximum value.
        # NB ls_channel should be a "view" of the whole lens shading array, so we don't
        # need to update the big array here.

    # What we actually want to calculate is the gains needed to compensate for the
    # lens shading - that's 1/lens_shading_table_float as we currently have it.
    gains: np.ndarray = 1.0 / lens_shading  # 32 is unity gain
    #gains[gains > 255] = 255  # clip at 255, maximum gain is 255/32
    #gains[gains < 32] = 32  # clip at 32, minimum gain is 1 (is this necessary?)
    #lens_shading_table: np.ndarray = gains.astype(np.uint8)
    set_lst_values(gains, scamera)

    return gains[::-1, :, :].copy()

def set_lst_values(gains: np.ndarray, scamera: PiCamera2Streamer):
    luminance = gains.mean(axis = 0)
    gains = gains/luminance
    green = (gains[1] + gains[2])/2.0
    red = gains[2]/green
    blue = gains[0]/green
    alsc = Picamera2.find_tuning_algo(scamera.tuning, "rpi.alsc")
    alsc["n_iter"] = 0 #disable the adaptive part
    alsc["luminance_strength"] = 1.0
    alsc["calibrations_Cr"] = [{
        "ct": 4500,
        "table": np.reshape(red, (-1)).tolist()
    }]
    alsc["calibrations_Cb"] = [{
        "ct": 4500,
        "table": np.reshape(blue, (-1)).tolist()
    }]

    alsc["luminance_lut"] = np.reshape(luminance, (-1)).tolist()

    scamera.update_tuning()

def lst_from_camera(camera: Picamera2, scamera: PiCamera2Streamer) -> np.ndarray:
    """Acquire a raw image and use it to calculate a lens shading table."""
    if camera.started:
        camera.stop_recording()
    config = camera.create_still_configuration(raw={"format": "SBGGR10"})
    camera.configure(config)
    camera.start()
    raw_image = camera.capture_array("raw")
    camera.stop()
    # Now we need to calculate a lens shading table that would make this flat.
    # raw_image is a 3D array, with full resolution and 3 colour channels.  No
    # de-mosaicing has been done, so 2/3 of the values are zero (3/4 for R and B
    # channels, 1/2 for green because there's twice as many green pixels).
    channels = channels_from_bayer_array(raw_image)
    return lst_from_channels(channels, scamera)


def recalibrate_camera(camera: Picamera2):
    """Reset the lens shading table and exposure settings.

    This method first resets to a flat lens shading table, then auto-exposes,
    then generates a new lens shading table to make the current view uniform.
    It should be run when the camera is looking at a uniform white scene.

    NB the only parameter ``camera`` is a ``PiCamera`` instance and **not** a
    ``StreamingCamera``.
    """
    #TODO
    #camera.lens_shading_table = flat_lens_shading_table(camera)
    #_ = rgb_image(camera)  # for some reason the camera won't work unless I do this!

    #lens_shading_table = lst_from_camera(camera)

    #camera.lens_shading_table = lens_shading_table
    #_ = rgb_image(camera)

    # Fix the AWB gains so the image is neutral
    stop_after = False
    if not camera.started:
        camera.start()
        stop_after = True

    channel_means = np.mean(np.mean(rgb_image(camera), axis=0, dtype=float), axis=0)
    old_gains = camera.controls.ColourGains #TODO: this won't work
    camera.controls.ColourGains = (
        channel_means[1] / channel_means[0] * old_gains[0],
        channel_means[1] / channel_means[2] * old_gains[1],
    )
    time.sleep(1)
    # Ensure the background is bright but not saturated
    adjust_exposure_to_setpoint(camera, 230)
    if stop_after:
        camera.stop()

if __name__ == "__main__":
    with Picamera2() as main_camera:
        main_camera.start_preview()
        time.sleep(3)
        logging.info("Recalibrating...")
        recalibrate_camera(main_camera)
        logging.info("Done.")
        time.sleep(2)
