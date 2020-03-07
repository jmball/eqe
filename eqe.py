"""Measure external quantum efficiency"""

import logging
import os
import pathlib
import sys

# insert parent folder containing instrument libraries into path
cwd = pathlib.Path.cwd()
sys.path.insert(0, str(cwd.parent))

# import instrument libraries
import sr830
import sp2150

# set up logger
console_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s|%(name)s|%(levelname)s|%(message)s",
    handlers=[console_handler],
)
logger = logging.getLogger()

# instantiate instrument objects and connect
lockin = sr830.sr830(address="ASRL2::INSTR", output_interface=0, err_check=False)
lockin.connect()
lockin_sn = lockin.serial_number
mono = sp2150(address="ASRL5::INSTR")
mono.connect()
logger.info(
    f"{lockin.manufacturer}, {lockin.model}, {lockin_sn}, {lockin.firmware_version} connected!"
)


def log_monochromator_response(resp):
    """Log monochromator response.
    
    Parameters
    ----------
    resp : str
        monochromator command response
    """
    resp.strip("\r\n")
    logger.debug(f"Monochromator response: {resp}")


def log_lockin_response(resp):
    """Log lock-in amplifier response.
    Parameters
    ----------
    resp : str
        monochromator command response
    """
    logger.debug(
        f"{lockin_sn}, cmd: {resp['cmd']}, resp: {resp['resp']}, fmt_resp: {resp['value']}, warning: {resp['warning']}, error: {resp['error']}
    )


def configure_monochromator(grating=1, scan_speed=1000, wl=550):
    """Configure monochromator settings.

    Parameters
    ----------
    grating : int
        grating number
    scan_speed : int or float
        scan speed in nm/min
    """
    resp = mono.set_grating(grating)
    log_monochromator_response(resp)

    resp = mono.set_scan_speed(scan_speed)
    log_monochromator_response(resp)

    resp = mono.set_wavelength(wl)
    log_monochromator_response(resp)


def configure_lockin(
    input_configuration=0,
    input_coupling=0,
    ground_shielding=1,
    line_notch_filter_status=3,
    ref_source=0,
    detection_harmonic=1,
    ref_trigger=1,
    ref_freq=1000,
    sensitivity=26,
    reserve_mode=1,
    time_constant=8,
    low_pass_filter_slope=1,
    sync_status=0,
    ch1_display=1,
    ch2_display=1,
    ch1_ratio=0,
    ch2_ratio=0,
):
    """Configure lock-in amplifier settings.

    Parameters
    ----------
    

    Returns
    -------
    """
    resp = lockin.set_input_configuration(input_configuration)
    resp = lockin.set_input_coupling(input_coupling)
    resp = lockin.set_input_shield_gnd(ground_shielding)
    resp = lockin.set_line_notch_status(line_notch_filter_status)
    resp = lockin.set_ref_source(ref_source)
    resp = lockin.set_detection_harmonic(detection_harmonic)
    resp = lockin.set_reference_trigger(ref_trigger)
    resp = lockin.set_ref_freq(ref_freq)
    resp = lockin.set_sensitivity(sensitivity)
    resp = lockin.set_reserve(reserve_mode)
    resp = lockin.set_time_constant(time_constant)
    resp = lockin.set_lp_filter_slope(low_pass_filter_slope)
    resp = lockin.set_sync_status(sync_status)
    resp = lockin.set_display(1, ch1_display, ch1_ratio)
    resp = lockin.set_display(2, ch2_display, ch2_ratio)
    

def measure(wl, auto_gain=True, auto_gain_method="user"):
    """Go to wavelength and measure data.
    
    Paremeters
    ----------
    auto_gain_method : {"instr", "user"}
        If auto_gain is True, method for automatically finding the correct gain setting.
        "instr" uses the instrument auto-gain feature, "user" implements a user-defined
        algorithm.

    Returns
    -------
    """
    if auto_gain is True:
        if auto_gain_method == "instr":
            lock_in.auto_gain()
    pass


def scan(start_wl=350, stop_wl=1100, step_wl=10, averages=1):
    """Perform a wavelength scan measurement.
    
    Paremeters
    ----------

    Returns
    -------
    """
    pass
