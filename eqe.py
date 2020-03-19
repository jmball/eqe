"""Measure external quantum efficiency"""

import csv
import configparser
import logging
import os
import pathlib
import sys
import time

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

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
    logger.debug(f"{lockin_sn}, {resp}")


def configure_monochromator(scan_speed=1000):
    """Configure monochromator settings.

    Parameters
    ----------
    scan_speed : int or float
        scan speed in nm/min
    """
    resp = mono.set_scan_speed(scan_speed)
    log_monochromator_response(resp)


def set_wavelength(wl, grating_change_wls=None, filter_change_wls=None):
    """Set monochromator wavelength.

    Parameters
    ----------
    wl : int or float
        Wavelength in nm.
    grating_change_wls : list or tuple of int or float
        Wavelength in nm at which to change to the grating.
    filter_change_wls : list or tuple of int or float
        Wavelengths in nm at which to change filters
    """
    # Set grating and filter wheel position depending on wl
    if grating_change_wls is not None:
        grating = len([i for i in grating_change_wls if i < wl]) + 1
        resp = mono.set_grating(grating)
        log_monochromator_response(resp)
    if filter_change_wls is not None:
        filter_pos = len([i for i in filter_change_wls if i < wl]) + 1
        resp = mono.set_filter(filter_pos)
        log_monochromator_response(resp)
    resp = mono.goto_wavelength(wl)
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
    input_configuration : {0, 1, 2, 3}
        Input configuration:

            * 0 : A
            * 1 : A-B
            * 2 : I (1 MOhm)
            * 3 : I (100 MOhm)
    input_coupling : {0, 1}
        Input coupling:

            * 0 : AC
            * 1 : DC
    ground_shielding : {0, 1}
        Input shield grounding:

            * 0 : Float
            * 1 : Ground
    line_notch_filter_status : {0, 1, 2, 3}
        Input line notch filter status:

            * 0 : no filters
            * 1 : Line notch in
            * 2 : 2 x Line notch in
            * 3 : Both notch filters in
    ref_source : {0, 1}
        Refernce source:

            * 0 : external
            * 1 : internal
    detection_harmonic : int
        Detection harmonic, 1 =< harmonic =< 19999.
    ref_trigger : {0, 1, 2}
        Trigger type:

            * 0: zero crossing
            * 1: TTL rising egde
            * 2: TTL falling edge
    ref_freq : float
        Frequency in Hz, 0.001 =< freq =< 102000.
    sensitivity : {0 - 26}
        Sensitivity in V/uA:

            * 0 : 2e-9
            * 1 : 5e-9
            * 2 : 10e-9
            * 3 : 20e-9
            * 4 : 50e-9
            * 5 : 100e-9
            * 6 : 200e-9
            * 7 : 500e-9
            * 8 : 1e-6
            * 9 : 2e-6
            * 10 : 5e-6
            * 11 : 10e-6
            * 12 : 20e-6
            * 13 : 50e-6
            * 14 : 100e-6
            * 15 : 200e-6
            * 16 : 500e-6
            * 17 : 1e-3
            * 18 : 2e-3
            * 19 : 5e-3
            * 20 : 10e-3
            * 21 : 20e-3
            * 22 : 50e-3
            * 23 : 100e-3
            * 24 : 200e-3
            * 25 : 500e-3
            * 26 : 1
    reserve_mode : {0, 1, 2}
        Reserve mode:

            * 0 : High reserve
            * 1 : Normal
            * 2 : Low noise
    time_constant : {0 - 19}
        Time constant in s:

            * 0 : 10e-6
            * 1 : 30e-6
            * 2 : 100e-6
            * 3 : 300e-6
            * 4 : 1e-3
            * 5 : 3e-3
            * 6 : 10e-3
            * 7 : 30e-3
            * 8 : 100e-3
            * 9 : 300e-3
            * 10 : 1
            * 11 : 3
            * 12 : 10
            * 13 : 30
            * 14 : 100
            * 15 : 300
            * 16 : 1e3
            * 17 : 3e3
            * 18 : 10e3
            * 19 : 30e3
    low_pass_filter_slope : {0, 1, 2, 3}
        Low pass filter slope in dB/oct:

            * 0 : 6
            * 1 : 12
            * 2 : 18
            * 3 : 24
    sync_status : {0, 1}
        Synchronous filter status:

            * 0 : Off
            * 1 : below 200 Hz
    ch1_display : {0, 1, 2, 3, 4}
        Display parameter for CH1:

            * 0 : X
            * 1 : R
            * 2 : X Noise
            * 3 : Aux In 1
            * 4 : Aux In 2
    ch2_display : {0, 1, 2, 3, 4}
        Display parameter for CH2:

            * 0 : Y
            * 1 : Phase
            * 2 : Y Noise
            * 3 : Aux In 3
            * 4 : Aux In 4
    ch1_ratio : {0, 1, 2}
        Ratio type for CH1:

            * 0 : none
            * 1 : Aux In 1
            * 2 : Aux In 2
    ch2_ratio : {0, 1, 2}
        Ratio type for CH1:

            * 0 : none
            * 1 : Aux In 3
            * 2 : Aux In 4

    Returns
    -------
    """
    lockin.set_input_configuration(input_configuration)
    log_lockin_response(f"set_input_configuration({input_configuration})")

    lockin.set_input_coupling(input_coupling)
    log_lockin_response(f"set_input_coupling({input_coupling})")

    lockin.set_input_shield_gnd(ground_shielding)
    log_lockin_response(f"set_input_shield_gnd({ground_shielding})")

    lockin.set_line_notch_status(line_notch_filter_status)
    log_lockin_response(f"set_line_notch_status({line_notch_filter_status})")

    lockin.set_ref_source(ref_source)
    log_lockin_response(f"set_ref_source({ref_source})")

    lockin.set_detection_harmonic(detection_harmonic)
    log_lockin_response(f"set_detection_harmonic({detection_harmonic})")

    lockin.set_reference_trigger(ref_trigger)
    log_lockin_response(f"set_reference_trigger({ref_trigger})")

    lockin.set_ref_freq(ref_freq)
    log_lockin_response(f"set_ref_freq({ref_freq})")

    lockin.set_sensitivity(sensitivity)
    log_lockin_response(f"set_sensitivity({sensitivity})")

    lockin.set_reserve(reserve_mode)
    log_lockin_response(f"set_reserve({reserve_mode})")

    lockin.set_time_constant(time_constant)
    log_lockin_response(f"set_time_constant({time_constant})")

    lockin.set_lp_filter_slope(low_pass_filter_slope)
    log_lockin_response(f"set_lp_filter_slope({low_pass_filter_slope})")

    lockin.set_sync_status(sync_status)
    log_lockin_response(f"set_sync_status({sync_status})")

    lockin.set_display(1, ch1_display, ch1_ratio)
    log_lockin_response(f"set_display(1, {ch1_display}, {ch1_ratio})")

    lockin.set_display(2, ch2_display, ch2_ratio)
    log_lockin_response(f"set_display(2, {ch2_display}, {ch2_ratio})")


def measure(
    wl,
    grating_change_wls=None,
    filter_change_wls=None,
    auto_gain=True,
    auto_gain_method="user",
):
    """Go to wavelength and measure data.

    Paremeters
    ----------
    wl : int or float
        Wavelength in nm.
    grating_change_wls : list or tuple of int or float
        Wavelength in nm at which to change to the grating.
    filter_change_wls : list or tuple of int or float
        Wavelengths in nm at which to change filters
    auto_gain : bool
        Automatically choose sensitivity. 
    auto_gain_method : {"instr", "user"}
        If auto_gain is True, method for automatically finding the correct gain setting.
        "instr" uses the instrument auto-gain feature, "user" implements a user-defined
        algorithm.

    Returns
    -------
    data : tuple
        X, Y, Aux1, Aux2, Aux3, Aux4, R, Phase, Freq, Ch1, and Ch2.
    """
    set_wavelength(wl, grating_change_wls, filter_change_wls)

    if auto_gain is True:
        if auto_gain_method == "instr":
            lockin.auto_gain()
        elif auto_gain_method == "user":
            gain_set = False
            while not gain_set:
                sensitivity = lockin.get_sensitivity()
                sensitivity_int = sensitivities.index(sensitivity)
                time.sleep(5 * lockin.get_time_constant())
                R = lockin.measure(3)
                if (R >= sensitivity * 0.9) and (sensitivity_int < 26):
                    new_sensitivity = sensitivity_int + 1
                elif (R <= 0.1 * sensitivity) and (sensitivity_int > 0):
                    new_sensitivity = sensitivity_int - 1
                else:
                    new_sensitivity = sensitivity_int
                    gain_set = True
                lockin.set_sensitivity(new_sensitivity)
        else:
            msg = f'Invalid auto-gain method: {auto_gain_method}. Must be "instr" or "user".'
            logger.error(msg)
            raise ValueError(msg)

    # wait to settle
    time.sleep(5 * lockin.get_time_constant())

    data1 = lockin.measure_multiple([1, 2, 5, 6, 7, 8])
    data2 = lockin.measure_multiple([3, 4, 9, 10, 11])
    ratio = data2[0] / data1[2]
    data = list(data1 + data2)
    data.insert(len(data), ratio)
    return data


def scan(
    save_folder,
    calibration,
    device_id=None,
    ref_measurement_name=None,
    ref_eqe_path=None,
    ref_spectrum_path=None,
    start_wl=350,
    end_wl=1100,
    num_points=76,
    grating_change_wls=None,
    filter_change_wls=None,
    auto_gain=True,
    auto_gain_method="user",
):
    """Perform a wavelength scan measurement.

    Paremeters
    ----------
    start_wl : int or float
        Start wavelength in nm.
    end_wl : int or float
        End wavelength in nm
    num_points : int
        Number of wavelengths in scan
    grating_change_wls : list or tuple of int or float
        Wavelength in nm at which to change to the grating.
    filter_change_wls : list or tuple of int or float
        Wavelengths in nm at which to change filters
    auto_gain : bool
        Automatically choose sensitivity. 
    auto_gain_method : {"instr", "user"}
        If auto_gain is True, method for automatically finding the correct gain setting.
        "instr" uses the instrument auto-gain feature, "user" implements a user-defined
        algorithm.
    """
    resp = lockin.set_sensitivity(0)
    log_lockin_response(resp)

    wls, dwl = np.linspace(start_wl, end_wl, num_points, endpoint=True, ret_step=True)

    i = 0
    if calibrate is True:
        save_path = save_folder.joinpath(f"reference_{i}.txt")
        while save_path.exists():
            i += 1
            save_path = save_folder.joinpath(f"reference_{i}.txt")
    else:
        save_path = save_folder.joinpath(f"{device_id}_{i}.txt")
        while save_path.exists():
            i += 1
            save_path = save_folder.joinpath(f"{device_id}_{i}.txt")
        ref_eqe = np.genfromtxt(ref_eqe_path, delimiter="\t", skip_header=1)
        ref_measurement = np.genfromtxt(
            save_folder.joinpath(ref_measurement_name), delimiter="\t", skip_header=1
        )
        # interpolate reference eqe spectrum and measurement data
        f_ref_eqe_spectrum = sp.interpolate.interp1d(
            ref_eqe[:, 0], ref_eqe[:, 1], kind="cubic", bounds_error=False, fill_value=0
        )
        f_ref_measurement = sp.interpolate.interp1d(
            ref_measurement[:, 0],
            ref_measurement[:, -1],
            kind="cubic",
            bounds_error=False,
            fill_value=0,
        )

    with open(save_path, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for wl in wls:
            data = list(
                measure(wl, grating_change_wls, filter_change_wls, True, "user")
            )
            data.insert(0, wl)
            if calibrate is not True:
                ref_eqe_at_wl = f_ref_eqe_spectrum(wl)
                ref_measurement_at_wl = f_ref_measurement(wl)
                eqe = data[-1] * ref_eqe_at_wl / ref_measurement_at_wl
                data.insert(len(data), eqe)
            writer.writerow(data)
            print(data)


# load configuration info
config = configparser.ConfigParser()
config.read("config.ini")

# paths
save_folder = pathlib.Path(config["paths"]["save_folder"])
ref_measurement_name = config["paths"]["ref_measurement_name"]
ref_eqe_path = pathlib.Path(config["paths"]["ref_eqe_path"])
ref_spectrum_path = pathlib.Path(config["paths"]["ref_spectrum_path"])

# experiment
calibrate = config["experiment"]["calibrate"]
if calibrate == "True" or calibrate == "False":
    calibrate = bool(calibrate)
else:
    raise ValueError(
        f"Invalid value for calibrate: '{calibrate}'. Must be either 'True' or 'False'."
    )
device_id = config["experiment"]["device_id"]
start_wl = float(config["experiment"]["start_wl"])
end_wl = float(config["experiment"]["end_wl"])
num_points = int(config["experiment"]["num_points"])

# lock-in amplifier
lia_address = config["lia"]["address"]
lia_output_interface = int(config["lia"]["output_interface"])
lia_input_configuration = int(config["lia"]["input_configuration"])
lia_input_coupling = int(config["lia"]["input_coupling"])
lia_ground_shielding = int(config["lia"]["ground_shielding"])
lia_line_notch_filter_status = int(config["lia"]["line_notch_filter_status"])
lia_ref_source = int(config["lia"]["ref_source"])
lia_detection_harmonic = int(config["lia"]["detection_harmonic"])
lia_ref_trigger = int(config["lia"]["ref_trigger"])
lia_ref_freq = int(config["lia"]["ref_freq"])
lia_sensitivity = int(config["lia"]["sensitivity"])
lia_reserve_mode = int(config["lia"]["reserve_mode"])
lia_time_constant = int(config["lia"]["time_constant"])
lia_low_pass_filter_slope = int(config["lia"]["low_pass_filter_slope"])
lia_sync_status = int(config["lia"]["sync_status"])
lia_ch1_display = int(config["lia"]["ch2_display"])
lia_ch2_display = int(config["lia"]["ch2_display"])
lia_ch1_ratio = int(config["lia"]["ch1_ratio"])
lia_ch2_ratio = int(config["lia"]["ch2_ratio"])
lia_auto_gain = int(config["lia"]["auto_gain"])
lia_auto_gain_method = config["lia"]["auto_gain_method"]

# monochromator
mono_address = config["monochromator"]["address"]
mono_grating_change_wls = config["monochromator"]["grating_change_wls"]
mono_filter_change_wls = config["monochromator"]["filter_change_wls"]
mono_scan_speed = int(config["monochromator"]["scan_speed"])

# instantiate instrument objects and connect
lockin = sr830.sr830(address=lia_address, output_interface=0, err_check=False)
lockin.connect()
lockin_sn = lockin.serial_number
sensitivities = lockin.sensitivities
time_constants = lockin.time_constants
mono = sp2150.sp2150(address=mono_address)
mono.connect()
logger.info(
    f"{lockin.manufacturer}, {lockin.model}, {lockin_sn}, {lockin.firmware_version} connected!"
)

# run scan
scan(
    start_wl,
    end_wl,
    num_points,
    mono_grating_change_wls,
    mono_filter_change_wls,
    lia_auto_gain,
    lia_auto_gain_method,
)

# disconnect instruments
lockin.disconnect()
mono.disconnect()
