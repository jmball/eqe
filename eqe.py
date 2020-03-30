"""Measure external quantum efficiency"""

import csv
import configparser
import logging
import pathlib
import sys
import time

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

# set up logger
file_handler = logging.FileHandler(f"C:/eqe/logs/{int(time.time())}.log")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s|%(name)s|%(levelname)s|%(message)s",
    handlers=[console_handler, file_handler],
)
logger = logging.getLogger()


# insert parent folder containing instrument libraries into path
cwd = pathlib.Path.cwd()
logger.debug(f"cwd: {cwd}")
parent = cwd.parent
logger.debug(f"parent: {parent}")
sr830_path = parent.joinpath("SRS_SR830")
if sr830_path.exists() is True:
    logger.debug(f"{sr830_path} exists")
else:
    logger.debug(f"{sr830_path} doesn't exist")
sp2150_path = parent.joinpath("ActonSP2150")
if sp2150_path.exists() is True:
    logger.debug(f"{sp2150_path} exists")
else:
    logger.debug(f"{sp2150_path} doesn't exist")
sys.path.insert(0, str(sr830_path))
sys.path.insert(0, str(sp2150_path))

# import instrument libraries
import sr830
import sp2150


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
            logger.debug(f"auto_gain()")
        elif auto_gain_method == "user":
            gain_set = False
            while not gain_set:
                sensitivity = lockin.get_sensitivity()
                sensitivity_int = sensitivities.index(sensitivity)
                time.sleep(5 * lockin.get_time_constant())
                R = lockin.measure(3)
                log_lockin_response(f"measure(3), {R}")
                if (R >= sensitivity * 0.9) and (sensitivity_int < 26):
                    new_sensitivity = sensitivity_int + 1
                elif (R <= 0.1 * sensitivity) and (sensitivity_int > 0):
                    new_sensitivity = sensitivity_int - 1
                else:
                    new_sensitivity = sensitivity_int
                    gain_set = True
                lockin.set_sensitivity(new_sensitivity)
                log_lockin_response(f"set_sensitivity({new_sensitivity})")
        else:
            msg = f'Invalid auto-gain method: {auto_gain_method}. Must be "instr" or "user".'
            logger.error(msg)
            raise ValueError(msg)

    # wait to settle
    time.sleep(5 * lockin.get_time_constant())

    data1 = list(lockin.measure_multiple([1, 2, 5, 6, 7, 8]))
    data2 = list(lockin.measure_multiple([3, 4, 9, 10, 11]))
    ratio = data2[0] / data1[2]
    data = data1 + data2
    data.insert(len(data), ratio)
    logger.debug(f"Measure: {data}")
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
    averages=1,
    grating_change_wls=None,
    filter_change_wls=None,
    auto_gain=True,
    auto_gain_method="user",
):
    """Perform a wavelength scan measurement.

    Paremeters
    ----------
    save_folder : Path
        Folder used for saving the measurement data file.
    calibration : bool
        Whether or not measurement should be interpreted as a calibration run.
    device_id : str
        Device identifier used for file name.
    ref_measurement_name : str
        Name of data file containing measurement data of the reference diode.
    ref_eqe_path : str
        Path to EQE calibration data for the reference diode.
    ref_spectrum_path : str
        Path to reference spectrum to use in an integrated Jsc calculation.
    start_wl : int or float
        Start wavelength in nm.
    end_wl : int or float
        End wavelength in nm
    num_points : int
        Number of wavelengths in scan
    averages : int
        Number of repeat measurements at each wavelength.
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
    lockin.set_sensitivity(26)
    log_lockin_response(f"set_sensitivity(26)")

    wls, dwl = np.linspace(start_wl, end_wl, num_points, endpoint=True, retstep=True)
    logger.debug(f"Wavelengths: {wls}")
    logger.debug(f"Wavelength step: {dwl}")

    # get config info for data files
    with open("config.ini") as f:
        config_header = f.readlines()
    config_header_len = len(config_header)

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
            save_folder.joinpath(ref_measurement_name),
            delimiter="\t",
            skip_header=config_header_len + 2,
        )
        # interpolate reference eqe spectrum and measurement data
        f_ref_eqe_spectrum = sp.interpolate.interp1d(
            ref_eqe[:, 0], ref_eqe[:, 1], kind="cubic", bounds_error=False, fill_value=0
        )
        f_ref_measurement = sp.interpolate.interp1d(
            ref_measurement[:, 1],
            ref_measurement[:, -1],
            kind="cubic",
            bounds_error=False,
            fill_value=0,
        )

    # write config info to data file
    with open(save_path, "w", newline="\n") as f:
        f.writelines(config_header)
        f.writelines("\n")

    # scan through wavelengths and append data to file
    with open(save_path, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "timestamp (s)",
            "wavelength (nm)",
            "X (V)",
            "Y (V)",
            "Aux In 1 (V)",
            "Aux In 2 (V)",
            "Aux In 3 (V)",
            "Aux In 4 (V)",
            "R (V)",
            "Phase (deg)",
            "Freq (Hz)",
            "Ch1 display",
            "Ch2 display",
            "R/Aux In 1",
        ]
        if calibrate is False:
            header.insert(len(header), "EQE")
        writer.writerow(header)
        for wl in wls:
            for i in range(averages):
                timestamp = time.time()
                data = list(
                    measure(
                        wl,
                        grating_change_wls,
                        filter_change_wls,
                        auto_gain,
                        auto_gain_method,
                    )
                )
                data.insert(0, wl)
                data.insert(0, timestamp)
                if calibrate is False:
                    ref_eqe_at_wl = f_ref_eqe_spectrum(wl)
                    ref_measurement_at_wl = f_ref_measurement(wl)
                    eqe = data[-1] * ref_eqe_at_wl / ref_measurement_at_wl
                    data.insert(len(data), eqe)
                writer.writerow(data)
                logger.info(f"Data: {data}")


# load configuration info
config = configparser.ConfigParser()
config.read("config.ini")

# paths
save_folder = pathlib.Path(config["paths"]["save_folder"])
if save_folder.exists() is False:
    save_folder.mkdir(parents=True)
ref_measurement_name = config["paths"]["ref_measurement_name"]
ref_eqe_path = pathlib.Path(config["paths"]["ref_eqe_path"])
ref_spectrum_path = pathlib.Path(config["paths"]["ref_spectrum_path"])
log_folder = pathlib.Path(config["paths"]["log_folder"])
if log_folder.exists() is False:
    log_folder.mkdir(parents=True)

logger.debug(f"Save folder: {save_folder}")
logger.debug(f"Ref measurement name: {ref_measurement_name}")
logger.debug(f"Ref EQE path: {ref_eqe_path}")
logger.debug(f"Ref spectrum path: {ref_spectrum_path}")

# experiment
calibrate = config["experiment"]["calibrate"]
if calibrate == "True":
    calibrate = True
elif calibrate == "False":
    calibrate = False
else:
    raise ValueError(
        f"Invalid value for calibrate: '{calibrate}'. Must be either 'True' or 'False'."
    )
device_id = config["experiment"]["device_id"]
start_wl = float(config["experiment"]["start_wl"])
end_wl = float(config["experiment"]["end_wl"])
num_points = int(config["experiment"]["num_points"])
averages = int(config["experiment"]["averages"])

logger.debug(f"Calibrate: {calibrate}")
logger.debug(f"Device ID: {device_id}")
logger.debug(f"Start WL: {start_wl}")
logger.debug(f"End WL: {end_wl}")
logger.debug(f"Number of wl points: {num_points}")
logger.debug(f"Averages: {averages}")

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
lia_auto_gain = config["lia"]["auto_gain"]
if (lia_auto_gain == "True") or (lia_auto_gain == "False"):
    lia_auto_gain = bool(lia_auto_gain)
else:
    raise ValueError(
        f"Invalid value for lia_auto_gain: '{lia_auto_gain}'. Must be either 'True' or 'False'."
    )
lia_auto_gain_method = config["lia"]["auto_gain_method"]

logger.debug(f"LIA address: {lia_address}")
logger.debug(f"LIA output interface: {lia_output_interface}")
logger.debug(f"LIA input configuration: {lia_input_configuration}")
logger.debug(f"LIA input coupling: {lia_input_coupling}")
logger.debug(f"LIA ground shielding: {lia_ground_shielding}")
logger.debug(f"LIA line notch filter status: {lia_line_notch_filter_status}")
logger.debug(f"LIA ref source: {lia_ref_source}")
logger.debug(f"LIA detection harmonic: {lia_detection_harmonic}")
logger.debug(f"LIA ref trigger: {lia_ref_trigger}")
logger.debug(f"LIA ref freq: {lia_ref_freq}")
logger.debug(f"LIA sensitivity: {lia_sensitivity}")
logger.debug(f"LIA reserve mode: {lia_reserve_mode}")
logger.debug(f"LIA time constant: {lia_time_constant}")
logger.debug(f"LIA low pass filter slope: {lia_low_pass_filter_slope}")
logger.debug(f"LIA sync status: {lia_sync_status}")
logger.debug(f"LIA ch1 display: {lia_ch1_display}")
logger.debug(f"LIA ch2 display: {lia_ch2_display}")
logger.debug(f"LIA ch1 ratio: {lia_ch1_ratio}")
logger.debug(f"LIA ch2 ratio: {lia_ch2_ratio}")
logger.debug(f"LIA auto gain: {lia_auto_gain}")
logger.debug(f"LIA auto gain method: {lia_auto_gain_method}")

# monochromator
mono_address = config["monochromator"]["address"]
mono_grating_change_wls = config["monochromator"]["grating_change_wls"]
mono_grating_change_wls = mono_grating_change_wls.split(",")
mono_grating_change_wls = [int(x) for x in mono_grating_change_wls]
mono_filter_change_wls = config["monochromator"]["filter_change_wls"]
mono_filter_change_wls = mono_filter_change_wls.split(",")
mono_filter_change_wls = [int(x) for x in mono_filter_change_wls]
mono_scan_speed = int(config["monochromator"]["scan_speed"])

logger.debug(f"Monochromator address: {mono_address}")
logger.debug(f"Monochromator grating change wls: {mono_grating_change_wls}")
logger.debug(f"Monochromator filter change wls: {mono_filter_change_wls}")
logger.debug(f"Monochromator scan speed: {mono_scan_speed}")

# instantiate instrument objects and connect
lockin = sr830.sr830(return_int=True, check_errors=True)
lockin.connect(
    resource_name=lia_address,
    output_interface=lia_output_interface,
    set_default_configuration=False,
)
lockin.set_configuration(
    input_configuration=lia_input_configuration,
    input_coupling=lia_input_coupling,
    ground_shielding=lia_ground_shielding,
    line_notch_filter_status=lia_line_notch_filter_status,
    ref_source=lia_ref_source,
    detection_harmonic=lia_detection_harmonic,
    ref_trigger=lia_ref_trigger,
    ref_freq=lia_ref_freq,
    sensitivity=lia_sensitivity,
    reserve_mode=lia_reserve_mode,
    time_constant=lia_time_constant,
    low_pass_filter_slope=lia_low_pass_filter_slope,
    sync_status=lia_sync_status,
    ch1_display=lia_ch1_display,
    ch2_display=lia_ch2_display,
    ch1_ratio=lia_ch1_ratio,
    ch2_ratio=lia_ch2_ratio,
)
lockin_sn = lockin.serial_number
sensitivities = lockin.sensitivities
time_constants = lockin.time_constants
mono = sp2150.sp2150(address=mono_address)
mono.connect()
logger.info(
    f"{lockin.manufacturer}, {lockin.model}, {lockin_sn}, {lockin.firmware_version} connected!"
)

# configure instruments
configure_monochromator(scan_speed=mono_scan_speed)

# run scan
scan(
    save_folder=save_folder,
    calibration=calibrate,
    device_id=device_id,
    ref_measurement_name=ref_measurement_name,
    ref_eqe_path=ref_eqe_path,
    ref_spectrum_path=ref_spectrum_path,
    start_wl=start_wl,
    end_wl=end_wl,
    num_points=num_points,
    grating_change_wls=mono_grating_change_wls,
    filter_change_wls=mono_filter_change_wls,
    auto_gain=lia_auto_gain,
    auto_gain_method=lia_auto_gain_method,
)

# disconnect instruments
lockin.disconnect()
mono.disconnect()
sr830.rm.close()

# close log file

file_handler.close()
