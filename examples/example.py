"""Measure an external quantum efficiency spectrum and save the data."""

import configparser
import csv
import logging
import pathlib
import sys
import time

# import instrument libraries
import sr830
import sp2150
import dp800

import eqe


class DummySMU:
    """Dummy class for SMU."""

    def __init__(self):
        """Initialise object."""
        pass

    def connect(self):
        """Connect insturment."""
        pass

    def disconnect(self):
        """Disconnect instrument."""
        pass

    def setOutput(self, voltage):
        """Set output voltage."""
        pass

    def outOn(self, on):
        """Turn output on/off."""
        pass


class EQEDataHandler:
    """Handler for processing live EQE data."""

    def __init__(self, device_id, save_folder, calibration, config_header):
        """Initialise object.

        Parameters
        ----------
        device_id : str
            Device identifier used for file name.
        save_folder : Path
            Folder used for saving the measurement data file.
        calibration : bool
            Whether or not measurement should be interpreted as a calibration run.
        config_header : list
            Configuration information to be included as the header in the save file.
        """
        self.device_id = device_id
        self.save_folder = save_folder
        self.calibration = calibration
        self.config_header = config_header

        # Keep count of number of lines of live data written to file or plot.
        # Can be used to initialise an new file or plot.
        self.save_counter = 0
        self.plot_counter = 0

    def save_data(self, data):
        """Save live data to file.

        data : array
            Array of measurement information from a scan point.
        """
        # init file if no data has been saved this scan
        if self.save_counter == 0:
            # generate filename
            i = 0
            if self.calibration is True:
                self.save_path = self.save_folder.joinpath(f"reference_{i}.txt")
                while self.save_path.exists():
                    i += 1
                    self.save_path = self.save_folder.joinpath(f"reference_{i}.txt")
            else:
                self.save_path = self.save_folder.joinpath(f"{self.device_id}_{i}.txt")
                while self.save_path.exists():
                    i += 1
                    self.save_path = self.save_folder.joinpath(
                        f"{self.device_id}_{i}.txt"
                    )

            # write config info to data file
            with open(self.save_path, "w", newline="\n") as f:
                f.writelines(self.config_header)
                f.writelines("\n")

            # write column headers
            with open(self.save_path, "a", newline="\n") as f:
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
                if self.calibration is False:
                    header.insert(len(header), "EQE")
                writer.writerow(header)

        with open(self.save_path, "a", newline="\n") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(data)

        self.save_counter += 1

    def plot_data(self, data):
        """Plot live data."""
        pass

    def handle_data(self, data):
        """Process live data."""
        self.save_data(data)
        self.plot_data(data)


# # insert parent folder containing instrument libraries into path
# cwd = pathlib.Path.cwd()
# logger.debug(f"cwd: {cwd}")
# parent = cwd.parent
# logger.debug(f"parent: {parent}")
# sr830_path = parent.joinpath("SRS_SR830")
# if sr830_path.exists() is True:
#     logger.debug(f"{sr830_path} exists")
# else:
#     logger.debug(f"{sr830_path} doesn't exist")
# sp2150_path = parent.joinpath("ActonSP2150")
# if sp2150_path.exists() is True:
#     logger.debug(f"{sp2150_path} exists")
# else:
#     logger.debug(f"{sp2150_path} doesn't exist")
# dp800_path = parent.joinpath("rigol_dp800")
# if dp800_path.exists() is True:
#     logger.debug(f"{dp800_path} exists")
# else:
#     logger.debug(f"{dp800_path} doesn't exist")
# sys.path.insert(0, str(sr830_path))
# sys.path.insert(0, str(sp2150_path))
# sys.path.insert(0, str(dp800_path))


# set up logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

file_handler = logging.FileHandler(f"C:/eqe/logs/{int(time.time())}.log")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s|%(name)s|%(levelname)s|%(message)s",
    handlers=[console_handler, file_handler],
)

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
repeats = int(config["experiment"]["repeats"])

logger.debug(f"Calibrate: {calibrate}")
logger.debug(f"Device ID: {device_id}")
logger.debug(f"Start WL: {start_wl}")
logger.debug(f"End WL: {end_wl}")
logger.debug(f"Number of wl points: {num_points}")
logger.debug(f"Repeats: {repeats}")

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

# psu
psu_address = config["psu"]["address"]
psu_ch1_current = config["psu"]["ch1_current"]
psu_ch1_voltage = config["psu"]["ch1_voltage"]
psu_ch2_current = config["psu"]["ch2_current"]
psu_ch2_voltage = config["psu"]["ch2_voltage"]
psu_ch3_current = config["psu"]["ch3_current"]
psu_ch3_voltage = config["psu"]["ch3_voltage"]

logger.debug(f"PSU address: {psu_address}")
logger.debug(f"PSU CH1 current: {psu_ch1_current}")
logger.debug(f"PSU CH1 voltage: {psu_ch1_voltage}")
logger.debug(f"PSU CH2 current: {psu_ch2_current}")
logger.debug(f"PSU CH2 voltage: {psu_ch2_voltage}")
logger.debug(f"PSU CH3 current: {psu_ch3_current}")
logger.debug(f"PSU CH3 voltage: {psu_ch3_voltage}")

# smu
smu_address = config["smu"]["address"]
smu_voltage = config["smu"]["voltage"]

logger.debug(f"SMU address: {smu_address}")
logger.debug(f"SMU voltage: {smu_voltage}")

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
mono.set_scan_speed(mono_scan_speed)

psu = dp800.dp800(check_errors=True)
psu.connect(resource_name=psu_address)

smu = DummySMU()

# set up data_handler
with open("config.ini") as f:
    config_header = f.readlines()
config_header_len = len(config_header)

data_handler = EQEDataHandler(device_id, save_folder, calibrate, config_header)

# run scan
eqe.scan(
    lockin,
    mono,
    psu,
    smu,
    psu_ch1_voltage,
    psu_ch1_current,
    psu_ch2_voltage,
    psu_ch2_current,
    psu_ch3_voltage,
    psu_ch3_current,
    smu_voltage,
    calibrate,
    ref_measurement_path=save_folder.joinpath(ref_measurement_name),
    ref_measurement_file_header=config_header_len,
    ref_eqe_path=ref_eqe_path,
    ref_spectrum_path=ref_spectrum_path,
    start_wl=start_wl,
    end_wl=end_wl,
    num_points=num_points,
    repeats=repeats,
    grating_change_wls=mono_grating_change_wls,
    filter_change_wls=mono_filter_change_wls,
    auto_gain=lia_auto_gain,
    auto_gain_method=lia_auto_gain_method,
    data_handler=data_handler.handle_data,
)

# disconnect instruments
lockin.disconnect()
mono.disconnect()
psu.disconnect()
smu.disconnect()
sr830.rm.close()

# close log file
file_handler.close()
