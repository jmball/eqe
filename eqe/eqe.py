"""Measure external quantum efficiency."""

import logging
import time

import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.integrate
from scipy.constants import h, c, e

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def integrated_jsc(wls, eqe, spec):
    """Calculate integrated Jsc.

    Parameters
    ----------
    wls : array
        Wavelengths in nm.
    eqe : array
        Fractional EQE at each wavelength.
    spec : array
        Spectral irradiance of reference spectrum at each wavelength.
    """
    Es = h * c / (wls * 1e-9)
    return sp.integrate.simps(eqe * spec / Es, wls) * 1000 / 10000


def set_wavelength(mono, wl, grating_change_wls=None, filter_change_wls=None):
    """Set monochromator wavelength.

    Parameters
    ----------
    mono : monochromator object
        Monochromator object.
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
    if filter_change_wls is not None:
        filter_pos = len([i for i in filter_change_wls if i < wl]) + 1
        resp = mono.set_filter(filter_pos)
    resp = mono.goto_wavelength(wl)


def measure(
    lockin,
    mono,
    wl,
    grating_change_wls=None,
    filter_change_wls=None,
    auto_gain=True,
    auto_gain_method="user",
):
    """Go to wavelength and measure data.

    Paremeters
    ----------
    lockin : lock-in amplifier object
        Lock-in amplifier object.
    mono : monochromator object
        Monochromator object.
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
    set_wavelength(mono, wl, grating_change_wls, filter_change_wls)

    if auto_gain is True:
        if auto_gain_method == "instr":
            lockin.auto_gain()
            logger.debug(f"auto_gain()")
        elif auto_gain_method == "user":
            gain_set = False
            while not gain_set:
                sensitivity = lockin.get_sensitivity()
                sensitivity_int = lockin.sensitivities.index(sensitivity)
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

    data1 = list(lockin.measure_multiple([1, 2, 5, 6, 7, 8]))
    data2 = list(lockin.measure_multiple([3, 4, 9, 10, 11]))
    ratio = data2[0] / data1[2]
    data = data1 + data2
    data.insert(len(data), ratio)
    return data


def scan(
    lockin,
    mono,
    psu,
    smu,
    psu_ch1_voltage=0,
    psu_ch1_current=0,
    psu_ch2_voltage=0,
    psu_ch2_current=0,
    psu_ch3_voltage=0,
    psu_ch3_current=0,
    smu_voltage=0,
    calibration=True,
    ref_measurement_path=None,
    ref_measurement_file_header=1,
    ref_eqe_path=None,
    ref_spectrum_path=None,
    start_wl=350,
    end_wl=1100,
    num_points=76,
    repeats=1,
    grating_change_wls=None,
    filter_change_wls=None,
    auto_gain=True,
    auto_gain_method="user",
    data_handler=None,
):
    """Perform a wavelength scan measurement.

    Paremeters
    ----------
    lockin : lock-in amplifier object
        Lock-in amplifier object.
    mono : monochromator object
        Monochromator object.
    psu : psu object
        PSU object.
    smu : smu object
        SMU object.
    psu_ch1_voltage : float, optional
        PSU channel 1 voltage.
    psu_ch1_current : float, optional
        PSU channel 1 current.
    psu_ch2_voltage : float, optional
        PSU channel 2 voltage.
    psu_ch2_current : float, optional
        PSU channel 2 current.
    psu_ch3_voltage : float, optional
        PSU channel 3 voltage.
    psu_ch3_current : float, optional
        PSU channel 3 current.
    calibration : bool, optional
        Whether or not measurement should be interpreted as a calibration run.
    ref_measurement_path : str
        Path of data file containing measurement data of the reference diode. Ignored
        if `calibration` is True.
    ref_measurement_file_header : int
        Number of header lines to skip in referece diode measurement data file. Ignored
        if `calibration` is True.
    ref_eqe_path : str
        Path to EQE calibration data for the reference diode. Ignored if `calibration`
        is True.
    ref_spectrum_path : str, optional
        Path to reference spectrum to use in an integrated Jsc calculation. Ignored if
        `calibration` is True.
    start_wl : int or float, optional
        Start wavelength in nm.
    end_wl : int or float, optional
        End wavelength in nm
    num_points : int, optional
        Number of wavelengths in scan
    repeats : int, optional
        Number of repeat measurements at each wavelength.
    grating_change_wls : list or tuple of int or float, optional
        Wavelength in nm at which to change to the grating.
    filter_change_wls : list or tuple of int or float, optional
        Wavelengths in nm at which to change filters
    auto_gain : bool, optional
        Automatically choose sensitivity.
    auto_gain_method : {"instr", "user"}, optional
        If auto_gain is True, method for automatically finding the correct gain setting.
        "instr" uses the instrument auto-gain feature, "user" implements a user-defined
        algorithm.
    data_handler : data_handler object, optional
        Object that processes live data produced during the scan.
    """
    # reset sensitivity to lowest setting to prevent overflow
    lockin.set_sensitivity(26)

    # get array of wavelengths to measure
    wls, dwl = np.linspace(start_wl, end_wl, num_points, endpoint=True, retstep=True)

    # look up reference data if not running a calibration scan
    if calibration is not True:
        ref_eqe = np.genfromtxt(ref_eqe_path, delimiter="\t", skip_header=1)
        ref_measurement = np.genfromtxt(
            ref_measurement_path,
            delimiter="\t",
            skip_header=ref_measurement_file_header,
        )
        ref_spectrum = np.genfromtxt(ref_spectrum_path, delimeter="\t", skip_header=1)
        # interpolate reference data
        f_ref_eqe = sp.interpolate.interp1d(
            ref_eqe[:, 0], ref_eqe[:, 1], kind="cubic", bounds_error=False, fill_value=0
        )
        f_ref_measurement = sp.interpolate.interp1d(
            ref_measurement[:, 1],
            ref_measurement[:, -1],
            kind="cubic",
            bounds_error=False,
            fill_value=0,
        )
        f_ref_spectrum = sp.interpolate.interp1d(
            ref_spectrum[:, 0],
            ref_spectrum[:, 1],
            kind="cubic",
            bounds_error=False,
            fill_value=0,
        )

    # initialise scan data container
    scan_data = []

    # turn on bias LEDs if required
    if (psu_ch1_current != 0) & (psu_ch1_voltage != 0):
        psu.set_apply(1, psu_ch1_voltage, psu_ch1_current)
        psu.set_output_enable(True, 1)
    if (psu_ch2_current != 0) & (psu_ch2_voltage != 0):
        psu.set_apply(2, psu_ch2_voltage, psu_ch2_current)
        psu.set_output_enable(True, 2)
    if (psu_ch3_current != 0) & (psu_ch3_voltage != 0):
        psu.set_apply(3, psu_ch3_voltage, psu_ch3_current)
        psu.set_output_enable(True, 3)

    # apply voltage bias
    smu.setOutput(smu_voltage)
    smu.outOn(True)

    # scan wavelength and measure lock-in parameters
    meas_wls = []
    meas_eqe = []
    for wl in wls:
        timestamp = time.time()
        data = np.empty((0, 11))
        for i in range(repeats):
            new_data = measure(
                lockin,
                mono,
                wl,
                grating_change_wls,
                filter_change_wls,
                auto_gain,
                auto_gain_method,
            )
            data = np.append(data, np.array(new_data).reshape((1, 11)), axis=0)
        # return average of repeats
        data = data.mean(axis=0).tolist()
        data.insert(0, wl)
        data.insert(0, timestamp)
        if calibration is False:
            # calculate eqe
            ref_eqe_at_wl = f_ref_eqe(wl)
            ref_measurement_at_wl = f_ref_measurement(wl)
            eqe = data[-1] * ref_eqe_at_wl / ref_measurement_at_wl

            # calculate integrated jsc
            meas_wls.append(wl)
            meas_eqe.append(eqe)
            jsc = integrated_jsc(
                np.array(meas_wls), np.array(meas_eqe), f_ref_spectrum(meas_wls)
            )

            # insert calc parameters into data
            data.insert(len(data), eqe)
            data.insert(len(data), jsc)
        scan_data.append(data)
        if data_handler is not None:
            data_handler(data)
        logger.info(f"{data}")

    # turn of smu if present
    smu.outOn(False)

    # turn off LEDs
    psu.set_output_enable(False, 1)
    psu.set_output_enable(False, 2)
    psu.set_output_enable(False, 3)

    return scan_data
