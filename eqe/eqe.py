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
    data : list
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
                sensitivity_int = lockin.get_sensitivity()
                sensitivity = lockin.sensitivities[sensitivity_int]
                time_constant_int = lockin.get_time_constant()
                time_constant = lockin.time_constants[time_constant_int]
                time.sleep(5 * time_constant)
                print(
                    f"Sentivity_int = {sensitivity_int}, sensitivity = {sensitivity}, time_constant_int = {time_constant_int}, time_constant = {time_constant}"
                )
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

    return data1 + data2


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
    start_wl=350,
    end_wl=1100,
    num_points=76,
    grating_change_wls=None,
    filter_change_wls=None,
    integration_time=8,
    auto_gain=True,
    auto_gain_method="user",
    handler=None,
    handler_kwargs={},
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
    start_wl : int or float, optional
        Start wavelength in nm.
    end_wl : int or float, optional
        End wavelength in nm
    num_points : int, optional
        Number of wavelengths in scan
    grating_change_wls : list or tuple of int or float, optional
        Wavelength in nm at which to change to the grating.
    filter_change_wls : list or tuple of int or float, optional
        Wavelengths in nm at which to change filters
    integration_time : int
        Integration time setting for the lock-in amplifier.
    auto_gain : bool, optional
        Automatically choose sensitivity.
    auto_gain_method : {"instr", "user"}, optional
        If auto_gain is True, method for automatically finding the correct gain setting.
        "instr" uses the instrument auto-gain feature, "user" implements a user-defined
        algorithm.
    handler : data_handler object, optional
        Object that processes live data produced during the scan.
    handler_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the handler.
    """
    # set lock-in integration time/time constant
    lockin.set_time_constant(integration_time)

    # reset sensitivity to lowest setting to prevent overflow
    lockin.set_sensitivity(26)

    # get array of wavelengths to measure
    wls = np.linspace(start_wl, end_wl, num_points, endpoint=True)

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
    smu.setSource(smu_voltage)
    smu.outOn(True)

    # scan wavelength and measure lock-in parameters
    scan_data = []
    for wl in wls:
        # get timestamp
        timestamp = time.time()

        # perform measurement
        data = measure(
            lockin,
            mono,
            wl,
            grating_change_wls,
            filter_change_wls,
            auto_gain,
            auto_gain_method,
        )

        data.insert(0, wl)
        data.insert(0, timestamp)

        scan_data.append(data)

        print(data)

        if handler is not None:
            handler(data, **handler_kwargs)
        logger.info(f"{data}")

    # turn of smu if present
    smu.outOn(False)

    # turn off LEDs
    psu.set_output_enable(False, 1)
    psu.set_output_enable(False, 2)
    psu.set_output_enable(False, 3)

    return scan_data
