"""Measure external quantum efficiency."""

import math
import logging
import time
import warnings

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
        mono.grating = grating
    if filter_change_wls is not None:
        filter_pos = len([i for i in filter_change_wls if i < wl]) + 1
        mono.filter = filter_pos
    mono.wavelength = wl


def wait_for_lia_to_settle(lockin, timeout):
    """Wait for lock-in amplifier to settle.

    Parameters
    ----------
    lockin : lock-in amplifier object
        Lock-in amplifier object.
    timeout : float
        Maximum time to wait for lock-in to settle before moving on.

    Returns
    -------
    R : float
        Mean sampled R value after settling. Taking the mean of a sample reduces
        influence of noise.
    """
    lockin.reset_data_buffers()
    lockin.start()
    time.sleep(0.1)
    lockin.pause()
    R = lockin.get_ascii_buffer_data(1, 0, lockin.buffer_size)
    old_mean_R = np.array(list(R)).mean()
    # if first measurement is way below the range, don't wait to settle
    if old_mean_R * 100 > lockin.sensitivities[lockin.sensitivity]:
        t_start = time.time()
        while True:
            if time.time() - t_start > timeout:
                print("Timed out waiting for signal to settle.")
                # init new_mean_R in case timeout is 0
                new_mean_R = old_mean_R
                break
            else:
                lockin.reset_data_buffers()
                lockin.start()
                time.sleep(0.1)
                lockin.pause()
                R = lockin.get_ascii_buffer_data(1, 0, lockin.buffer_size)
                new_mean_R = np.array(list(R)).mean()
                if math.isclose(old_mean_R, new_mean_R, rel_tol=0.1):
                    break
                old_mean_R = new_mean_R
    else:
        new_mean_R = old_mean_R

    return new_mean_R


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
            wait_for_lia_to_settle(lockin, 10)
            logger.debug(f"auto_gain()")
        elif auto_gain_method == "user":
            while True:
                sensitivity_int = lockin.sensitivity
                sensitivity = lockin.sensitivities[sensitivity_int]

                R = wait_for_lia_to_settle(lockin, 10)
                if (R >= sensitivity * 0.8) and (sensitivity_int < 26):
                    new_sensitivity = sensitivity_int + 1
                elif (R <= 0.2 * sensitivity) and (sensitivity_int > 0):
                    new_sensitivity = sensitivity_int - 1
                else:
                    break
                lockin.sensitivity = new_sensitivity
        else:
            raise ValueError(
                f"Invalid auto-gain method: {auto_gain_method}. Must be 'instr' or "
                + "'user'."
            )

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
    smu_compliance=1,
    start_wl=350,
    end_wl=1100,
    num_points=76,
    grating_change_wls=None,
    filter_change_wls=None,
    time_constant=8,
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
    smu_voltage : float
        SMU voltage bias in V.
    smu_compliance : float
        SMU compliance current in A.
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
    time_constant : int
        Time constant setting for the lock-in amplifier.
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
    # basic monochromator setup
    mono.scan_speed = 1000

    # basic lock-in setup
    lockin.line_notch_filter_status = 3
    lockin.reference_source = 0
    lockin.reference_trigger = 1
    lockin.harmonic = 1
    lockin.input_configuration = 1
    lockin.input_shield_grounding = 1
    lockin.input_coupling = 0
    lockin.reserve_mode = 1
    lockin.lowpass_filter_slope = 1
    lockin.set_display(1, 1, 0)
    lockin.set_display(2, 1, 0)
    lockin.alarm_status = 0

    # set lock-in integration time/time constant
    lockin.time_constant = time_constant

    # reset sensitivity to lowest setting to prevent overflow
    lockin.sensitivity = 26

    # determine whether to turn on the synchronous filter
    if lockin.reference_frequency < 200:
        lockin.sync_filter_status = 1

    # set up reading into buffer
    lockin.trigger_start_mode = 0
    lockin.end_of_buffer_mode = 0
    lockin.sample_rate = 13
    lockin.data_transfer_mode = 0

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
    smu.setupDC(
        sourceVoltage=True,
        compliance=smu_compliance,
        setPoint=smu_voltage,
        senseRange="f",
    )
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

        print(f"wl = {data[1]}, R = {data[8]}, phase = {data[9]}")

        scan_data.append(data)

        # update the keithley display
        # measures AC + DC so can't rely on it for DC background measurement
        smu.measure()

        if handler is not None:
            handler(data, **handler_kwargs)
        logger.info(f"{data}")

    # reset sensitivity to lowest setting to prevent overflow
    lockin.sensitivity = 26

    # return wavelength to white
    set_wavelength(mono, 0, grating_change_wls, filter_change_wls)

    # turn of smu if present
    smu.outOn(False)

    # turn off LEDs
    psu.set_output_enable(False, 1)
    psu.set_output_enable(False, 2)
    psu.set_output_enable(False, 3)

    return scan_data


if __name__ == "__main__":
    import sr830
    import sp2150
    import dp800

    from central_control_dev.virt import k2400

    lia_address = "GPIB::8::INSTR"
    mono_address = "ASRL5::INSTR"
    psu_address = "TCPIP0::10.42.0.101::INSTR"
    smu_address = ""

    # connect to instruments
    lia = sr830.sr830()
    lia.connect(lia_address, output_interface=1, reset=False, **{"timeout": 30000})

    print(lia.idn)

    mono = sp2150.sp2150()
    mono.connect(mono_address)

    psu = dp800.dp800()
    psu.connect(psu_address)

    smu = k2400()

    # run a scan
    scan(
        lia,
        mono,
        psu,
        smu,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.1,
        350,
        1100,
        10,
        [1200],
        [370, 640, 715, 765],
        8,
        True,
        "user",
    )

    lia.disconnect()
    mono.disconnect()
    psu.disconnect()
    smu.disconnect()
