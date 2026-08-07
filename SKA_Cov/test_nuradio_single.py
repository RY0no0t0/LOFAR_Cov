import os
import glob
import h5py
import logging
import argparse
import random
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy import optimize as opt
from scipy import constants
from astropy import time
import datetime

from NuRadioReco.framework import station
import NuRadioReco.framework.event
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.efieldGalacticNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.voltageToEfieldConverterPerChannel
import NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup

from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator, readCoREASDetector
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities.trace_utilities import get_electric_field_energy_fluence

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from NuRadioReco.detector.antennapattern import AntennaPatternProvider, preprocess_LOFAR_txt

import json

# Define some functions for later
def calculate_fluence_around_peak(
    trace,
    sampling,
    signal_window: float = 25 * units.ns,
    sample_axis: int = 1,
    return_uncertainty=False,
):
    """
    Calculate the fluence values near the peak of the signal. This is done by taking a single window,
    and calculating the fluence around there. The signal window can be set as an input.

    Parameters:
    -----------
    trace: np.ndarray
        The electric field trace for which to calculate the fluence.
    sampling: float
        The sampling rate of the trace.
    signal_window: float
        The width of the signal window around the peak to consider for the fluence calculation. Default is 25 ns.
    sample_axis: int
        The axis along which to find the peak sample. Default is 1 (assuming shape is (n_antennas, n_samples)).
    return_uncertainty: bool
        Whether to return the uncertainty of the fluence calculation. Default is False.
    """
    conversion_factor_integrated_signal = (
        constants.c * constants.epsilon_0 * units.joule / units.s / units.volt**2
    )

    peak_sample = np.argmax(trace, axis=sample_axis)
    window_lower = np.clip(
        peak_sample - int(signal_window / sampling), 0, trace.shape[sample_axis]
    )
    window_higher = np.clip(
        peak_sample + int(signal_window / sampling), 0, trace.shape[sample_axis]
    )

    fluence = []
    fluence_error = []
    # here the iteration is over the polarisations of the traces?
    for signal, low, high in zip(trace, window_lower, window_higher):
        noise_start = -500

        f_signal = np.sum(signal[low:high] ** 2)
        f_noise = np.sum(signal[noise_start:] ** 2)
        # print(f_signal, f_noise)

        f_signal -= f_noise * (high - low) / len(signal[noise_start:])
        if f_signal < 0:
            f_signal = 0

        RMSNoise = np.sqrt(np.mean(signal[noise_start:] ** 2))

        signal_energy_fluence = (
            f_signal * sampling * conversion_factor_integrated_signal
        )

        signal_window_duration = (high - low) * sampling
        signal_energy_fluence_error = (
            4
            * np.abs(signal_energy_fluence / conversion_factor_integrated_signal)
            * RMSNoise**2
            * sampling
            + 2 * signal_window_duration * RMSNoise**4 * sampling
        ) ** 0.5 * conversion_factor_integrated_signal

        fluence.append(np.sum(signal_energy_fluence))
        fluence_error.append(np.sqrt(np.sum(signal_energy_fluence_error**2)))

    if return_uncertainty:
        return np.asarray(fluence), np.asarray(fluence_error)

    return np.asarray(fluence)

def make_detector_from_coreas_shower(coreas_hdf5_file, path_to_json, site='lofar'):
    """
    make a generic detector json file from a given coreas hdf5 file.
    """
    detector_dict = {
        "stations" : {},
        "channels" : {}
    }

    ant_types = {
        'lofar' : ["LOFAR_LBA_Y", "LOFAR_LBA_X"],
        'ska' : ["SKALA_v4_Xpol", "SKALA_v4_Ypol"]
    }[site]

    ant_orientations = [(135,90), (225, 90)]

    corsika_evt = coreas.read_CORSIKA7(coreas_hdf5_file, declination=None, site=site)
    # here we force vertical core coordinate
    corsika_evt.get_first_sim_shower().set_parameter(
        shp.core,
        np.array(
            [0, 0, corsika_evt.get_first_sim_shower().get_parameter(shp.observation_level)]
        ),
    )

    evt = NuRadioReco.framework.event.Event(corsika_evt.get_run_number(), corsika_evt.get_id())

    # create sim shower, core is already set in read_CORSIKA7()
    sim_shower = coreas.create_sim_shower(corsika_evt)
    evt.set_event_time(corsika_evt.get_event_time())
    evt.add_sim_shower(sim_shower)

    # add simulated pulses as sim station
    corsika_efields = corsika_evt.get_station(0).get_sim_station().get_electric_fields()
    for station_id, corsika_efield in enumerate(corsika_efields):
        station = NuRadioReco.framework.station.Station(station_id)
        sim_station = coreas.create_sim_station(station_id, corsika_evt)
        efield_trace = corsika_efield.get_trace()
        efield_sampling_rate = corsika_efield.get_sampling_rate()
        efield_times = corsika_efield.get_times()


        channel_ids = (np.array([0, 1]) + 2 * station_id).astype(int)

        coreas.add_electric_field_to_sim_station(
            sim_station, channel_ids, efield_trace, efield_times[0],
            sim_shower.get_parameter(shp.zenith), sim_shower.get_parameter(shp.azimuth),
            efield_sampling_rate)

        station.set_sim_station(sim_station)
        evt.set_station(station)

        efield_pos = corsika_efield.get_position()

        detector_dict["stations"][str(station_id)] = {
            "pos_altitude" : 0.0,
            "pos_northing" : efield_pos[1],
            "pos_easting" : efield_pos[0],
            "station_id" : int(station_id),
            "pos_site" : site,
            "commission_time" : "{TinyDate}:2010-06-12T00:00:00",
            "decommission_time" : "{TinyDate}:2038-01-01T00:00:00"
        }

        for i, channel_id in enumerate(channel_ids):
            detector_dict["channels"][str(channel_id)] = {
                "ant_rotation_phi": 0.0,
                "ant_rotation_theta": 0.0,
                "ant_type":ant_types[i],
                "channel_id" : int(channel_id),
                "station_id" : int(station_id),
                "commission_time" : "{TinyDate}:2010-06-12T00:00:00",
                "decommission_time" : "{TinyDate}:2038-01-01T00:00:00",
                "ant_position_x": 0.0,
                "ant_position_y": 0.0,
                "ant_position_z": efield_pos[2],
                "ant_orientation_phi":ant_orientations[i][0],
                "ant_orientation_theta":ant_orientations[i][1],
                "adc_n_samples": 256,
                "adc_sampling_frequency": 0.2,
                "amp_type": "100",
                "channel_group_id": int(channel_id),
            }

    with open(path_to_json, "w") as json_output:
        json.dump(detector_dict, json_output)

    return detector_dict


class NuRadioRecoReader:
    def __init__(self, det, filt_settings, sky_model="gsm2008"):
        """
        Wrapper module to read in NuRadioReco events from CoREAS simulations with applied antenna response, bandpass, and noise, and optionally simulating the trigger and signal reconstruction.

        This can be used to read events to be applied to any reconstruction, such as a fluence-based reconstruction.

        Parameters:
        –----------
        det: NuRadioReco.detector.Detector object
            The detector object containing the station and antenna information.
        filt_settings: dict
            The settings for the bandpass filter to be applied to the electric field and voltage traces.

            Example:
                filter_settings = {
                    "passband": [30 * units.MHz, 80 * units.MHz],
                    "filter_type": "butter",
                    "order": 10,
                }
        """
        self.det = det
        self.filter_settings = filt_settings
        self.logger = logging.getLogger("NuRadioFluenceReco")
        self.logger.setLevel(logging.ERROR)

        self.__initialize_modules(sky_model)

    def __initialize_modules(self, sky_model="gsm2008"):
        """
        Initializes all required modules from NuRadio. This includes:

        - bandpass filter in Efield level
        - Signal reconstruction from Voltage to Electric Field
        - Galactic noise adder in Efield level

        - Voltage to Electric Field converter in Efield level
        - Generic noise adder in Voltage level
        - Trigger simulator
        - Bandpass filter in Voltage level
        - Event type identifier
        - Signal reconstructor in Voltage level
        - coreas reader
        """

        # Initialize the modules
        self.efieldBandpassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
        self.efieldBandpassFilter.begin()
        self.electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
        self.electricFieldSignalReconstructor.begin(
            noise_window=400 * units.ns
        )
        self.efieldGalacticNoiseAdder = (
            NuRadioReco.modules.efieldGalacticNoiseAdder.efieldGalacticNoiseAdder()
        )
        self.efieldToVoltageConverter = (
            NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(
                log_level=logging.INFO
            )
        )
        self.efieldToVoltageConverter.begin(
            debug=False, pre_pulse_time=0, post_pulse_time=400 * units.ns
        )
        self.channelGalacticNoiseAdder = (
            NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
        )
        self.channelGalacticNoiseAdder.begin(skymodel=sky_model)

        self.channelGenericNoiseAdder = (
            NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        )
        self.channelGenericNoiseAdder.begin()
        self.triggerSimulator = (
            NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
        )
        self.triggerSimulator.begin()
        self.channelBandPassFilter = (
            NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        )
        self.channelBandPassFilter.begin()
        self.eventTypeIdentifier = (
            NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
        )
        self.channelSignalReconstructor = (
            NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
        )
        self.channelSignalReconstructor.begin()
        self.voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
        self.voltageToEfieldConverter.begin()

        self.coreas_reader = readCoREASDetector.readCoREASDetector()

    def read_data_event_with_noise(
        self,
        in_file,
        site="lofar",
        Tnoise=300 * units.K,
        core=np.array([0, 0, 0]) * units.m,
        selected_station_channel_ids = None
    ):
        """
        Read in a single event from the CoREAS simulation, apply the antenna response, bandpass filter, and noise, and simulate the trigger and signal reconstruction. This is useful for testing the fluence reconstruction on a "realistic" signal with noise and trigger effects included.
        """
        evt = coreas.read_CORSIKA7(in_file, site=site)

        # here we force vertical core coordinate
        evt.get_first_sim_shower().set_parameter(
            shp.core,
            np.array(
                [0, 0, evt.get_first_sim_shower().get_parameter(shp.observation_level)]
            ),
        )

        interpolator = coreasInterpolator.coreasInterpolator(evt)
        interpolator.initialize_efield_interpolator(
            interp_lowfreq=self.filter_settings["passband"][0], interp_highfreq=self.filter_settings["passband"][1]
        )
        self.coreas_reader.coreas_interpolator = interpolator  # skip begin() function because HDF5 does not have good CoreCoordinateVertical
        self.coreas_reader._readCoREASDetector__corsika_evt = evt
        # we only need a single realization of the shower, so we set the core position to zero for simplicity
        iplot = 0
        plot_idx = 32
        plot_colors = ["royalblue", "forestgreen", "indianred"]
        # Randomize time for galactic noise
        start_date = datetime.datetime(2013, 1, 1, 0, 0, 0)
        end_date = datetime.datetime(2026, 1, 1, 0, 0, 0)
        total_seconds = (end_date - start_date).total_seconds()

        for _, evt in enumerate(self.coreas_reader.run(self.det, [core], selected_station_channel_ids=selected_station_channel_ids)):
            station_idx = 0
            for station in evt.get_stations():

                random_seconds = random.uniform(0, total_seconds)
                random_date = start_date + datetime.timedelta(seconds=random_seconds)
                station.set_station_time(random_date)  # set station time to a fixed value for simplicity

                # self.efieldBandpassFilter.run(evt, station.get_sim_station(), self.det, **self.filter_settings)
                
                if iplot == plot_idx:
                    fig, ax = plt.subplots()
                    for iefield, efield in enumerate(station.get_sim_station().get_electric_fields()):
                        traces = efield.get_trace()/(units.V / units.m)
                        times = efield.get_times()

                        if iefield == 0:
                            for itr, tr in enumerate(traces):
                                ax.plot(times, tr, label=f"Pol {itr}", color=plot_colors[itr])
                    ax.legend()
                    ax.set_xlabel("Time / ns", fontsize=16)
                    ax.set_ylabel("Electric Field / (V/m)", fontsize=16)

                    ax.tick_params(axis='both', which='major', labelsize=14, size=6)

                    fig.savefig("./after_running_efield.pdf", dpi=200, bbox_inches='tight')
                    plt.clf()

                # for channel in station.iter_channels():
                #     trace = channel.get_trace() / units.V
                #     times = channel.get_times()
                #     ax.plot(times, trace)


                # fig.savefig("./after_running.pdf", dpi=200, bbox_inches='tight')

                # plt.clf()

                # apply antenna response
                self.efieldToVoltageConverter.run(evt, station, self.det)

                if iplot == plot_idx:
                    fig, ax = plt.subplots()

                    for ichannel, channel in enumerate(station.iter_channels()):
                        trace = channel.get_trace() / units.V
                        times = channel.get_times()
                        ax.plot(times, trace, label=f'Pol {ichannel}', color=plot_colors[ichannel])

                    ax.set_xlabel("Time / ns", fontsize=16)
                    ax.set_ylabel("Voltage / V", fontsize=16)
                    ax.legend(fontsize=12)

                    ax.tick_params(axis='both', which='major', labelsize=14, size=6)
                    fig.savefig("./after_efield_voltage_voltage.pdf", dpi=200, bbox_inches='tight')

                    plt.clf()

                # approximate the rest of the signal chain with a bandpass filter
                self.channelBandPassFilter.run(evt, station, self.det, **self.filter_settings)

                if iplot == plot_idx:
                    fig, ax = plt.subplots()

                    for ichannel, channel in enumerate(station.iter_channels()):
                        trace = channel.get_trace() / units.V
                        times = channel.get_times()
                        ax.plot(times, trace, label=f'Pol {ichannel}', color=plot_colors[ichannel])

                    ax.set_xlabel("Time / ns", fontsize=16)
                    ax.set_ylabel("Voltage / V", fontsize=16)
                    ax.legend(fontsize=12)

                    ax.tick_params(axis='both', which='major', labelsize=14, size=6)

                    fig.savefig("./after_channel_bandpass_voltage.pdf", dpi=200, bbox_inches='tight')

                    plt.clf()

                # calculate Vrms and normalize such that after filtering the correct Vrms is obtained
                min_freq = 0
                max_freq = 0.5 * self.det.get_sampling_frequency(station.get_id(), station.get_channel_ids()[0])
                ff = np.linspace(0, max_freq, 10000)
                filt = self.channelBandPassFilter.get_filter(
                    ff, station.get_id(), None, self.det, **self.filter_settings
                )
                bandwidth = np.trapezoid(np.abs(filt) ** 2, ff)
                Vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
                amplitude = Vrms / (bandwidth / max_freq) ** 0.5
                print(f"Calculated Vrms: {Vrms:.2e} V, applying noise with amplitude {amplitude:.2e} V")
                self.channelGenericNoiseAdder.run(
                    evt,
                    station,
                    self.det,
                    type="rayleigh",
                    amplitude=amplitude,
                    min_freq=min_freq,
                    max_freq=max_freq,
                )

                self.channelGalacticNoiseAdder.run(
                    evt, station, self.det, passband=self.filter_settings['passband']
                )

                if iplot == plot_idx:
                    fig, ax = plt.subplots()

                    for ichannel, channel in enumerate(station.iter_channels()):
                        trace = channel.get_trace() / units.V
                        times = channel.get_times()
                        ax.plot(times, trace, label=f'Pol {ichannel}', color=plot_colors[ichannel])

                    ax.set_xlabel("Time / ns", fontsize=16)
                    ax.set_ylabel("Voltage / V", fontsize=16)
                    ax.legend(fontsize=12)

                    ax.tick_params(axis='both', which='major', labelsize=14, size=6)

                    fig.savefig("./after_noise_voltage.pdf", dpi=200, bbox_inches='tight')

                    plt.clf()

                self.triggerSimulator.run(
                    evt, station, self.det, number_concidences=1, threshold=10 * Vrms
                )
                if station.get_trigger("default_simple_threshold").has_triggered():
                    self.eventTypeIdentifier.run(evt, station, "forced", "cosmic_ray")

                    self.channelSignalReconstructor.run(evt, station, self.det)

                    if iplot == plot_idx:
                        fig, ax = plt.subplots()

                        for ichannel, channel in enumerate(station.iter_channels()):
                            trace = channel.get_trace() / units.V
                            times = channel.get_times()
                            ax.plot(times, trace, label=f'Pol {ichannel}', color=plot_colors[ichannel])

                        ax.set_xlabel("Time / ns", fontsize=16)
                        ax.set_ylabel("Voltage / V", fontsize=16)
                        ax.legend(fontsize=12)

                        ax.tick_params(axis='both', which='major', labelsize=14, size=6)

                        fig.savefig("./after_trigger_voltage.pdf", dpi=200, bbox_inches='tight')

                        plt.clf()
                    
                    # reconstruct the electric field for each dual-polarized antenna through standard unfolding
                    self.voltageToEfieldConverter.run(evt, station, self.det, use_channels=station.get_channel_ids(), use_MC_direction=True)

                    if iplot == plot_idx:
                        fig, ax = plt.subplots()
                        for iefield, efield in enumerate(station.get_electric_fields()):
                            traces = efield.get_trace()/(units.V / units.m)
                            times = efield.get_times()

                            for itr, tr in enumerate(traces):
                                ax.plot(times, tr, label=f"Pol {itr}", color=plot_colors[itr])
                        ax.set_xlabel("Time / ns", fontsize=16)
                        ax.set_ylabel("Electric Field / (V/m)", fontsize=16)
                        ax.legend(fontsize=12)

                        ax.tick_params(axis='both', which='major', labelsize=14, size=6)


                        fig.savefig("./after_efield_reco_efield.pdf", dpi=200, bbox_inches='tight')

                        antenna_position = self.det.get_absolute_position(station.get_id()) + self.det.get_relative_position(station.get_id(), channel.get_id())
                        print(f"Antenna POsition: {antenna_position}")

                        # raise Exception("Stop after plotting the reconstructed electric field for a single station.")


                iplot += 1

        return evt

    def plot_all_traces(self, my_event, my_detector, savefig_path=None):
        """
        Given an event and a detector description, plot all traces for each station,
        split by the channel orientation.

        Parameters
        ----------
        my_event : Event
            The event to plot traces for
        my_detector : Detector
            The detector description
        title : str, optional
            Title to give to the plot
        """
        for my_station in my_event.get_stations():
            channels_per_orientation = my_detector.get_parallel_channels(my_station.get_id())

            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            ax = ax.flatten()

            for my_channel in my_station.iter_channels():
                if my_channel.get_id() in set(channels_per_orientation[0]):
                    ax[0].plot(my_channel.get_times() / units.ns, my_channel.get_trace()  / units.V, label=f'Pol {my_channel.get_id()}')
                else:
                    ax[1].plot(my_channel.get_times() / units.ns, my_channel.get_trace()  / units.V, label=f'Pol {my_channel.get_id()}')

            ax[0].set_title(f'Station CS{my_station.get_id():03d}')
            # ax[0].legend()
            ax[0].set_ylabel('Voltage X [V]')
            ax[1].set_xlabel('Time [ns]')
            ax[1].set_ylabel('Voltage Y [V]')

            if savefig_path is not None:
                os.makedirs(savefig_path, exist_ok=True)
                fig.savefig(os.path.join(
                    savefig_path, f"station_{my_station.get_id():03d}.png"), dpi=200, bbox_inches='tight')


    def get_fluence_interpolator(self, input_file, site="lofar"):
        """
        Given a CoREAS simulation file, read in the event and return a fluence interpolator object.

        Parameters
        ----------
        input_file : str
            Path to the CoREAS simulation file.
        site : str, optional
            The site for the simulation.
        """
        # read in CoREAS simulation
        current_sim_evt: NuRadioReco.framework.event.Event = coreas.read_CORSIKA7(
            input_file, site=site
        )
        current_sim_evt.get_first_sim_shower().set_parameter(
            shp.core,
            np.array(
                [
                    0,
                    0,
                    current_sim_evt.get_first_sim_shower().get_parameter(
                        shp.observation_level
                    ),
                ]
            ),
        )

        for station in current_sim_evt.get_stations():
            self.efieldBandpassFilter.run(
                current_sim_evt,
                station.get_sim_station(),
                None,
                **self.filter_settings,
            )

        interpolator = coreasInterpolator.coreasInterpolator(current_sim_evt)
        
        # this sets the fluence from all electric fields
        # stored in sim_station
        interpolator.set_fluence_of_efields(
            lambda trace: calculate_fluence_around_peak(
                trace,
                current_sim_evt.get_station()
                .get_sim_station()
                .get_electric_fields()[0]
                .get_sampling_rate(),
            )
        )

        interpolator.initialize_fluence_interpolator()

        return interpolator
                
    def plot_footprint(self, my_event, my_detector, input_file, savefig_path=None):
        """
        Given an event and a detector description, plot the footprint of the event.

        Parameters
        ----------
        my_event : Event
            The event to plot the footprint for
        my_detector : Detector
            The detector description
        title : str, optional
            Title to give to the plot
        """
        fluences = []
        positions = []
        for station in my_event.get_stations():
            for channel in station.iter_channels():
                sim_station = station.get_sim_station()
                efields = sim_station.get_electric_fields()
                for efield in efields:
                    chid_efield = efield.get_unique_identifier()[0][0]
                    if channel.get_id() != chid_efield:
                        print(f"Channel ID mismatch: {channel.get_id()} != {chid_efield}")
                        continue
                    trace = efield.get_trace()
                    fluence = calculate_fluence_around_peak(
                        trace,
                        efield.get_sampling_rate(),
                    )
                    # fluence = get_electric_field_energy_fluence(trace, efield.get_times())
                    fluences.append(np.sum(fluence))
                    positions.append(my_detector.get_absolute_position(station.get_id()) + my_detector.get_relative_position(station.get_id(), channel.get_id()))

        positions = np.array(positions)
        fluences = np.array(fluences)

        fluence_cmap = cm.plasma
        fluence_norm = mcolors.Normalize(vmin=0, vmax=fluences.max(), clip=True)

        # plot the footprint
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(positions[:, 0], positions[:, 1], c = fluences, cmap=fluence_cmap, marker=".", norm=fluence_norm, s=80.0, edgecolor='black', linewidth=0.5, zorder=10)
        ax.tick_params(axis='both', which='major', labelsize=20, size=6)
        ax.tick_params(axis='both', which='minor', labelsize=20, size=4)

        ax.set_xlabel("x / m", fontsize=22)
        ax.set_ylabel("y / m", fontsize=22)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel("Fluence / eV m$^{-2}$", fontsize=22)
        cbar.ax.tick_params(axis='both', which='major', labelsize=20, size=6)

        # plot the interpolated footprint behind it also
        interpolator = self.get_fluence_interpolator(input_file, site="lofar")
        # Make color plot of f(x, y), using a meshgrid
        ti = np.linspace(-400, 400, 500)
        XI, YI = np.meshgrid(ti, ti)

        ZI = np.zeros_like(XI)
        R = np.sqrt(XI**2 + YI**2)
        mask = R <= 600
        idxs = np.argwhere(mask)
        xs, ys = XI[mask], YI[mask]
        for (i, j), xi, yi in zip(idxs, xs, ys):
            ZI[i, j] = interpolator.get_interp_fluence_value((xi, yi))


        ax.pcolormesh(XI, YI, ZI, norm=fluence_norm, cmap=fluence_cmap, alpha=0.9, shading='auto', zorder=0)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])

        fig.savefig(os.path.join(savefig_path, "footprint.png"), dpi=200, bbox_inches='tight')

if __name__ == "__main__":

    # NOTE: modify this to the path where you save the LOFAR antenna response.
    path_to_response = "/home/rkitahara/Research/Cov/test_simulation/antenna_response_lofar"

     # NOTE: modify this to the path where you save the coreas simulation file. This is just an example path, you need to change it to your own.
    path_to_coreas_sim_file = "/home/rkitahara/Research/Cov/test_simulation/93970574/0/proton/SIM000018.hdf5"

    det = detector.Detector(
        "LOFAR/LOFAR.json", source="json", antenna_by_depth=False
    )
    det.update(datetime.datetime(2013, 1, 1, 0, 0, 0))
    selected_station_channel_ids = {}
    for staid in [1, 2, 3, 4, 5, 6, 7]:
        selected_station_channel_ids[staid] = det.get_channel_ids(staid)

    preprocess_LOFAR_txt(path_to_response, orientation="Y")
    preprocess_LOFAR_txt(path_to_response, orientation="X")
    # selected_station_channel_ids = None
    # make_detector_from_coreas_shower(path_to_coreas_sim_file, "./temp_detector_file.json", site="lofar")
    # det = detector.Detector(
    #     "./temp_detector_file.json", source="json", assume_inf=False, antenna_by_depth=False,
    # )
    det.update(datetime.datetime(2013, 1, 1, 0, 0, 0))

    nu_radio_reader = NuRadioRecoReader(det, filt_settings={"passband": [30 * units.MHz, 80 * units.MHz], "filter_type": "butter", "order": 10})

    core_xy = np.array([0, 0]) * units.m
    mc_data_evt = nu_radio_reader.read_data_event_with_noise(
            path_to_coreas_sim_file,
            selected_station_channel_ids=selected_station_channel_ids,
            core=core_xy,
            Tnoise=200 * units.K, # you can modify this
        )
    
    # # example how to plot the voltage traces after noise addition
    # nu_radio_reader.plot_all_traces(mc_data_evt, det, savefig_path="./plots_after_noise")
     # example how to plot the voltage traces after noise addition
    nu_radio_reader.plot_footprint(mc_data_evt, det, input_file=path_to_coreas_sim_file, savefig_path="../plots_after_noise")