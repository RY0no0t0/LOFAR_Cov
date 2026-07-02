import os
import glob
import h5py
import logging
import argparse
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
        for _, evt in enumerate(self.coreas_reader.run(self.det, [core], selected_station_channel_ids=selected_station_channel_ids)):
            station_idx = 0
            for station in evt.get_stations():

                # self.efieldBandpassFilter.run(evt, station.get_sim_station(), self.det, **self.filter_settings)
                
                # fig, ax = plt.subplots()
                # for iefield, efield in enumerate(station.get_sim_station().get_electric_fields()):
                #     traces = efield.get_trace()/(units.V / units.m)
                #     times = efield.get_times()

                #     for itr, tr in enumerate(traces):
                #         ax.plot(times, tr, label=f"Pol {itr}")
                # ax.legend()

                # fig.savefig("./after_running_efield.png")
                # plt.clf()

                # for channel in station.iter_channels():
                #     trace = channel.get_trace() / units.V
                #     times = channel.get_times()
                #     ax.plot(times, trace)

                # fig.savefig("./after_running.png")

                # plt.clf()

                # apply antenna response
                self.efieldToVoltageConverter.run(evt, station, self.det)

                # fig, ax = plt.subplots()

                # for ichannel, channel in enumerate(station.iter_channels()):
                #     trace = channel.get_trace() / units.V
                #     times = channel.get_times()
                #     ax.plot(times, trace, label=f'Pol {ichannel}')

                # fig.savefig("./after_efield_voltage_voltage.png")

                # plt.clf()

                # approximate the rest of the signal chain with a bandpass filter
                self.channelBandPassFilter.run(evt, station, self.det, **self.filter_settings)

                # fig, ax = plt.subplots()

                # for ichannel, channel in enumerate(station.iter_channels()):
                #     trace = channel.get_trace() / units.V
                #     times = channel.get_times()
                #     ax.plot(times, trace, label=f'Pol {ichannel}')

                # fig.savefig("./after_channel_bandpass_voltage.png")

                # plt.clf()

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

                # self.channelGalacticNoiseAdder.run(
                #     evt, station, self.det, passband=self.filter_settings['passband']
                # )


                # fig, ax = plt.subplots()

                # for ichannel, channel in enumerate(station.iter_channels()):
                #     trace = channel.get_trace() / units.V
                #     times = channel.get_times()
                #     ax.plot(times, trace, label=f'Pol {ichannel}')

                # fig.savefig("./after_noise_voltage.png")

                # plt.clf()

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
            ax[0].legend()
            ax[0].set_ylabel('Voltage X [V]')
            ax[1].set_xlabel('Time [ns]')
            ax[1].set_ylabel('Voltage Y [V]')

            if savefig_path is not None:
                os.makedirs(savefig_path, exist_ok=True)
                fig.savefig(os.path.join(
                    savefig_path, f"station_{my_station.get_id():03d}.png"), dpi=200, bbox_inches='tight')
                
    def plot_footprint(self, my_event, my_detector, savefig_path=None):
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
                    fluence = get_electric_field_energy_fluence(trace, efield.get_times())
                    fluences.append(np.sum(fluence))
                    positions.append(my_detector.get_absolute_position(station.get_id()) + my_detector.get_relative_position(station.get_id(), channel.get_id()))

        positions = np.array(positions)
        fluences = np.array(fluences)

        fluence_cmap = cm.plasma
        fluence_norm = mcolors.Normalize(vmin=0, vmax=fluences.max(), clip=True)

        # plot the footprint
        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(positions[:, 0], positions[:, 1], c = fluences, cmap=fluence_cmap, norm=fluence_norm, s=10.0)
        ax.tick_params(axis='both', which='major', labelsize=20, size=6)
        ax.tick_params(axis='both', which='minor', labelsize=20, size=4)

        ax.set_xlabel("x / m", fontsize=22)
        ax.set_ylabel("y / m", fontsize=22)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel("Fluence / eV m$^{-2}$", fontsize=22)
        cbar.ax.tick_params(axis='both', which='major', labelsize=20, size=6)

        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])

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

    nu_radio_reader = NuRadioRecoReader(det, filt_settings={"passband": [30 * units.MHz, 80 * units.MHz], "filter_type": "butter", "order": 10})

    core_xy = np.array([0, 0]) * units.m
    mc_data_evt = nu_radio_reader.read_data_event_with_noise(
            path_to_coreas_sim_file,
            selected_station_channel_ids=selected_station_channel_ids,
            core=core_xy,
            Tnoise=200 * units.K, # you can modify this
        )
    
    # # example how to plot the voltage traces after noise addition
    nu_radio_reader.plot_all_traces(mc_data_evt, det, savefig_path="./plots_after_noise")
     # example how to plot the voltage traces after noise addition
    nu_radio_reader.plot_footprint(mc_data_evt, det, savefig_path="./plots_after_noise")