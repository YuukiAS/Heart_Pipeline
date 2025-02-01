"""
This file defines an ECG processor class that can be used to process the ECG data from the UK Biobank dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging
from utils.ecg_xml_reader import CardioSoftECGXMLReader as XMLReader

logger = setup_logging("ecg_processor")


class ECG_Processor:
    def __init__(self, subject, retest=False, sampling_rate=500):
        if type(subject) is not str:
            raise TypeError("Subject must be a string")
        self.subject = subject
        self.retest = retest
        self.sampling_rate = sampling_rate

        self.voltages_rest = None
        self.voltages_exercise = None
        if self.check_data_rest():
            self.voltages_rest = self._load_data_rest()
        if self.check_data_exercise():
            # define Four stages: pretest, exercise(constant), exercise(rampling) and rest
            self.voltages_exercise = self._load_data_exercise()

    def check_data_rest(self):
        if self.retest is False:
            return os.path.exists(os.path.join(config.data_visit1_dir, self.subject, "ecg_rest.xml"))
        else:
            return os.path.exists(os.path.join(config.data_visit2_dir, self.subject, "ecg_rest.xml"))

    def check_data_exercise(self):
        if self.retest is False:
            return os.path.exists(os.path.join(config.data_visit1_dir, self.subject, "ecg_exercise.xml"))
        else:
            return os.path.exists(os.path.join(config.data_visit2_dir, self.subject, "ecg_exercise.xml"))

    def _load_data_rest(self):
        if not self.check_data_rest():
            raise FileNotFoundError("ECG data does not exist for the subject")

        if self.retest is False:
            xml_file = os.path.join(config.data_visit1_dir, self.subject, "ecg_rest.xml")
        else:
            xml_file = os.path.join(config.data_visit2_dir, self.subject, "ecg_rest.xml")

        try:
            data_reader = XMLReader(xml_file, ecg_type="rest")
            voltages = data_reader.getVoltages()
        except ValueError as e:
            logger.error(f"Subject {self.subject}: While processing rest XML file: Error {e}")
            voltages = None
        return voltages

    def _load_data_exercise(self):
        if not self.check_data_exercise():
            raise FileNotFoundError("ECG data does not exist for the subject")

        if self.retest is False:
            xml_file = os.path.join(config.data_visit1_dir, self.subject, "ecg_exercise.xml")
        else:
            xml_file = os.path.join(config.data_visit2_dir, self.subject, "ecg_exercise.xml")

        try:
            data_reader = XMLReader(xml_file, ecg_type="exercise")
            voltages_divided = data_reader.getDividedVoltages()
        except ValueError as e:
            logger.error(f"Subject {self.subject}: While processing exercise XML file: Error {e}")
            voltages_divided = None
        return voltages_divided

    def get_voltages_rest(self, lead=None):
        if self.voltages_rest is None:
            raise ValueError(f"Subject {self.subject}: ECG rest data is not available")
        if not lead:
            return self.voltages_rest
        else:
            if lead in self.voltages_rest:
                return self.voltages_rest[lead]
            else:
                raise ValueError(f"Subject {self.subject}: Lead {lead} is not available in ECG rest data")

    def get_voltages_rest_cleaned(self, lead=None):
        """
        Clean the rest ECG signal for a specifc lead using a bandpass filter.
        """
        ecg = self.get_voltages_rest(lead)
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=self.sampling_rate, method="pantompkins1985")
        return ecg_cleaned

    def get_voltages_exercise(self):
        if self.voltages_exercise is None:
            raise ValueError(f"Subject {self.subject}: ECG exercise data is not available")
        return self.voltages_exercise

    def determine_RR_interval(self):
        """
        Determine the mean RR interval based on rest ECG XML file.
        """
        if self.voltages_rest is None:
            raise ValueError(f"Subject {self.subject}: ECG rest data is not available")

        RR_interval_leads = []
        for lead in self.voltages_rest:
            ecg_signal = self.voltages_rest[lead]
            _, info = nk.ecg_process(ecg_signal, sampling_rate=self.sampling_rate)
            R_peaks = np.array(info["ECG_R_Peaks"])
            RR_interval = np.mean(np.diff(R_peaks))
            RR_interval_leads.append(RR_interval)
        # If sample rate is 500 (in UK Biobnak), then we actually need to multiply by 2
        return np.mean(np.array(RR_interval_leads)) * (1000 / self.sampling_rate)  # unit: ms

    def determine_timepoint_LA(
        self, methods=["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012"], log=False
    ):
        """
        Determine the pre-contraction point for left atrium based on rest ECG XML file.
        The point for maximum volume is also reported, but does not yield satisfactory results.
        """
        if self.voltages_rest is None:
            raise ValueError(f"Subject {self.subject}: ECG rest data is not available")

        t_max_total = []
        t_pre_a_total = []

        for lead in self.voltages_rest:
            # Ref Principles of ECG Gating for CMR https://link.springer.com/10.1007/978-3-319-22141-0_9
            # define t_pre_a is the peak of P wave
            # define t_max is the end of T wave
            if lead in ["II"]:  # This is subject to change
                ecg_signal = self.voltages_rest[lead]

                t_max_lead = []
                t_pre_a_lead = []

                # Ref NeuroKit https://neuropsychology.github.io/NeuroKit/functions/ecg.html
                for method in methods:
                    try:
                        _, info = nk.ecg_process(ecg_signal, sampling_rate=self.sampling_rate, method=method)
                        L = len(info["ECG_R_Peaks"])
                    except ZeroDivisionError as e:
                        logger.warning(f"Method {method} failed for lead {lead} due to ZeroDivisionError {e}")
                        continue
                    except ValueError as e:
                        logger.warning(f"Method {method} failed for lead {lead} due to ValueError {e}")
                        continue
                    except IndexError as e:
                        logger.warning(f"Method {method} failed for lead {lead} due to IndexError {e}")
                        continue

                    t_max_values = []
                    for i in range(1, L - 1):
                        t_max_values.append((info["ECG_T_Offsets"][i] - info["ECG_R_Peaks"][i]) * 2)
                    if np.all(np.isnan(t_max_values)):  # a meaningless list full of nan
                        continue
                    t_max_mean = np.nanmean(t_max_values)
                    t_max_lead.append(t_max_mean)

                    t_pre_a_values = []
                    for i in range(1, L - 1):
                        t_pre_a_values.append((info["ECG_P_Peaks"][i + 1] - info["ECG_R_Peaks"][i]) * 2)
                    if np.all(np.isnan(t_pre_a_values)):
                        continue
                    t_pre_a_mean = np.nanmean(t_pre_a_values)
                    t_pre_a_lead.append(t_pre_a_mean)

                if len(t_max_lead) == 0 or len(t_pre_a_lead) == 0:
                    logger.error(f"All methods failed for lead {lead}!")
                    raise ValueError(f"All methods failed for lead {lead}!")

                # Remove maximum and minimum values if using multiple methods
                if len(t_max_lead) >= 3:
                    t_max_lead.remove(max(t_max_lead))
                    t_max_lead.remove(min(t_max_lead))
                if len(t_pre_a_lead) >= 3:
                    t_pre_a_lead.remove(max(t_pre_a_lead))
                    t_pre_a_lead.remove(min(t_pre_a_lead))

                if log:
                    print(f"Lead {lead}: t_max {np.nanmean(t_max_lead)}, t_pre_a {np.nanmean(t_pre_a_lead)}")

                t_max_total.append(np.nanmean(t_max_lead))
                t_pre_a_total.append(np.nanmean(t_pre_a_lead))

        return {"t_max": np.mean(t_max_total), "t_pre_a": np.mean(t_pre_a_total)}

    def visualize_ecg_info(self, lead="II", last_num=2):
        """
        Return a dictionary that can be used to visualize one segment ECG signal alongwith other information such as volumes.
        By default, we will use the lead II ECG, as recommended when constructing the Wiggers diagram.
        """
        ecg = self.get_voltages_rest(lead)
        signals, info = nk.ecg_process(ecg, sampling_rate=self.sampling_rate)

        P_Peaks = info["ECG_P_Peaks"]
        P_Peaks_index = np.where(~np.isnan(P_Peaks))[0][-1]
        Q_Peaks = info["ECG_Q_Peaks"]
        Q_Peaks_index = np.where(~np.isnan(Q_Peaks))[0][-1]
        R_Peaks = info["ECG_R_Peaks"]
        R_Peaks_index = np.where(~np.isnan(R_Peaks))[0][-1]
        S_Peaks = info["ECG_S_Peaks"]
        S_Peaks_index = np.where(~np.isnan(S_Peaks))[0][-1]
        T_Peaks = info["ECG_T_Peaks"]
        T_Peaks_index = np.where(~np.isnan(T_Peaks))[0][-1]

        # We choose the peaks at the later half, which should be accurate than the previous half
        R_index = np.min([P_Peaks_index, Q_Peaks_index, R_Peaks_index, S_Peaks_index, T_Peaks_index]) - last_num
        T_start = R_Peaks[R_index]  # index of the signal, not actual time
        T_end = R_Peaks[R_index + 1]

        # Choose the cloest point on the right to the R peak
        P_index = np.where(P_Peaks > R_Peaks[R_index])[0][0]
        Q_index = np.where(Q_Peaks > R_Peaks[R_index])[0][0]
        S_index = np.where(S_Peaks > R_Peaks[R_index])[0][0]
        T_index = np.where(T_Peaks > R_Peaks[R_index])[0][0]

        time = np.arange(0, T_end - T_start) * 1000 / self.sampling_rate  # unit: ms
        signal = signals["ECG_Clean"][T_start:T_end]

        ecg_info = {
            "time": time,
            "signal": np.array(signal),
            "P_index": P_Peaks[P_index] - T_start,  # index of P peak in the selected part
            "Q_index": Q_Peaks[Q_index] - T_start,
            "R_index": R_Peaks[R_index] - T_start,
            "S_index": S_Peaks[S_index] - T_start,
            "T_index": T_Peaks[T_index] - T_start,
        }
        return ecg_info

    def plot_ecg_signal(time, signal):
        """
        We directly use the code provided in https://www.indigits.com/post/2022/10/ecg_python/.
        Time and signal should be two numpy arrays of equal length. Time should be in unit seconds.
        """
        if len(time) != len(signal):
            raise ValueError("Time and signal should have the same length")

        fig = plt.figure(figsize=(15, 3))
        ax = plt.axes()
        ax.plot(time, signal)
        # setup major and minor ticks
        min_t = int(np.min(time))
        max_t = round(np.max(time))
        major_ticks = np.arange(min_t, max_t + 1)
        ax.set_xticks(major_ticks)
        # Turn on the minor ticks on
        ax.minorticks_on()
        # Make the major grid
        ax.grid(which="major", linestyle="-", color="red", linewidth="1.0")
        # Make the minor grid
        ax.grid(which="minor", linestyle=":", color="black", linewidth="0.5")
        plt.xlabel("Time (sec)")
        plt.ylabel("Amplitude")
        return fig, ax
