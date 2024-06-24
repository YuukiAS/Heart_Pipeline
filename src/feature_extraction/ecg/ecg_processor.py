"""
This file defines an ECG processor class that is used to process the ECG data from the UK Biobank dataset.
"""

import os
import numpy as np
import neurokit2 as nk

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging
from utils.ecg_xml_reader import CardioSoftECGXMLReader as XMLReader

logger = setup_logging("ecg_processor")

class ECG_Processor:
    def __init__(self, subject, retest=False):
        self.subject = subject
        self.retest = retest
        
        self.voltages_rest = None
        self.voltages_exercise = None
        if self.check_data_rest():
            self.voltages_rest = self._load_data_rest()
        if self.check_data_exercise():
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

        data_reader = XMLReader(xml_file, ecg_type = "rest")
        leads = data_reader.getVoltages()
        return leads

    def _load_data_exercise(self):
        if not self.check_data_exercise():
            raise FileNotFoundError("ECG data does not exist for the subject")

        if self.retest is False:
            xml_file = os.path.join(config.data_visit1_dir, self.subject, "ecg_exercise.xml")
        else:
            xml_file = os.path.join(config.data_visit2_dir, self.subject, "ecg_exercise.xml")

        data_reader = XMLReader(xml_file, ecg_type = "exercise")
        leads = data_reader.getVoltages()
        return leads

    def get_voltages_rest(self):
        if self.voltages_rest is None:
            raise ValueError("No ECG rest data available for the subject")
        return self.voltages_rest

    def get_voltages_exercise(self):
        if self.voltages_exercise is None:
            raise ValueError("No ECG exercise data available for the subject")
        return self.voltages_exercise
    

    def determine_timepoint_LA(
        self, methods=["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012"], log=False
    ):
        """
        Determine the pre-contraction point for left atrium based on ECG XML file.
        The point for maximum volume is also reported, but does not yield satisfactory results.
        """
        t_max_total = []
        t_pre_a_total = []

        for lead in self.voltages_rest:
            # Ref https://link.springer.com/10.1007/978-3-319-22141-0_9
            # Define t_pre_a is the peak of P wave
            # Define t_max is the end of T wave
            # * We will only make use of certain precordial leads as indicated by fig9.2
            if lead in ["V3", "V4", "V5"]:
                ecg_signal = self.voltages_rest[lead]

                t_max_lead = []
                t_pre_a_lead = []

                # Ref https://neuropsychology.github.io/NeuroKit/functions/ecg.html
                for method in methods:
                    try:
                        _, info = nk.ecg_process(ecg_signal, sampling_rate=500, method=method)
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
