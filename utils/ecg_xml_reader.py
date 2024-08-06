"""
This script is based on https://github.com/paubrunet97/CardioSoftECGXMLreader and https://github.com/broadinstitute/ml4h.
"""

import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import xml.etree.ElementTree as et
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging

logger = setup_logging("ECG-XMLReader")


class CardioSoftECGXMLReader:
    """Extract voltage data from a CardioSoftECG XML file"""

    def __init__(self, path, ecg_type, encoding="ISO8859-1"):
        self.path = path
        # UKBiobank category 104
        if ecg_type == "rest":
            self.type = "rest"
            logger.info("Processing rest ECG XML file")
            with open(path, "rb") as xml:
                self.Data = xmltodict.parse(xml.read().decode(encoding))["CardiologyXML"]

                if "StripData" in self.Data:
                    self.StripData = self.Data["StripData"]
                    self.FullDisclosure = False

                    self.PID = self.Data["PatientInfo"]["PID"]
                    self.Sex = self.Data["PatientInfo"]["Gender"]
                    self.Race = self.Data["PatientInfo"]["Race"]
                    try:
                        self.BirthDateTime = datetime.date(
                            year=int(self.Data["PatientInfo"]["BirthDateTime"]["Year"]),
                            month=int(self.Data["PatientInfo"]["BirthDateTime"]["Month"]),
                            day=int(self.Data["PatientInfo"]["BirthDateTime"]["Day"]),
                        )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.BirthDateTime = False

                    self.ObservationDateTime = datetime.datetime(
                        year=int(self.Data["ObservationDateTime"]["Year"]),
                        month=int(self.Data["ObservationDateTime"]["Month"]),
                        day=int(self.Data["ObservationDateTime"]["Day"]),
                        hour=int(self.Data["ObservationDateTime"]["Hour"]),
                        minute=int(self.Data["ObservationDateTime"]["Minute"]),
                        second=int(self.Data["ObservationDateTime"]["Second"]),
                    )

                    self.SamplingRate = int(self.StripData["SampleRate"]["#text"])
                    self.NumLeads = int(self.StripData["NumberOfLeads"])
                    self.WaveformData = self.StripData["WaveformData"]

                    self.Segmentations = {}

                    try:
                        self.Segmentations["Pon"] = int(self.Data["RestingECGMeasurements"]["POnset"]["#text"])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.Segmentations["Pon"] = float("NaN")

                    try:
                        self.Segmentations["Poff"] = int(self.Data["RestingECGMeasurements"]["POffset"]["#text"])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.Segmentations["Poff"] = float("NaN")

                    try:
                        self.Segmentations["QRSon"] = int(self.Data["RestingECGMeasurements"]["QOnset"]["#text"])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.Segmentations["QRSon"] = float("NaN")

                    try:
                        self.Segmentations["QRSoff"] = int(self.Data["RestingECGMeasurements"]["QOffset"]["#text"])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.Segmentations["QRSoff"] = float("NaN")

                    try:
                        self.Segmentations["Toff"] = int(self.Data["RestingECGMeasurements"]["TOffset"]["#text"])
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(e)
                        self.Segmentations["Toff"] = False

                self.LeadVoltages = self._makeLeadVoltages()

        # UKBiobnak category 100012, data field 6025
        elif ecg_type == "exercise":
            self.type = "exercise"
            logger.info("Processing exercise ECG XML file")
            with open(path, "rb") as xml:
                self.Data = xmltodict.parse(xml.read().decode(encoding))["CardiologyXML"]

                self.PID = self.Data["PatientInfo"]["PID"]
                self.Sex = self.Data["PatientInfo"]["Gender"]
                self.Race = self.Data["PatientInfo"]["Race"]

                try:
                    self.BirthDateTime = datetime.date(
                        year=int(self.Data["PatientInfo"]["BirthDateTime"]["Year"]),
                        month=int(self.Data["PatientInfo"]["BirthDateTime"]["Month"]),
                        day=int(self.Data["PatientInfo"]["BirthDateTime"]["Day"]),
                    )
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.BirthDateTime = False

                self.ObservationDateTime = datetime.datetime(
                    year=int(self.Data["ObservationDateTime"]["Year"]),
                    month=int(self.Data["ObservationDateTime"]["Month"]),
                    day=int(self.Data["ObservationDateTime"]["Day"]),
                    hour=int(self.Data["ObservationDateTime"]["Hour"]),
                    minute=int(self.Data["ObservationDateTime"]["Minute"]),
                    second=int(self.Data["ObservationDateTime"]["Second"]),
                )

                self.MaxHeartRate = self.Data["ExerciseMeasurements"]["MaxHeartRate"]
                self.MaxPredictedHR = self.Data["ExerciseMeasurements"]["MaxPredictedHR"]
                self.MaxWorkload = self.Data["ExerciseMeasurements"]["MaxWorkload"]  # unit: Watts

                if "FullDisclosure" in self.Data:
                    self.StripData = False
                    self.FullDisclosure = self.Data["FullDisclosure"]
                    try:
                        self.StartTime = self.FullDisclosure["StartTime"]
                        self.StartTime = float(self.StartTime["Minute"]) * 60 + float(self.StartTime["Second"])
                        self.SamplingRate = int(self.FullDisclosure["SampleRate"]["#text"])
                        self.NumLeads = int(self.FullDisclosure["NumberOfChannels"])
                        self.FullDisclosureData = self.FullDisclosure["FullDisclosureData"]
                    except TypeError:
                        # * for some subjects like 1009128, there is error in the field and we cannot extract features
                        raise ValueError("FullDisclosure in exercise XML file is not in correct format.")
 
                self.LeadVoltages = self._makeLeadVoltages()

                voltages = self._divideLeadVoltages()
                self.PretestVoltages = voltages[0]
                self.ExerciseVoltagesConstant = voltages[1]
                self.ExerciseVoltagesRampling = voltages[2]
                self.RestVoltages = voltages[3]
        else:
            raise ValueError("Unknown ECG Type")

    def _makeLeadVoltages(self):
        leads = {}

        if not self.FullDisclosure:
            assert self.type == "rest", "FullDisclosure should be False for rest ECG"
            for lead in self.WaveformData:
                lead_name = lead["@lead"]
                lead_voltages = np.array([int(volt) for volt in lead["#text"].split(",")])
                leads[lead_name] = lead_voltages

        elif not self.StripData:
            assert self.type == "exercise", "StripData should be False for exercise ECG"
            LeadOrder = self.FullDisclosure["LeadOrder"].split(",")

            root = et.parse(self.path).getroot()
            leads = {}
            for lead in LeadOrder:
                leads[lead] = []

            count = 0
            for full_d in root.findall("./FullDisclosure/FullDisclosureData"):
                for full_line in re.split("\n|\t", full_d.text):
                    for sample in re.split(",", full_line):
                        if sample == "":
                            continue
                        lead_idx = (count % (len(LeadOrder) * self.SamplingRate)) // self.SamplingRate
                        leads[LeadOrder[lead_idx]].append(int(sample))
                        count += 1

            for lead in LeadOrder:
                leads[lead] = np.array(leads[lead])

        return leads

    def _divideLeadVoltages(self):
        """
        For ECG during exercise, the ECG can be divided into:
        1. Initial 15 seconds rest (pretest resting ECG)
        2. 2 minute phase at constant power
        3. Linear increase over 4 minutes from Start to Peak power level
        4. Concluded by a 1 minute recovery period
        """
        if self.type == "rest":
            raise ValueError("Cannot divide lead voltages for rest ECG")

        leads_lengths = [len(voltages) for voltages in self.LeadVoltages.values()]
        assert all(
            lead_length == leads_lengths[0] for lead_length in leads_lengths
        ), "All leads should have same length"

        len_phase1 = int((15 - self.StartTime) * self.SamplingRate)
        len_phase2 = 2 * 60 * self.SamplingRate
        len_phase3 = 4 * 60 * self.SamplingRate
        len_phase4 = leads_lengths[0] - len_phase1 - len_phase2 - len_phase3
        if len_phase4 < 0:
            raise ValueError("Cannot determine length of each phase as length of phase 4 is negative.")
        logger.info(
            f"Phase1 has length {len_phase1}, "
            f"Phase2 has length {len_phase2}, "
            f"Phase3 has length {len_phase3}, "
            f"Phase4 has length {len_phase4}"
        )

        leads_phase1 = {}
        leads_phase2 = {}
        leads_phase3 = {}
        leads_phase4 = {}
        for key, voltages in self.LeadVoltages.items():
            leads_phase1[key] = voltages[:len_phase1]
            leads_phase2[key] = voltages[len_phase1 : len_phase1 + len_phase2]
            leads_phase3[key] = voltages[len_phase1 + len_phase2 : len_phase1 + len_phase2 + len_phase3]
            leads_phase4[key] = voltages[(leads_lengths[0] - len_phase4) :]
        return (leads_phase1, leads_phase2, leads_phase3, leads_phase4)

    def getLeads(self):
        """
        Return name of each lead in the ECG XML files.
        """
        return list(self.LeadVoltages.keys())

    def getVoltages(self):
        """
        Return a dict of voltages for each lead.
        """
        voltage_array = {}
        for lead in self.LeadVoltages:
            voltage_array[lead] = np.transpose(self.LeadVoltages[lead][np.newaxis])
            voltage_array[lead] = [voltage for sublist in voltage_array[lead] for voltage in sublist]
        return voltage_array

    def getDividedVoltages(self):
        """
        Return voltages for each phase for exercise ECG XML file.
        """
        if self.type == "rest":
            raise ValueError("Cannot divide lead voltages for rest ECG")

        return (self.PretestVoltages, self.ExerciseVoltagesConstant, self.ExerciseVoltagesRampling, self.RestVoltages)

    def getSegmentations(self):
        return self.Segmentations

    def plotLead(self, lead):
        plt.plot(self.LeadVoltages[lead])
        plt.show()
