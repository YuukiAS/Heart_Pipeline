"""
This script is based on https://github.com/paubrunet97/CardioSoftECGXMLreader
"""
import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging
logger = setup_logging("ECG-XMLReader")

# Two types of XML in the LongQT Dataset:
#   (1) XML containing 'StripData' dictionary, fs=500Hz.
#   (2) XML containing 'FullDisclosure' dictionary, fs=100Hz.
# For each of them, different extraction to get an array of size (samplesize, 12) with .getVoltages() method

class CardioSoftECGXMLReader:
    """ Extract voltage data from a CardioSoftECG XML file """

    def __init__(self, path, encoding="ISO8859-1"):
        with open(path, 'rb') as xml:
            self.Data = xmltodict.parse(xml.read().decode(encoding))['CardiologyXML']

            if 'StripData' in self.Data:
                self.StripData = self.Data['StripData']
                self.FullDisclosure = False

                self.PID = self.Data['PatientInfo']['PID']
                self.Sex = self.Data['PatientInfo']['Gender']
                self.Race = self.Data['PatientInfo']['Race']
                try:
                    self.BirthDateTime = datetime.date(
                        year=int(self.Data['PatientInfo']['BirthDateTime']['Year']),
                        month=int(self.Data['PatientInfo']['BirthDateTime']['Month']),
                        day=int(self.Data['PatientInfo']['BirthDateTime']['Day']))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.BirthDateTime = False

                self.ObservationDateTime = datetime.datetime(
                    year=int(self.Data['ObservationDateTime']['Year']),
                    month=int(self.Data['ObservationDateTime']['Month']),
                    day=int(self.Data['ObservationDateTime']['Day']),
                    hour=int(self.Data['ObservationDateTime']['Hour']),
                    minute=int(self.Data['ObservationDateTime']['Minute']),
                    second=int(self.Data['ObservationDateTime']['Second']))

                self.SamplingRate = int(self.StripData['SampleRate']['#text'])
                self.NumLeads = int(self.StripData['NumberOfLeads'])
                self.WaveformData = self.StripData['WaveformData']

                self.Segmentations = {}

                try:
                    self.Segmentations['Pon'] = int(self.Data['RestingECGMeasurements']['POnset']['#text'])
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.Segmentations['Pon'] = float('NaN')

                try:
                    self.Segmentations['Poff'] = int(self.Data['RestingECGMeasurements']['POffset']['#text'])
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.Segmentations['Poff'] = float('NaN')

                try:
                    self.Segmentations['QRSon'] = int(self.Data['RestingECGMeasurements']['QOnset']['#text'])
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.Segmentations['QRSon'] = float('NaN')

                try:
                    self.Segmentations['QRSoff'] = int(self.Data['RestingECGMeasurements']['QOffset']['#text'])
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.Segmentations['QRSoff'] = float('NaN')

                try:
                    self.Segmentations['Toff'] = int(self.Data['RestingECGMeasurements']['TOffset']['#text'])
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.Segmentations['Toff'] = False


            elif 'FullDisclosure' in self.Data:
                self.StripData = False
                self.FullDisclosure = self.Data['FullDisclosure']

                self.PID= self.Data['PatientInfo']['PID']
                self.Sex = self.Data['PatientInfo']['Gender']
                self.Race = self.Data['PatientInfo']['Race']
                try:
                    self.BirthDateTime = datetime.date(
                        year=int(self.Data['PatientInfo']['BirthDateTime']['Year']),
                        month=int(self.Data['PatientInfo']['BirthDateTime']['Month']),
                        day=int(self.Data['PatientInfo']['BirthDateTime']['Day']))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(e)
                    self.BirthDateTime = False
                self.ObservationDateTime = datetime.datetime(
                    year=int(self.Data['ObservationDateTime']['Year']),
                    month=int(self.Data['ObservationDateTime']['Month']),
                    day=int(self.Data['ObservationDateTime']['Day']),
                    hour=int(self.Data['ObservationDateTime']['Hour']),
                    minute=int(self.Data['ObservationDateTime']['Minute']),
                    second=int(self.Data['ObservationDateTime']['Second']))

                self.SamplingRate = int(self.FullDisclosure['SampleRate']['#text'])
                self.NumLeads = int(self.FullDisclosure['NumberOfChannels'])
                self.FullDisclosureData = self.FullDisclosure['FullDisclosureData']['#text']
                self.Segmentations = False


            self.LeadVoltages = self.makeLeadVoltages()

    def makeLeadVoltages(self):

        leads = {}

        if not self.FullDisclosure:
            for lead in self.WaveformData:
                lead_name = lead['@lead']
                lead_voltages = np.array([int(volt) for volt in lead['#text'].split(',')])
                leads[lead_name] = lead_voltages

        elif not self.StripData:
            voltages_str = self.FullDisclosureData.split(',')

            voltage_lines = []
            voltage_line = []
            for volt in voltages_str:
                if '\n' in volt:
                    voltage_lines.append(voltage_line)
                    voltage_line = []
                    voltage_line.append(int(volt))

                elif volt == '':
                    voltage_lines.append(voltage_line)

                else:
                    voltage_line.append(int(volt))

            LeadOrder = self.FullDisclosure['LeadOrder'].split(',')

            for lead_name in LeadOrder:
                leads[lead_name] = []

            for lead_num in np.arange(0, self.NumLeads):
                for i in np.arange(lead_num, len(voltage_lines), self.NumLeads):
                    leads[LeadOrder[lead_num]] = leads[LeadOrder[lead_num]] + voltage_lines[i]
                leads[LeadOrder[lead_num]] = np.array(leads[LeadOrder[lead_num]])

        return leads

    def getLeads(self):
        return list(self.LeadVoltages.keys())

    def getVoltages(self):
        voltage_array = {}
        for lead in self.LeadVoltages:
            voltage_array[lead] = np.transpose(self.LeadVoltages[lead][np.newaxis])
            voltage_array[lead] = [voltage for sublist in voltage_array[lead] for voltage in sublist]
        return voltage_array
    
    def getSegmentations(self):
        return self.Segmentations

    def plotLead(self, lead):
        plt.plot(self.LeadVoltages[lead])
        plt.show()
