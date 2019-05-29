from abc import abstractmethod
import abc
import pandas as pd

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()}) 

class Data(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def addFeature(self, feature_name_list, dates, inputs):
        """
        This is a method to add features to dataframe .

        @param feature_name_list: list of the name of features
        @param dates: dates of dataset
        @return: None
        @raise keyError: raises an exception
        """
        pass

    @abstractmethod
    def appendDate(self,dates):
        """
        This is a method to append dates to dataframe .

        @param dates: dates of dataset
        @return: None
        @raise keyError: raises an exception
        """
        pass

    @abstractmethod
    def createDateFrame(self,dates,datalist,tags):
        """
        This is a method to create dataframe .

        @param dates: dates of dataset
        @param datalist: dataset(1D or 2D array)
        @param tags:  list of the name of the data in dataset
        @return: None
        @raise keyError: raises an exception
        """
        pass

    @abstractmethod
    def getInputMatrix(self):
        """
        This is a method to create input matrix .
        @return: A list(1D or 2D)

        """
        pass
