import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import find_peaks
import matplotlib.transforms as mtransforms
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'


class Spectrum():
    def __init__(self, spectrum = None):
        self.spectrum = spectrum

    def load_txt(self, path=""):
        if path is not "":
            self.path = path
        try: 
            self.spectrum = spectrum = np.loadtxt()
            self.x, self.y = self.spectrum[:, 0], self.spectrum[:,1]

        except Exception as e:
            print(f"An error occurred: {e}")    

        
    def plot_spectrum(self, spectrum = None,
                  findpeaks = True,
                  threshold = 0.01,
                  distance = 3,
                  matchpeaks = True,
                  plot_matched = False,
                  ythresh = None,
                  matched_dct={},
                  dfs_matched = [],
                  x_var = ""):
        
        if spectrum is not None:
            self.spectrum = spectrum


            
        pass
    
    def pick_peaks(self):
        pass

    def match_peaks(self):
        pass

class Spectra():
    def __init__(self, spectra):
        self.spectra = spectra
        pass

    
    


    
