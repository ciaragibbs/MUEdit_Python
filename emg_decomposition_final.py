import glob, os, scipy
import xml.etree.ElementTree as ET
import tarfile as tf
import numpy as np
import matplotlib.pyplot as plt
from processing_tools import *
import tkinter as tk
from tkinter import simpledialog
from scipy import signal
import pickle
import time
root = tk.Tk() # Initialising for GUI
np.random.seed(1337) # Fixes random generation to get same results each time the script is run

class EMG():

    def __init__(self):
        self.its = 20 # number of iterations of the fixed point algorithm 
        self.ref_exist = 1 # if ref_signal exist ref_exist = 1; if not ref_exist = 0 and manual selection of windows
        self.windows = 1  # number of segmented windows over each contraction
        self.check_emg = 1 # 0 = Automatic selection of EMG channels (remove 5% of channels) ; 1 = Visual checking
        self.drawing_mode = 1 # 0 = Output in the command window ; 1 = Output in a figure
        self.differential_mode = 0 # 0 = no; 1 = yes (filter out the smallest MU, can improve decomposition at the highest intensities
        self.peel_off = 0 # 0 = no; 1 = yes (update the residual EMG by removing the motor units with the highest SIL value
        self.sil_thr = 0.9 # Threshold for SIL values when discarding MUs after two fastICA phases
        self.silthrpeeloff = 0.9 # Threshold for MU removed from the signal (if the sparse  deflation is on)
        self.ext_factor = 1000 # extension of observations for numerical stability 
        self.edges2remove = 0.2 # Extent of signal clipping after whitening 
        self.target_thres = 0.8  # Threshold for segmenting and batching the EMG signals based on a target force profile
        self.initialisation = 0 # 0 = initialisation based on the a maximum value in the EMG signal, 1 = random initialisaiton
        self.cov_thr = 0.5 # Threshold for CoV values when discarding MUs after two fastICA phases
        self.cov_filter = 1
        self.dup_thr = 0.3 # Correlation threshold for defining a pair of spike trains as derived from the same MU, hence a duplicate
        self.refine_mu = 1
        self.dup_bgrids = 0

#######################################################################################################
########################################## OFFLINE EMG ################################################
#######################################################################################################

class offline_EMG(EMG):

    # child class of EMG, so will inherit it's initialisaiton
    def __init__(self, save_dir, to_filter):
        super().__init__()
        self.save_dir = save_dir # directory at which final discharges will be saved
        self.to_filter = to_filter # whether or not you notch and butter filter the 

    def modify_electrode_names(self,electrode_names, muscle_names, cutoff):
        root = tk.Tk()
        root.withdraw()  # Hides the main window

        index = 0
        while index < cutoff:
            element = electrode_names[index]
            # Ask the user if they want to change or remove the current element
            response = simpledialog.askstring("Update or Remove", f"Do you want to change or remove this element: {element}?\nType your new value to change, 'REMOVE' to remove, 'KEEP' to keep it.")

            if response is None:  # User cancelled the dialog
                index += 1
                continue

            if response.lower() == 'remove':
                # If the user types 'REMOVE', remove the current element
                electrode_names.pop(index)
                muscle_names.pop(index)
                # Don't increment index, as the next element shifts into the current index
            elif response.lower() == 'keep':
                # If the user just presses 'OK' without typing anything, keep the element
                index += 1
            else:
                # If the user types a new value, update the element in place
                electrode_names[index] = response
                index += 1

        root.destroy()  # Clean up the root window after finishing
        return electrode_names, muscle_names

    
    def open_otb(self, inputfile):

        print(inputfile)
        file_name = inputfile.split('/')[1]
        temp_dir = os.path.join(self.save_dir, 'temp_tarholder')

        # make a temporary directory to store the data of the otb file if it doesn't exist yet
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

        # Open the .tar file and extract all data
        with tf.open(inputfile, 'r') as emg_tar:
            emg_tar.extractall(temp_dir)

        #os.chdir(temp_dir)
        sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
        trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
        trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
        trial_label_sig = os.path.join(temp_dir, trial_label_sig)
        trial_label_xml = os.path.join(temp_dir, trial_label_xml)

        # read the contents of the trial xml file
        with open(trial_label_xml, encoding='utf-8') as file:
            xml=ET.fromstring(file.read())

        # get sampling frequency, no. bits of AD converter, no. channels, electrode names and muscle names
        fsamp = int(xml.find('.').attrib['SampleFrequency'])
        nADbit = int(xml.find('.').attrib['ad_bits'])
        nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
        electrode_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
        muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
        # here, add a dialog to ask to ammend the name or remove it?
        index_of_first_aux = next((i for i, s in enumerate(electrode_names) if 'AUX' in s), None)
        print(index_of_first_aux)
        electrode_names, muscle_names = self.modify_electrode_names(electrode_names, muscle_names, index_of_first_aux)
       
        # count the number of electrodes
        ngrids = sum(1 for name in electrode_names if name.startswith('GR'))
        nneedles = len(electrode_names) - (sum(1 for name in electrode_names if any(substring.lower() in name.lower() for substring in ['aux', 'ramp', 'buffer'])) +  ngrids) 
        nelectrodes = nneedles + ngrids

        # read in the EMG trial data
        emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
        emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
        emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division

        # convert the data from bits to microvolts
        for i in range(nchans):
            emg_data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids, nneedles = nneedles, nelectrodes = nelectrodes, electrodes = electrode_names[:nneedles+ngrids],muscles = muscle_names[:nneedles+ngrids]) # discard the other muscle and grid entries, not relevant

        # if the signals were recorded with a feedback generated by OTBiolab+, get the target and the path performed by the participant
        if self.ref_exist:

            # only opening the last two .sip files because the first is not needed for analysis
            # would only need MSE between the participant path (file 2) and the target path (file 3)
            ######## path #########
            _ , path_label, target_label = sorted(glob.glob(f'{temp_dir}/*.sip'))

            with open(path_label) as file:
                path = np.fromfile(file, dtype='float64')
                path = path[:np.shape(emg_data)[1]]
            ######## target ########
            with open(target_label) as file:
                target = np.fromfile(file,dtype='float64')
                target = target[:np.shape(emg_data)[1]]
            
            signal['path'] = path
            signal['target'] = target
        
        # delete the temp_tarholder directory since everything we need has been taken out of it
        for file_name in os.listdir(temp_dir):
            file = os.path.join(temp_dir, file_name)
            if os.path.isfile(file):
                os.remove(file)

        os.rmdir(temp_dir)

        self.signal_dict = signal
        self.decomp_dict = {} # initialising this dictionary here for later use
        self.mu_dict = dict(pulse_trains = [], discharge_times = [[] for item in range(1)])# initialising a dictionary that is an empty nested list

        return

        
    def electrode_formatter(self):

        """ Match up the signals with the electrode shape and numbering """

        electrode_names = self.signal_dict['electrodes']
        ElChannelMap = []
        coordinates = []
        self.emgopt = []
        IED = []
        c_map = []
        r_map = []
        rejected_channels = []
        chans_per_electrode = []
        self.signal_dict['filtered_data'] = np.zeros([np.shape(self.signal_dict['data'])[0],np.shape(self.signal_dict['data'])[1]])
        

        for i in range(self.signal_dict['nelectrodes']):
            
            # print(grid_names[i])
            if electrode_names[i] == 'GR04MM1305':

                ElChannelMap.append([[0, 24, 25, 50, 51], 
                        [0, 23, 26, 49, 52], 
                        [1, 22, 27, 48, 53], 
                        [2, 21, 28, 47, 54], 
                        [3, 20, 29, 46, 55], 
                        [4, 19, 30, 45, 56], 
                        [5, 18, 31, 44, 57], 
                        [6, 17, 32, 43, 58],  
                        [7, 16, 33, 42, 59], 
                        [8, 15, 34, 41, 60],  
                        [9, 14, 35, 40, 61], 
                        [10, 13, 36, 39, 62], 
                        [11, 12, 37, 38, 63]])
                
                rejected_channels.append(np.zeros([65]))
                IED.append(4)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1])-1)
                self.emgopt.append('surface')

            elif electrode_names[i] == 'ELSCH064NM2':

                ElChannelMap.append([[0, 0, 1, 2, 3],
                        [15, 7, 6, 5, 4],
                        [14, 13, 12, 11, 10],
                        [18, 17, 16, 8, 9],
                        [19, 20, 21, 22, 23],
                        [27, 28, 29, 30, 31],
                        [24, 25, 26, 32, 33],
                        [34, 35, 36, 37, 38],
                        [44, 45, 46, 47, 39],
                        [43, 42, 41, 40, 38],
                        [53, 52, 51, 50, 49],
                        [54, 55, 63, 62, 61],
                        [56, 57, 58, 59, 60]])

                rejected_channels.append(np.zeros([65]))
                IED.append(8)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1])-1)
                self.emgopt.append('surface')

            elif electrode_names[i] == 'GR08MM1305':

                ElChannelMap.append([[0, 24, 25, 50, 51], 
                    [0, 23, 26, 49, 52], 
                    [1, 22, 27, 48, 53], 
                    [2, 21, 28, 47, 54], 
                    [3, 20, 29, 46, 55], 
                    [4, 19, 30, 45, 56], 
                    [5, 18, 31, 44, 57], 
                    [6, 17, 32, 43, 58],  
                    [7, 16, 33, 42, 59], 
                    [8, 15, 34, 41, 60],  
                    [9, 14, 35, 40, 61], 
                    [10, 13, 36, 39, 62], 
                    [11, 12, 37, 38, 63]])
              

                
                rejected_channels.append(np.zeros([65]))
                IED.append(8)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1])-1)
                self.emgopt.append('surface')

            elif electrode_names[i] == 'GR10MM0808':

                ElChannelMap.append([[7, 15, 23, 31, 39, 47, 55, 63],
                        [6, 14, 22, 30, 38, 46, 54, 62],
                        [5, 13, 21, 29, 37, 45, 53, 61],
                        [4, 12, 20, 28, 36, 44, 52, 60],
                        [3, 11, 19, 27, 35, 43, 51, 59],
                        [2, 10, 18, 26, 34, 42, 50, 58],
                        [1, 9, 17, 25, 33, 41, 49, 57],
                        [0, 8, 16, 24, 32, 40, 48, 56]])

                rejected_channels.append(np.zeros([65]))
                IED.append(10)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1])-1)
                self.emgopt.append('surface')


            elif electrode_names[i] == 'other':

                # TO DO: match up to the relevant configuration for a thin-film
                ElChannelMap.append([[0, 24, 25, 50, 51], 
                        [0, 23, 26, 49, 52], 
                        [1, 22, 27, 48, 53], 
                        [2, 21, 28, 47, 54], 
                        [3, 20, 29, 46, 55], 
                        [4, 19, 30, 45, 56], 
                        [5, 18, 31, 44, 57], 
                        [6, 17, 32, 43, 58],  
                        [7, 16, 33, 42, 59], 
                        [8, 15, 34, 41, 60],  
                        [9, 14, 35, 40, 61], 
                        [10, 13, 36, 39, 62], 
                        [11, 12, 37, 38, 63]])
                
                rejected_channels.append(np.zeros([65]))
                IED.append(1)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1])-1)
                self.emgopt.append('intra')

            elif electrode_names[i] == 'Thin film':
                
                ElChannelMap.append([[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 23, 33],
                        [4, 14, 24, 34],
                        [5, 15, 25, 35],
                        [6, 16, 26, 36],
                        [7, 17, 27, 37],
                        [8, 18, 28, 38],
                        [9, 19, 29, 39]])

                rejected_channels.apppend(np.zeros([40]))
                IED.append(1)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1]))
                self.emgopt.append('intra')

            elif electrode_names[i] == '4-wire needle':

                ElChannelMap.append([[0, 8],
                                    [1, 9],
                                    [2, 10],
                                    [3, 11],
                                    [4, 12],
                                    [5, 13],
                                    [6, 14],
                                    [7, 15]])
                
                rejected_channels.apppend(np.zeros([16]))
                IED.append(4)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1]))
                self.emgopt.append('intra')
                    

            elif electrode_names[i] == 'Myomatrix Monopolar':

                ElChannelMap.append([[0, 8, 16, 24],
                                    [1, 9, 17, 25],
                                    [2, 10, 18, 26],
                                    [3, 11, 19, 27],
                                    [4, 12, 20, 28],
                                    [5, 13, 21, 29],
                                    [6, 14, 22, 30],
                                    [7, 15, 23, 31]])
                
                rejected_channels.apppend(np.zeros([16]))
                IED.append(4)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1]))
                self.emgopt.append('intra')

            else:

                # assume that it is some variation of an intramusuclar array
                ElChannelMap.append([[0, 8],
                                    [1, 9],
                                    [2, 10],
                                    [3, 11],
                                    [4, 12],
                                    [5, 13],
                                    [6, 14],
                                    [7, 15]])
                
                rejected_channels.apppend(np.zeros([16]))
                IED.append(4)
                ElChannelMap[i] = np.squeeze(np.array(ElChannelMap[i]))
                chans_per_electrode.append((np.shape(ElChannelMap[i])[0] * np.shape(ElChannelMap[i])[1]))
                self.emgopt.append('intra')


                      
            coordinates.append(np.zeros([chans_per_electrode[i],2]))
            rows, cols = ElChannelMap[i].shape
            row_indices, col_indices = np.unravel_index(np.arange(ElChannelMap[i].size), (rows, cols))
            coordinates[i][:, 0] = row_indices[1:]
            coordinates[i][:, 1] = col_indices[1:]

          
            c_map.append(np.shape(ElChannelMap[i])[1])
            r_map.append(np.shape(ElChannelMap[i])[0])
            
            electrode = i + 1 
            
            # notch filtering
            self.signal_dict['filtered_data'][chans_per_electrode[i]*(electrode-1):electrode*chans_per_electrode[i],:] = notch_filter(self.signal_dict['data'][chans_per_electrode[i]*(electrode-1):electrode*chans_per_electrode[i],:],self.signal_dict['fsamp'])
            # bandpass filtering
            print(self.emgopt)
            self.signal_dict['filtered_data'][chans_per_electrode[i]*(electrode-1):electrode*chans_per_electrode[i],:] = bandpass_filter(self.signal_dict['filtered_data'][chans_per_electrode[i]*(electrode-1):electrode*chans_per_electrode[i],:],self.signal_dict['fsamp'],emg_type = self.emgopt[i])        
        
        self.c_maps = c_map
        self.r_maps = r_map
        self.rejected_channels = rejected_channels
        self.ied = IED
        self.chans_per_electrode = chans_per_electrode
        self.coordinates = coordinates
        
        
        print('Preprocessing i.e. formatting of the emg data structure is complete')
                    
    def manual_rejection(self):

        """ Manual rejection for channels with noise/artificats by inspecting plots of the electrode channels """
        for i in range(self.signal_dict['nelectrodes']):

            electrode = i + 1
            sig2inspect = self.signal_dict['filtered_data'][self.chans_per_electrode[i]*(electrode-1):electrode*self.chans_per_electrode[i],:]

            for c in range(self.c_maps[i]):
                for r in range(self.r_maps[i]):

                    num_chans2reject = []
                    if (c+r) > 0: # TO-DO: remove the assumption of the left corner channel being invalid
                        plt.ion()
                        plt.plot(sig2inspect[(c*self.r_maps[i])+r-1,:]/max(sig2inspect[(c*self.r_maps[i])+r-1,:])+r+1)
                plt.show()
                inputchannels = simpledialog.askstring(title="Channel Rejection",
                                  prompt="Please enter channel numbers to be rejected (1-13), input with spaces between numbers:")
                plt.clf(); plt.ioff()
                print("The selected channels for rejection are:", inputchannels)

                if inputchannels:
                    str_chans2reject = inputchannels.split(" ")
                    for j in range(len(str_chans2reject)):

                        num_chans2reject.append(int(str_chans2reject[j])+c*self.r_maps[i]-1)

                    self.rejected_channels[i][num_chans2reject] =  1
        
            self.rejected_channels[i] = self.rejected_channels[i][1:] # get rid of the irrelevant top LHC channel
      
    def batch_w_target(self):

        plateau = np.where(self.signal_dict['target'] >= max(self.signal_dict['target'])*self.target_thres)[0] # finding where the plateau is
        discontinuity = np.where(np.diff(plateau) > 1)[0]
        print(discontinuity)
        
        if self.windows > 1 and not discontinuity: 
        
            plat_len = plateau[-1] - plateau[0]
            wind_len = np.floor(plat_len/self.windows)
            batch = np.zeros(self.windows*2)
            for i in range(self.windows):

                batch[i*2] = plateau[0] + i*wind_len + 1
                batch [(i+1)*2-1] = plateau[0] + (i+1)*wind_len

            self.plateau_coords = batch
            
        elif self.windows >= 1 and discontinuity.any(): 
           
            prebatch = np.zeros([len(discontinuity)+1,2])
            
            prebatch[0,:] = [plateau[0],plateau[discontinuity[0]]]
            n = len(discontinuity)
            for i, d in enumerate(discontinuity):
                if i < n - 1:
                    prebatch[i+1,:] = [plateau[d+1],plateau[discontinuity[i+1]]]
                else:
                    prebatch[i+1,:] = [plateau[d+1],plateau[-1]]

            plat_len = prebatch[:,-1] - prebatch[:,0]
            wind_len = np.floor(plat_len/self.windows)
            batch = np.zeros([len(discontinuity)+1,self.windows*2])
            
            for i in range(self.windows):
                
                batch[:,i*2] = prebatch[:,0] + i*wind_len +1
                batch[:,(i+1)*2-1] = prebatch[:,0] + (i+1)*wind_len

            batch = np.sort(batch.reshape([1, np.shape(batch)[0]*np.shape(batch)[1]]))
            self.plateau_coords = batch
            
        else:
            # the last option is having only one window and no discontinuity in the plateau; in that case, you leave as is
            batch = [plateau[0],plateau[-1]]
            self.plateau_coords = batch
        
        # with the markers for windows and plateau discontinuities, batch the emg data ready for decomposition
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['nelectrodes'] * n_intervals)
        
        for i in range(int(self.signal_dict['nelectrodes'])):
            
            electrode = i + 1
            for interval in range(n_intervals):
                
                data_slice = self.signal_dict['data'][self.chans_per_electrode[i]*(electrode-1):electrode*self.chans_per_electrode[i], int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i] == 1
                # Remove rejected channels
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1

        self.signal_dict['batched_data'] = batched_data

        print('Data batched across all surface and intramusuclar arrays')
            

    def batch_wo_target(self):

        fake_ref = np.zeros([np.shape(self.signal_dict['data'])][1])
        plt.figure(figsize=(10,8))
        plt.plot(self.signal_dict['data'][0,:])
        plt.grid()
        plt.show()
        window_clicks = plt.ginput(2*self.windows, show_clicks = True)
        self.plateau_coords = np.zeros([1,self.windows *2])
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['nelectrodes'] * n_intervals) 

        for interval in range(self.windows):
            
            self.plateau_coords[interval*2] = np.floor(window_clicks[interval*2][0])
            self.plateau_coords[(interval+1)*2-1] = np.floor(window_clicks[(interval+1)*2-1][0])
        

        for i in range(int(self.signal_dict['nelectrodes'])):

            electrode = i + 1
            for interval in range(n_intervals):
                
                data_slice = self.signal_dict['data'][self.chans_per_electrode[i]*(electrode-1):electrode*self.chans_per_electrode[i], int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i] == 1
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1
        
        self.signal_dict['batched_data'] = batched_data



  
################################ CONVOLUTIVE SPHERING ########################################
    def convul_sphering(self,g,interval,tracker):

        # g is the index for the electrodes
        """ 1) Filter the batched EMG data 2) Extend to improve speed of convergence/reduce numerical instability 3) Remove any DC component  4) Whiten """
        chans_per_electrode = self.chans_per_electrode
        electrode = g+1

        print(self.emgopt)
        if self.to_filter: # adding since will need to avoid this step if doing real-time decomposition + biofeedback, but fine for offline analysis

            self.signal_dict['batched_data'][tracker]= notch_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'])
            self.signal_dict['batched_data'][tracker] = bandpass_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'],emg_type = self.emgopt[g])  

        # differentiation - typical EMG generation model treats low amplitude spikes/MUs as noise, which is common across channels so can be cancelled with a first order difference. Useful for high intensities - where cross talk has biggest impact.
        if self.differential_mode: # just a basic 1st order differential (bipolar processing)
           
            self.signal_dict['batched_data'][tracker]= []
            self.signal_dict['batched_data'][tracker]= np.diff(self.signal_dict['batched_data'][tracker],n=1,axis=-1)

        # signal extension - increasing the number of channels to 1000
        # Holobar 2007 -  Multichannel Blind Source Separation using Convolutive Kernel Compensation (describes matrix extension)
        extension_factor = int(np.round(self.ext_factor/len(self.signal_dict['batched_data'][tracker])))
        self.ext_number =  extension_factor
        self.signal_dict['extend_obvs_old'][interval] = extend_emg(self.signal_dict['extend_obvs_old'][interval], self.signal_dict['batched_data'][tracker], extension_factor)
        self.signal_dict['sq_extend_obvs'][interval] = (self.signal_dict['extend_obvs_old'][interval] @ self.signal_dict['extend_obvs_old'][interval].T) / np.shape(self.signal_dict['extend_obvs_old'][interval])[1]
        self.signal_dict['inv_extend_obvs'][interval] = np.linalg.pinv(self.signal_dict['sq_extend_obvs'][interval]) # different method of pinv in MATLAB --> SVD vs QR
        
        # de-mean the extended emg observation matrix
        self.signal_dict['extend_obvs_old'][interval] = scipy.signal.detrend(self.signal_dict['extend_obvs_old'][interval], axis=- 1, type='constant', bp=0)
        
        # whiten the signal + impose whitened extended observation matrix has a covariance matrix equal to the identity for time lag zero
        self.decomp_dict['whitened_obvs_old'][interval],self.decomp_dict['whiten_mat'][interval], self.decomp_dict['dewhiten_mat'][interval] = whiten_emg(self.signal_dict['extend_obvs_old'][interval])
        
        # remove the edges
        self.signal_dict['extend_obvs'][interval] = self.signal_dict['extend_obvs_old'][interval][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)-1):-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        self.decomp_dict['whitened_obvs'][interval] = self.decomp_dict['whitened_obvs_old'][interval][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)-1):-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        
        if g == 0: # don't need to repeat for every electrode, since the path and target info (informing the batches), is the same for all electrodes
            self.plateau_coords[interval*2] = self.plateau_coords[interval*2]  + int(np.round(self.signal_dict['fsamp']*self.edges2remove)) - 1
            self.plateau_coords[(interval+1)*2 - 1] = self.plateau_coords[(interval+1)*2-1]  - int(np.round(self.signal_dict['fsamp']*self.edges2remove))

        print('Signal extension and whitening complete')
        
######################### FAST ICA AND CONVOLUTIVE KERNEL COMPENSATION  ############################################

    def fast_ICA_and_CKC(self,g,interval,tracker,cf_type = 'square'):

        
        init_its = np.zeros([self.its],dtype=int) # tracker of initialisaitons of separation vectors across iterations
        fpa_its = 500 # maximum number of iterations for the fixed point algorithm
       
       
        ####### TESTING WITH A RANDOM (SEEDED) DATA MATRIX ###########

        # random test to compare to matlab
        # self.decomp_dict['whitened_obvs'][interval] = np.random.random((np.shape(self.decomp_dict['whitened_obvs'][interval])[1],np.shape(self.decomp_dict['whitened_obvs'][interval])[0])).T

        Z = np.array(self.decomp_dict['whitened_obvs'][interval]).copy() # copy of the whitened signal, that will be modified through fast ICA
        time_axis = np.linspace(0,np.shape(Z)[1],np.shape(Z)[1])/self.signal_dict['fsamp']  # create a time axis for spiking activity

        # choosing contrast function here, avoid repetitively choosing within the iteration loop
        if cf_type == 'square':
            cf = square
            dot_cf = dot_square
        elif cf_type == 'skew':
            cf = skew
            dot_cf = dot_skew
        elif cf_type == 'exp':
            cf = exp
            dot_cf = dot_exp
        elif cf_type == 'logcosh':
            cf = logcosh
            dot_cf = dot_logcosh
      
        for i in range(self.its):

                #################### FIXED POINT ALGORITHM #################################

                if self.initialisation:
                    # generate a random vector
                    random_init = np.random.randn(np.size(self.decomp_dict['whitened_obvs'][interval])[0],np.size(self.decomp_dict['whitened_obvs'][interval])[0]) # dimension extended channels x extended channels
                    self.decomp_dict['w_sep_vect'] = random_init[:,0]
                else:
                    if i == 0 :
                        # identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
                         # occurs. Then, the projection vector is initialised to the whitened observation vector, at this located time instant.
                        sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 0)))

                    init_its[i] = sort_sq_sum_Z[-(i+1)] # since the indexing starts at -1 the other way (for ascending order list)
                    self.decomp_dict['w_sep_vect'] = Z[:,int(init_its[i])].copy() # retrieve the corresponding signal value to initialise the separation vector
                
                # orthogonalise separation vector before fixed point algorithm
                self.decomp_dict['w_sep_vect'] -= np.dot(self.decomp_dict['B_sep_mat'] @ self.decomp_dict['B_sep_mat'].T, self.decomp_dict['w_sep_vect'])
               
                # normalise separation vector before fixed point algorithm 
                self.decomp_dict['w_sep_vect'] /= np.linalg.norm(self.decomp_dict['w_sep_vect'])
            
                # use the fixed point algorithm to identify consecutive separation vectors
                self.decomp_dict['w_sep_vect'] = fixed_point_alg(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'],Z, cf, dot_cf,fpa_its)
                
                # get the first iteration of spikes using k means ++
                fICA_source, spikes = get_spikes(self.decomp_dict['w_sep_vect'],Z, self.signal_dict['fsamp'])
            
                ################# MINIMISATION OF COV OF DISCHARGES ############################
                if len(spikes) > 1:

                    # determine the interspike interval
                    ISI = np.diff(spikes/self.signal_dict['fsamp'])
                    # determine the coefficient of variation
                    CoV = np.std(ISI)/np.mean(ISI)
                    # update the sepearation vector by summing all the spikes
                    w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
                    # minimisation of covariance of interspike intervals
                    self.decomp_dict['MU_filters'][interval][:,i], spikes, self.decomp_dict['CoVs'][interval,i] = min_cov_isi(w_n_p1,self.decomp_dict['B_sep_mat'],Z, self.signal_dict['fsamp'],CoV,spikes)
                    self.decomp_dict['B_sep_mat'][:,i] = (self.decomp_dict['w_sep_vect']).real # no need to shallow copy here

                    # calculate SIL
                    fICA_source, spikes, self.decomp_dict['SILs'][interval,i] = get_silohuette(self.decomp_dict['MU_filters'][interval][:,i],Z,self.signal_dict['fsamp'])
                    # peel off
                    if self.peel_off == 1 and self.decomp_dict['SILs'][interval,i] > self.sil_thr:
                        Z = peel_off(Z, spikes, self.signal_dict['fsamp'])


                    print(self.decomp_dict['SILs'][interval,i])
            
                    if self.drawing_mode == 1:
                        plt.clf()
                        plt.ion()
                        plt.show()
                        plt.subplot(2, 1, 1)
                        plt.plot(self.signal_dict['target'], 'k--', linewidth=2)
                        plt.plot([self.plateau_coords[interval*2], self.plateau_coords[interval*2]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.plot([self.plateau_coords[(interval+1)*2 - 1], self.plateau_coords[(interval+1)*2 - 1]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.title('Electrode #{} - Iteration #{} - Sil = {}'.format(g, i+1, self.decomp_dict['SILs'][interval,i]))
                        plt.subplot(2, 1, 2)
                        plt.plot(time_axis, fICA_source,linewidth = 0.5)
                        plt.plot(time_axis[spikes],fICA_source[spikes],'o')
                        plt.grid()
                        plt.draw()
                        plt.pause(1e-6)
                    else:
                        print('Electrode #{} - Iteration #{} - Sil = {} - CoV = {}'.format(g+1, i, self.decomp_dict['SILs'][interval,i],self.decomp_dict['CoVs'][interval,i]))

                else:
                    print('Electrode #{} - Iteration #{} - less than 10 spikes '.format(g, i))
                    # without enough spikes, we skip minimising the covariation of discharges to improve the separation vector
                    self.decomp_dict['B_sep_mat'][:,i] = self.decomp_dict['w_sep_vect'].real  # no need to shallow copy here

        ####################################### MU FILTER THRESHOLDING ###############################################

        # remove the MU filters that fall below the imposed metric thresholds
        SIL_condition = self.decomp_dict['SILs'][interval,:] >= self.sil_thr
        final_condition = SIL_condition.copy()

        if self.cov_filter:
            # mask combines CoV and SIL threshold crtieria
            CoV_condition = self.decomp_dict['CoVs'][interval,:] <= self.cov_thr
            final_condition = SIL_condition & CoV_condition


        mask = np.broadcast_to(final_condition.reshape(1, -1), (np.shape(self.decomp_dict['whitened_obvs'][interval])[0], self.its))

        self.decomp_dict['masked_mu_filters'].append(self.decomp_dict['MU_filters'][interval][mask].reshape(np.shape(self.decomp_dict['whitened_obvs'][interval])[0], np.sum(mask, axis=1)[0]))

    
        plt.close() #closes the fixed point algorithm plots
        
################################################## POST PROCESSING #######################################################

    def post_process_EMG(self,electrode):

        self.mus_in_array = np.zeros(self.signal_dict['nelectrodes'])
        electrode += 1
        # batch processing over each window
        pulse_trains, discharge_times = batch_process_filters(self.decomp_dict['whitened_obvs'],self.decomp_dict['masked_mu_filters'],self.plateau_coords,self.ext_number,self.differential_mode,np.shape(self.signal_dict['data'])[1],self.signal_dict['fsamp'])

        
        if np.shape(pulse_trains)[0] > 0: # if there are existing MUs
            
            self.mus_in_array[electrode-1] = 1 # if pulse trains were not extracted for this electrode, then it remains at a value of 0
            # removing duplicate MUs
            discharge_times_new, pulse_trains_new, mu_filters_new =  remove_duplicates(pulse_trains, discharge_times,discharge_times,np.squeeze(self.decomp_dict['masked_mu_filters']),np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])
            self.decomp_dict['masked_mu_filters'] = []
            self.decomp_dict['masked_mu_filters'] = mu_filters_new

            if self.refine_mu:
                # removing outliers generating irrelvant discharge rates before manual edition (1st time)
                discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'], self.cov_thr)
                
                pulse_trains_new, discharge_times_new = refine_mus(self.signal_dict['data'][self.chans_per_electrode[electrode-1]*(electrode-1):electrode*self.chans_per_electrode[electrode-1],:], self.rejected_channels[electrode-1], pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'])

                discharge_times_new = remove_outliers(pulse_trains_new, discharge_times_new, self.signal_dict['fsamp'], self.cov_thr)
            
            self.mu_dict['pulse_trains'].append(pulse_trains_new)
            
        if electrode != 1:
            self.mu_dict['discharge_times'].append([])

        for j in range(len(discharge_times_new)):

            self.mu_dict['discharge_times'][electrode-1].append(discharge_times_new[j])


    def post_process_EMG_for_biofeedback(self,electrode,interval):

        self.mus_in_array = np.zeros(self.signal_dict['nelectrodes'])
        electrode += 1
        self.decomp_dict['masked_mu_filters'] =  self.decomp_dict['dewhiten_mat'][interval] @ self.decomp_dict['masked_mu_filters']
        # get the pulse train for the entire signal

        pulse_trains, discharge_times, _ = get_pulse_trains(self.signal_dict['data'], self.rejected_channels, self.decomp_dict['masked_mu_filters'], self.chans_per_electrode, self.signal_dict['fsamp'],electrode-1)
      
        if np.shape(pulse_trains)[0] > 0: # if there are existing MUs
            
            self.mus_in_array[electrode-1] = 1 # if pulse trains were not extracted for this electrode, then it remains at a value of 0
           
            # removing duplicate MUs
            discharge_times_new, _ , _ =  remove_duplicates(pulse_trains, discharge_times,discharge_times,np.squeeze(self.decomp_dict['masked_mu_filters']),np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])
            del pulse_trains, discharge_times

            # get the decomposition parameters for the biofeedback
            new_mu_filters = get_mu_filters(self.signal_dict['data'],self.rejected_channels, discharge_times_new, self.chans_per_electrode, electrode-1)

            # find the pulse trains again
            pulse_trains, discharge_times, _ = get_pulse_trains(self.signal_dict['data'], self.rejected_channels, self.decomp_dict['masked_mu_filters'], self.chans_per_electrode, self.signal_dict['fsamp'],electrode-1)

            # get online parameters
            _ , inv_extended_data, norm, centroids =  get_online_parameters(self.signal_dict['data'], self.rejected_channels, new_mu_filters, self.chans_per_electrode, self.signal_dict['fsamp'],electrode-1)

            self.mu_dict['pulse_trains'].append(pulse_trains)
            self.mu_dict['inv_extended_data'].append(inv_extended_data)
            self.mu_dict['norm'].append(norm)
            self.mu_dict['centroids'].append(centroids)
            self.mu_dict['mu_filters'].append(new_mu_filters)
            
        if electrode != 1:
            self.mu_dict['discharge_times'].append([])

        for j in range(len(discharge_times)):

            self.mu_dict['discharge_times'][electrode-1].append(discharge_times[j])

    def post_process_across_arrays(self):

        mu_count = 0
        no_arrays = np.shape(self.mu_dict['pulse_trains'])[0]
        for i in range(no_arrays):

            if self.mu_dict['pulse_trains'][i].size: # if the np.array for a given grid/needle is not empty, continue with post-processing
                mu_count += np.shape(self.mu_dict['pulse_trains'][i])[0]

        all_pulse_trains = np.zeros([mu_count, np.shape(self.signal_dict['target'])[0]])
        all_discharge_times = [] # different mus will have discharge time arrays of different lengths
        muscle = np.zeros(mu_count,dtype=int)

        mu = 0

        for i in range(no_arrays): # iterating over arrays
             if self.mu_dict['pulse_trains'][i].size:
                 for j in range(np.shape(self.mu_dict['pulse_trains'][i])[0]): # iterating over the mus per array

                    all_pulse_trains[mu,:] = self.mu_dict['pulse_trains'][i][j]
                    all_discharge_times.append(self.mu_dict['discharge_times'][i][j])
                    muscle[mu] = i
                    mu += 1

        discharge_times_new, pulse_trains_new, muscle_new = remove_duplicates_between_arrays(all_pulse_trains, all_discharge_times, muscle, np.round(self.signal_dict['fsamp']/40),0.00025, self.dup_thr, self.signal_dict['fsamp'])

        # now that duplicates are removed across arrays, use the muscle variable to regroup the mus back into their respective electrodes (or wherever they had the lowest ISI)

        del self.mu_dict["discharge_times"]
        self.mu_dict["discharge_times"] = [[]] # empty nested list
        del self.mu_dict["pulse_trains"]
        self.mu_dict["pulse_trains"] = []

        for i in range(no_arrays):

            if i != 0:
                self.mu_dict['discharge_times'].append([])
            
            idx = np.where(muscle_new == i)[0] # find the indices for mu -> array mapping
            self.mu_dict["pulse_trains"].append(pulse_trains_new[idx])
            for j in range(np.shape(self.mu_dict["pulse_trains"][i])[0]):

                self.mu_dict["discharge_times"][i].append(discharge_times_new[idx[j]])


        self.mu_dict["muscle"] = muscle_new
        print('Processing across electrodes complete')

        

        
    