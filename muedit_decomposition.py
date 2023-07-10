import glob, os, scipy
import xml.etree.ElementTree as ET
import tarfile as tf
import numpy as np
import matplotlib.pyplot as plt
from processing_tools import *
import tkinter as tk
from tkinter import simpledialog, filedialog
from scipy import signal
import time
root = tk.Tk()
np.random.seed(1337)

class EMG():

    def __init__(self):

        # pre and during processing
        self.its = 50 # number of iterations of the fixed point algorithm 
        self.ref_exist = 1 # Boolean for whether an existing reference is used for signal batching (otherwise, manual selection)
        self.windows = 3  # number of segmented windows over each contraction
        self.check_emg = 1 # Boolean for the review process of EMG channels, where 0 = Automatic selection 1 = Visual checking
        self.drawing_mode = 1 # 0 = Output in the command window ; 1 = Output in a figure
        self.differential_mode = 0 # 0 = no; 1 = yes (filter out the smallest MU, can improve decomposition at the highest intensities
        self.peel_off = 1 # Boolean to determine whether we will update the residual EMG by removing the motor units with the highest SIL value
        self.peeloffwin = 0.025 # Duraiton of windows for checking for MUAPs in the EMG signal
        self.cov_filter = 0 # if we want to threshold the MU filters based on CoV also
        self.cov_thr = 0.5 # Threshold for CoV of ISI values
        self.sil_thr = 0.9 # Threshold for SIL values (matching Negro et.al 2016)
        self.silthrpeeloff = 0.9 # Threshold for MU removed from the signal (if the sparse  deflation is on)
        self.ext_factor = 1000 # extension of observations for numerical stability 
        self.edges2remove = 2 # trimming the batched data, to remove the effects of spectral leakage
        self.plat_thr = 0.8 # giving some padding about the segmentation of the plateau region, if used
        # post processing
        self.alignMUAP = 0 # Boolean to determine whether we will realign the discharge times with the peak of MUAPs (channel with the MUAP with the highest p2p amplitudes, from double diff EMG signal)
        self.refineMU = 0 # Boolean to determine whether we refine MUs, involve 1) removing outliers (1st time), 2) revaluating the MU pulse trains
        self.dup_thr = 0.3 # Threshold that defines the minimal percentage of common discharge times between duplicated motor units
        self.cov_dr = 0.3 # Threshold that define the CoV of discharge that we aim to reach, if we refine the MUs (i.e. refineMU = 1)


#######################################################################################################
########################################## OFFLINE EMG ################################################
#######################################################################################################

class offline_EMG(EMG):

    # child class of EMG, so will inherit it's initialisaiton
    def __init__(self): #, save_dir, to_filter):
        super().__init__()
        #self.save_dir = save_dir # Directory at which final discharges will be saved
        #self.to_filter = to_filter # Boolean to determine whether or not you notch and butter filter the signal, more relevant for matching real-time protocols


    def open_otb_new(self,inputfile):

        #file_name = inputfile.split('/')[1]
        
        temp_dir = os.path.join(self.save_dir, 'temp_tarholder')
        print(temp_dir)

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

        # get sampling frequency, no. bits of AD converter, no. channels, grid names and muscle names
        fsamp = int(xml.find('.').attrib['SampleFrequency'])
        nADbit = int(xml.find('.').attrib['ad_bits'])
        nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
        grid_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
        muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
        ngrids =int(np.floor(nchans/64))

        # read in the EMG trial data
        emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
        emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
        emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division

        # convert the data from bits to microvolts
        for i in range(nchans):
            emg_data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids,grids = grid_names[:ngrids],muscles = muscle_names[:ngrids]) # discard the other muscle and grid entries, not relevant

        # if the signals were recorded with a feedback generated by OTBiolab+, get the target and the path performed by the participant
        if self.ref_exist:

            # only opening the last two .sip files because the first is not needed for analysis
            # would only need MSE between the participant path (file 2) and the target path (file 3)
            ######## path #########
            _, target_label, path_label = glob.glob(f'{temp_dir}/*.sip')
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

        return


    def open_otb(self, inputfile):

        file_name = inputfile.split('/')[1]
        temp_dir = os.path.join(self.save_dir, 'temp_tarholder')
        print(temp_dir)

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

        # get sampling frequency, no. bits of AD converter, no. channels, grid names and muscle names
        fsamp = int(xml.find('.').attrib['SampleFrequency'])
        nADbit = int(xml.find('.').attrib['ad_bits'])
        nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
        grid_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
        muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
        ngrids =int(np.floor(nchans/64))

        # read in the EMG trial data
        emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
        emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
        emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division

        # convert the data from bits to microvolts
        for i in range(nchans):
            emg_data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids,grids = grid_names[:ngrids],muscles = muscle_names[:ngrids]) # discard the other muscle and grid entries, not relevant

        # if the signals were recorded with a feedback generated by OTBiolab+, get the target and the path performed by the participant
        if self.ref_exist:

            # only opening the last two .sip files because the first is not needed for analysis
            # would only need MSE between the participant path (file 2) and the target path (file 3)
            ######## path #########
            _, target_label, path_label = glob.glob(f'{temp_dir}/*.sip')
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

        return
        
    def grid_formatter(self):

        """ Match up the signals with the grid shape and numbering """

        grid_names = self.signal_dict['grids']
        self.signal_dict['filtered_data'] = np.zeros([np.shape(self.signal_dict['data'])[0],np.shape(self.signal_dict['data'])[1]])
        c_map = []
        r_map = []

        for i in range(self.signal_dict['ngrids']):
       
            if grid_names[i] == 'GR04MM1305':

                ElChannelMap = [[0, 24, 25, 50, 51], 
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
                        [11, 12, 37, 38, 63]]
                
                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 4
                self.emg_type = 0

            elif grid_names[i] == 'ELSCH064NM2':

                ElChannelMap = [[0, 0, 1, 2, 3],
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
                        [56, 57, 58, 59, 60]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 8
                self.emg_type = 0

            elif grid_names[i] == 'GR08MM1305':

                ElChannelMap = [[0, 24, 25, 50, 51], 
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
                    [11, 12, 37, 38, 63]]
              

                
                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 8
                self.emg_type = 0

            elif grid_names[i] == 'GR10MM0808':

                ElChannelMap = [[7, 15, 23, 31, 39, 47, 55, 63],
                        [6, 14, 22, 30, 38, 46, 54, 62],
                        [5, 13, 21, 29, 37, 45, 53, 61],
                        [4, 12, 20, 28, 36, 44, 52, 60],
                        [3, 11, 19, 27, 35, 43, 51, 59],
                        [2, 10, 18, 26, 34, 42, 50, 58],
                        [1, 9, 17, 25, 33, 41, 49, 57],
                        [0, 8, 16, 24, 32, 40, 48, 56]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 10
                self.emg_type = 0

            elif grid_names[i] == 'intraarrays':
                
                ElChannelMap = [[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 23, 33],
                        [4, 14, 24, 34],
                        [5, 15, 25, 35],
                        [6, 16, 26, 36],
                        [7, 17, 27, 37],
                        [8, 18, 28, 38],
                        [9, 19, 29, 39]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],40])
                IED = 1
                self.emg_type = 1
            
            ElChannelMap = np.array(ElChannelMap)
            chans_per_grid = (np.shape(ElChannelMap)[0] * np.shape(ElChannelMap)[1]) - 1
            coordinates = np.zeros([chans_per_grid,2])
            rows, cols = ElChannelMap.shape
            row_indices, col_indices = np.unravel_index(np.arange(ElChannelMap.size), (rows, cols))
            coordinates[:, 0] = row_indices[1:]
            coordinates[:, 1] = col_indices[1:]

          
            c_map.append(np.shape(ElChannelMap)[1])
            r_map.append(np.shape(ElChannelMap)[0])
            
            grid = i + 1 
            
            # notch filtering
            self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:] = notch_filter(self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid,:],self.signal_dict['fsamp'])
            
            self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:] = bandpass_filter(self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:],self.signal_dict['fsamp'],emg_type = self.emg_type)        

        self.c_maps = c_map
        self.r_maps = r_map
        self.rejected_channels = rejected_channels
        self.ied = IED
        self.coordinates = coordinates
        

                    
    def manual_rejection(self):

        """ Manual rejection for channels with noise/artificats by inspecting plots of the grid channels """

        for i in range(self.signal_dict['ngrids']):

            grid = i + 1
            chans_per_grid = (self.r_maps[i] * self.c_maps[i]) - 1
            sig2inspect = self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:]

            for c in range(self.c_maps[i]):
                for r in range(self.r_maps[i]):

                    num_chans2reject = []
                    if (c+r) > 0: # TO-DO: remove the assumption of the left corner channel being invalid
                        plt.plot(sig2inspect[(c*self.r_maps[i])+r-1,:]/max(sig2inspect[(c*self.r_maps[i])+r-1,:])+r+1)
                plt.show()
                
                inputchannels = simpledialog.askstring(title="Channel Rejection",
                                  prompt="Please enter channel numbers to be rejected (1-13), input with spaces between numbers:")
                print("The selected channels for rejection are:", inputchannels)
                
                if inputchannels:
                    str_chans2reject = inputchannels.split(" ")
                    for j in range(len(str_chans2reject)):

                        num_chans2reject.append(int(str_chans2reject[j])+c*self.r_maps[i]-1)

                    self.rejected_channels[i,num_chans2reject] =  1
        
        # TO DO: remove the assumption of the top LHC channel needing to be rejected
        self.rejected_channels = self.rejected_channels[:,1:] # get rid of the irrelevant top LHC channel
      
    def batch_w_target(self):

        plateau = np.where(self.signal_dict['target'] >= max(self.signal_dict['target']))[0] * self.plat_thr # finding where the plateau is
        discontinuity = np.where(np.diff(plateau) > 1)[0]
        if self.windows > 1 and not discontinuity: 
        
            plat_len = plateau[-1] - plateau[0]
            wind_len = np.floor(plat_len/self.windows)
            batch = np.zeros(self.windows*2)
            for i in range(self.windows):

                batch[i*2] = plateau[0] + i*wind_len + 1
                batch [(i+1)*2-1] = plateau[0] + (i+1)*wind_len

            self.plateau_coords = batch
            
        elif self.windows >= 1 and discontinuity: 
           
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
        chans_per_grid = (self.r_maps[i] * self.c_maps[i]) - 1
        self.chans_per_grid = chans_per_grid
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['ngrids'] * n_intervals)

        for i in range(int(self.signal_dict['ngrids'])):
            
            grid = i + 1
            for interval in range(n_intervals):
                
                data_slice = self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid, int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i,:] == 1
                # Remove rejected channels
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1

        self.signal_dict['batched_data'] = batched_data
            

    def batch_wo_target(self):

        fake_ref = np.zeros([np.shape(self.signal_dict['data'])][1])
        
        # only looking at the first half of all EMG channel data
        tmp = np.zeros(int(np.floor(np.shape(self.signal_dict['data'])[0]/2)), np.shape(self.signal_dict['data'])[1])
        for i in range(np.shape(tmp)[0]):
            tmp[i,:] =  moving_mean1d(abs(self.signal_dict['data'][i,:]),self.signal_dict['fsamp'])
        
        fake_ref = np.mean(tmp,axis=0)
        signal['path'] = fake_ref
        signal['target'] = fake_ref
        plt.figure(figsize=(10,8))
        plt.plot(tmp.T, color = [0.5,0.5,0.5], linewidth = 0.5)
        plt.plot(fake_ref,'k',linewidth=1)
        plt.ylim([0, max(fake_ref)*1.5])
        plt.title('EMG amplitude for first half of channel data')
        plt.grid()
        plt.show()
        tmp = None
        window_clicks = plt.ginput(2*self.windows, show_clicks = True) # TO-DO: is the output of plt.ginput 1d?
        self.plateau_coords = np.zeros([1,self.windows *2])
        chans_per_grid = (self.r_maps[i] * self.c_maps[i]) - 1
        self.chans_per_grid = chans_per_grid
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['ngrids'] * n_intervals) 

        for interval in range(self.windows):
            
            self.plateau_coords[interval*2] = np.floor(window_clicks[interval*2][0])
            self.plateau_coords[(interval+1)*2-1] = np.floor(window_clicks[(interval+1)*2-1][0])
        
        for i in range(int(self.signal_dict['ngrids'])):

            grid = i + 1
            for interval in range(n_intervals):
                
                data_slice = self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid, int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i,:] == 1
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1
        
        self.signal_dict['batched_data'] = batched_data

  
################################ CONVOLUTIVE SPHERING ########################################
    def convul_sphering(self,g,interval,tracker):

        """ 1) Filter the batched EMG data 2) Extend to improve speed of convergence/reduce numerical instability 3) Remove any DC component  4) Whiten """
        chans_per_grid = self.chans_per_grid
        grid = g+1
        if self.to_filter: # adding since will need to avoid this step if doing real-time decomposition + biofeedback

            self.signal_dict['batched_data'][tracker]= notch_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'])
            self.signal_dict['batched_data'][tracker] = bandpass_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'],emg_type = self.emg_type)  

        # differentiation - typical EMG generation model treats low amplitude spikes/MUs as noise, which is common across channels so can be cancelled with a first order difference. Useful for high intensities - where cross talk has biggest impact.
        if self.differential_mode:
            
            to_shift = self.signal_dict['batched_data'][tracker]
            self.signal_dict['batched_data'][tracker]= []
            self.signal_dict['batched_data'][tracker]= np.diff(self.signal_dict['batched_data'][tracker],n=1,axis=-1)

        # signal extension - increasing the number of channels to 1000
        # Holobar 2007 -  Multichannel Blind Source Separation using Convolutive Kernel Compensation (describes matrix extension)
        extension_factor = int(np.round(self.ext_factor/len(self.signal_dict['batched_data'][tracker])))
      
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
        
        if g == 0: # don't need to repeat for every grid, since the path and target info (informing the batches), is the same for all grids
            self.plateau_coords[interval*2] = self.plateau_coords[interval*2]  + int(np.round(self.signal_dict['fsamp']*self.edges2remove)) - 1
            self.plateau_coords[(interval+1)*2 - 1] = self.plateau_coords[(interval+1)*2-1]  - int(np.round(self.signal_dict['fsamp']*self.edges2remove))

        print('Signal extension and whitening complete')
        
######################### FAST ICA AND CONVOLUTIVE KERNEL COMPENSATION  ############################################

    def fast_ICA_and_CKC(self,g,interval,tracker,cf_type = 'square',ortho_type = 'ord_deflation'):

        
        init_its = np.zeros([self.its],dtype=int) # tracker of initialisaitons of separation vectors across iterations
        fpa_its = 500 # maximum number of iterations for the fixed point algorithm
        # identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
        # occurs. Then, projection vector is initialised to the whitened observation vector, at this located time instant.
       
        Z = np.array(self.decomp_dict['whitened_obvs'][interval]).copy()
        sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 0)))
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
      

        temp_MU_filters = np.zeros([np.shape(self.decomp_dict['whitened_obvs'][interval])[0],self.its])
        temp_CoVs = np.zeros([self.its])

        for i in range(self.its):

                #################### FIXED POINT ALGORITHM #################################
                init_its[i] = sort_sq_sum_Z[-(i+1)] # since the indexing starts at -1 the other way (for ascending order list)
                self.decomp_dict['w_sep_vect'] = Z[:,int(init_its[i])].copy() # retrieve the corresponding signal value to initialise the separation vector
                
                # orthogonalise separation vector before fixed point algorithm
                if ortho_type == 'ord_deflation':
                    self.decomp_dict['w_sep_vect'] -= np.dot(self.decomp_dict['B_sep_mat'] @ self.decomp_dict['B_sep_mat'].T, self.decomp_dict['w_sep_vect'])
                elif ortho_type == 'gram_schmidt':
                    self.decomp_dict['w_sep_vect'] = ortho_gram_schmidt(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'])
             
                # normalise separation vector before fixed point algorithm 
                self.decomp_dict['w_sep_vect'] /= np.linalg.norm(self.decomp_dict['w_sep_vect'])
            
                # use the fixed point algorithm to identify consecutive separation vectors
                self.decomp_dict['w_sep_vect'] = fixed_point_alg(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'],Z, cf, dot_cf,fpa_its,ortho_type)
                
                # get the first iteration of spikes using k means ++
                fICA_source, spikes = get_spikes(self.decomp_dict['w_sep_vect'],Z, self.signal_dict['fsamp'])
            
                ################# MINIMISATION OF COV OF DISCHARGES ############################
                if len(spikes) > 1:

                    # determine the interspike interval
                    ISI = np.diff(spikes/self.signal_dict['fsamp'])
                    # determine the coefficient of variation
                    temp_CoVs[i] = np.std(ISI)/np.mean(ISI)
            
                    # update the sepearation vector by summing all the spikes
                    w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
                    # minimisation of covariance of interspike intervals
                    temp_MU_filters[:,i], spikes, temp_CoVs[i] = min_cov_isi(w_n_p1,self.decomp_dict['B_sep_mat'],Z, self.signal_dict['fsamp'],temp_CoVs[i],spikes)
                
                    self.decomp_dict['B_sep_mat'][:,i] = (self.decomp_dict['w_sep_vect']).real # no need to shallow copy here

                    # calculate SIL
                    fICA_source, spikes, self.decomp_dict['SILs'][interval,i] = get_silohuette(temp_MU_filters[:,i],Z,self.signal_dict['fsamp'])
                    # peel off
                    if self.peel_off == 1 and self.decomp_dict['SILs'][interval,i] > self.sil_thr:
                        Z = peel_off(Z, spikes, self.signal_dict['fsamp'])

                    if self.drawing_mode == 1:
                        plt.clf()
                        plt.ion()
                        plt.show()
                        plt.subplot(2, 1, 1)
                        plt.plot(self.signal_dict['target'], 'k--', linewidth=2)
                        plt.plot([self.plateau_coords[interval*2], self.plateau_coords[interval*2]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.plot([self.plateau_coords[(interval+1)*2 - 1], self.plateau_coords[(interval+1)*2 - 1]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.title('Grid #{} - Iteration #{} - Sil = {}'.format(g, i+1, self.decomp_dict['SILs'][interval,i]))
                        plt.subplot(2, 1, 2)
                        plt.plot(time_axis, fICA_source,linewidth = 0.5)
                        plt.plot(time_axis[spikes],fICA_source[spikes],'o')
                        plt.grid()
                        plt.draw()
                        plt.pause(1e-6)
                    else:
                        print('Grid #{} - Iteration #{} - Sil = {}'.format(g, i, self.decomp_dict['SILs'][interval,i]))

                else:
                    print('Grid #{} - Iteration #{} - less than 10 spikes '.format(g, i))
                    # without enough spikes, we skip minimising the covariation of discharges to improve the separation vector
                    self.decomp_dict['B_sep_mat'][:,i] = self.decomp_dict['w_sep_vect'].real  # no need to shallow copy here

                print('Grid #{} - Iteration #{} - Sil = {}'.format(g, i, self.decomp_dict['SILs'][interval,i]))
                
        # keep the MU filters that had associated SIL values equal or greater than the imposed SIL threshold
        temp_MU_filters = temp_MU_filters[:,self.decomp_dict['SILs'][interval,:] >= self.sil_thr]
        if self.cov_filter:
            
            temp_CoVs = temp_CoVs[self.decomp_dict['SILs'][interval,:] >= self.sil_thr]
            temp_MU_filters = temp_MU_filters[:,temp_CoVs <= self.cov_thr]
        
        self.decomp_dict['MU_filters'][interval] = temp_MU_filters
        self.decomp_dict['COVs'][interval] = temp_CoVs

        temp_MU_filters = None
        temp_CoVs = None

######################################## POST PROCESSING #####################################################

    def post_process_EMG(self,g,tracker):
        
        # batch processing over each window
        pulse_trains, discharge_times = batch_process_filters(self.decomp_dict['dewhiten_mat'], self.decomp_dict['MU_filters'],self.signal_dict['inv_extend_obvs'],self.signal_dict['extend_obvs'],self.plateau,self.ext_factor,self.differential_mode,np.shape(self.signal_dict['data'])[1],self.signal_dict['fsamp'])
        
        # realign the discharge times with the centre of the MUAP
        # UPDATE: no longer realigning
        discharge_times_aligned = discharge_times.copy()

        # remove duplicate MU discharge trains
        pulse_trains, discharge_times_new = remove_duplicates(pulse_trains, discharge_times, discharge_times_aligned, (self.signal_dict['fsamp']/40), 0.00025, self.dup_thr, self.signal_dict['fsamp'])

        # if we want further automatic refinement of MUs, prior to manual edition
        if self.refineMU: 

            # Remove outliers generating irrelevant discharge rates before manual edition (1st time)
            discharge_times_new = remove_outliers(pulse_trains, discharge_times_new, self.CoVDR, self.signal_dict['fsamp'])
            # Re-evaluate all of the UNIQUE MUs over the contraction
            # TO DO: adjust to have adding of length(signal.EMGmask{i}), :) --> generalises to cases where the upper left electrode is not excluded
            self.decomp_dict['pulse_trains'][g], discharge_times_new = refine_mus(self.signal_dict['data'][self.chans_per_grid*(g):self.chans_per_grid*(g) + len(self.rejected_channels[g]),:], self.rejected_channels[g], pulse_trains, discharge_times_new, self.signal_dict['fsamp'])
            
            # Remove outliers generating irrelevant discharge rates before manual edition (2nd time)
            discharge_times_new = remove_outliers(pulse_trains, discharge_times_new, self.CoVDR, self.signal_dict['fsamp'])
        else:
            self.decomp_dict['pulse_trains'][g] = pulse_trains
        