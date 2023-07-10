import numpy as np
import matplotlib, time, threading
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import struct
import tkinter as tk
from tkinter import simpledialog
import socket  # for sockets
import sys  # for exit

######### BASIC PLOT SET UP ###########

plt.rc('font', family='Avenir')
dark_backcol = '#000000'
light_datacol = '#F3F0EF'
mid_groudcol = '#7B7775'
# resizing for canvas embedding 
w, h = figsize = (20, 20)  # figure size
fig = matplotlib.figure.Figure(figsize=figsize)
fig_agg = None

################################## AMP CONFIG UPDATE ######################################



class Amplifier:
   
    def __init__(self):
       
        ####### GUI states #################################
        self.acquisition_state = False
        self.recording_state = False
        self.connection_state = False
        self.CONVERSION_FACTOR = 0.000286
        self.acquisition_data = None
        self.recordings = None
        self.rec_test = None
        self.acq_start_time = 0
        self.acq_real_time = 0
        self.recording_length = 0
        self.recording_real = 0
        self.logs = []
        ####### Acquisition paramters ######################
        self.Fsampling = None
        self.lpf = None
        self.hpf = None
        self.refresh_rate = None
        self.ANOUT_GAIN = self.binaryToDecimal('00110000') # see configuraiton protocol doc -> equivalent to analog output gain of 16
        self.INSEL = self.binaryToDecimal('00001100') # see configuration protocol doc -> equivalent to source input originating from AUX IN
        self.AN_OUT_CH_SEL = self.binaryToDecimal('00000000') # see configuration protocol ->  AN_OUT_IN_SEL byte, this number indicates 
        #which channels of that input have to be provided at the ANALOG OUT BNC on the rear panel.
        #######################
        self.chanarray = []
        self.musclearray = []
        self.ngrids = 0
        self.nchans = 0
        self.nsamp = 0
        self.grid4display = None
        ######## Socket and amplifier configuration ########
        self.conf_string = np.zeros([40],dtype=int)
        self.remote_ip = "169.254.1.10"  # should match the IP address of the quattrocento in use
        self.port = 23456
        ############## Buffers ##########################
        self.input_buffer_size = None
        self.path2file = None
        self.filename = None

    def decimalToBinary(self, n, min_digits):
        # converting decimal to binary
        # and removing the prefix(0b)
        # add 0_padding in front until min_digits
        return str(bin(n).replace("0b", "")).zfill(min_digits)

    def binaryToDecimal(self, n):
        return int(n, 2)

    def configure(self):

        Fsamp = 8*(self.Fsampling == 2048) + 16*(self.Fsampling == 5120) + 24*(self.Fsampling == 10240)
        # e.g. 24 is 00011000, so it 24 positions correctly within the ACQ_SETT BYTE description
        nchan = 0*(self.nchans == 120) + 2*(self.nchans == 216) + 4*(self.nchans == 312) + 6*(self.nchans== 408)
        hp = 48*(self.hpf == 200) + 32*(self.hpf == 100) + 16*(self.hpf == 10) + 0*(self.hpf == 0.3)
        lp = 12*(self.lpf == 4400) + 8*(self.lpf == 900) + 4*(self.lpf == 500) + 0*(self.lpf == 130)
        # assign values to the configuration string
        # 10000000, 7th bit is always set to 1 , 0th bit is 1 when transfer  is active, 0 when it is not
        self.conf_string[0] = self.binaryToDecimal('10000000') + Fsamp + nchan
        #self.conf_string[1] = self.ANOUT_GAIN + self.INSEL
        #self.conf_string[2] = self.AN_OUT_CH_SEL
        self.conf_string[1] = self.binaryToDecimal('00111100')
        self.conf_string[2] = self.binaryToDecimal('00000000')
        ########## INPUTS 1 to 8 and MULTIPLE INPUTS 1 to 4 ##################
        self.conf_string[3::3] = 0
        self.conf_string[np.arange(5,39,3)] = hp + lp
        self.conf_string[39] = self.CRC8(39)
        print('Configuration string completed')

        print(self.conf_string)

    def start_or_stop(self,change_state):

        # change state: can be -1, 0 or 1
        # 0 complete closure/inactiviation
        # -1 flushing for new recording
        # 1 complete activation

        if change_state != 0 :
            self.conf_string[0] += change_state
        else:
            self.conf_string[0] = self.binaryToDecimal('10000000') # terminating byte
        self.conf_string[39] = self.CRC8(39)

        print(self.conf_string)

    def CRC8(self,length):
        # Cyclic Redundancy Check
        # Input is a byte stream, outputs a 16 or 32 bit integer value
        # For detection of errors
        crc = 0
        for i in range(length):
            extract = self.conf_string[i]
            for j in range(8): # 8 since 8 bits in a byte
                sum = (crc ^ extract) & 1
                crc >>= 1 # shifting bit right by 1
                if sum:
                    crc ^= 0x8C
                extract >>= 1
        return crc
        

    def socket_connect(self):

        ################# Create socket ###############################
        try:
            # create an AF_INET, STREAM socket (TCP), store in class as socket object
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # must specify you want to reuse the socket on other iterations, other it will be rendered unuseable
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print("Socket created.")
        except socket.error:
            print('Failed to create socket.')
            sys.exit()
        ############### Connect socket to quattrocento ###############
        try:
            # Connect to remote server
            self.socket.connect((self.remote_ip, self.port))
            print('Connected to quattrocento with IP: ' + self.remote_ip)
        except socket.error as ex:
            print('Failed to connect to quattrocento with IP ' + self.remote_ip)


    def socket_disconnect(self):

        ################ Destroy socket ################################
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            print('Disonnected from quattrocento with IP: ' + self.remote_ip)
        except:
            print('Failed to disconnect from quattrocento with IP ' + self.remote_ip)


    def socket_send(self):

        # configuration string needs to be converted into a byte stream prior to sending as a command to the quattrocento
        cmd = struct.pack("<40B", *self.conf_string) # < specifies little-endian
        # socket updates configuration of the quattrocento
        try:
            self.socket.sendall(cmd)
        except socket.error:
            print("The socket connection has broken. Please restart")

        # maybe better for real-time?...
        # Send the data in chunks
        """
        total_sent = 0
        while total_sent < len(cmd):
            sent = socket.send(cmd[total_sent:])
            if sent == 0:
                raise RuntimeError("The socket connection has broken. Please restart")
            total_sent += sent
            print(f'{total_sent} bytes sent')
        """


    def visualise_noise(self,window,fig_emg,ax_emg,canvas_emg):
        
        # how many iterations/buffer windows are required to collect 15 seconds of data
        # self.nsamp is therefore the number of samples per window
        nwin = int(np.floor(15*float(self.Fsampling)/float(self.nsamp)))
        # TO DO: remove the assumption of the number of channels per grid -> DONE: 4 x ngrids to len(self.chanarray)
        allEMG = np.zeros([len(self.chanarray),15*self.Fsampling]) # total number of channels x total number of samples across 15 seconds 
        channels = np.arange(1, 65)
        offset = np.ones([64, self.nsamp])
        offset = offset * channels[:, np.newaxis]

        time_axis = np.linspace(0, 5, 5*int(self.Fsampling))
        buffer = np.zeros([5*int(self.Fsampling), 64])
        i = 0
        buffer_count = 0
        # we are plotting 5 seconds, but the total recording span is 15 seconds
        buffer_max = int(np.floor(5*float(self.Fsampling)/float(self.nsamp))) - 1 # accounting for Python 0-indexing

        noise_check = np.zeros([64])

        # to initiate a recording, the 0th bit of the first byte in the configuration string must be changed from 0 to 1
        # assuming the socket is already opened?
        self.start_or_stop(1)
        self.socket_send()

        while i <= nwin - 1: # accounting for Python 0-indexing

            bytes_to_read = self.nchans * self.nsamp * 2 # 2 bytes per int16
            
            received_data_temp = self.socket.recv(bytes_to_read,socket.MSG_WAITALL)
           
            # convert the recieved data from a bytes object to a numpy array
            # unpack the data into a numpy array of int16
            # < means unpacking assumes the byte order is little-endian
            # h  means unpacted as short integers (int16)
            received_data = np.array(struct.unpack("<{}h".format((self.nchans*self.nsamp)),received_data_temp))
            received_data = received_data.reshape((self.nsamp,self.nchans)).T
            allEMG[:,self.nsamp*i:(i+1)*self.nsamp] = received_data[self.chanarray,:] # selecting the right channels from the set-up
            buffer[buffer_count*self.nsamp:(buffer_count+1)*self.nsamp,:] = (allEMG[(self.grid4display-1)*64:(self.grid4display)*64,self.nsamp*i:(i+1)*self.nsamp]*0.0005 + offset).T
            
            ax_emg.cla()                    # Clear axes first if required
            ax_emg.set_facecolor((0,0,0))
            ax_emg.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
            for spine_type in ['left','bottom']:
                ax_emg.spines[spine_type].set_color('#FFFFFF')
            ax_emg.set_xlabel("Time (s)",c='#FFFFFF',fontsize=15)
            ax_emg.set_ylabel("Channels", c='#FFFFFF',fontsize=15)
            ax_emg.set_xlim(0,5)
            ax_emg.set_ylim(0,66)
            # new attempt for plotting
            ax_emg.plot(time_axis,buffer,'w',linewidth = 1)
            ax_emg.axvline(x=time_axis[buffer_count* self.nsamp], linewidth=1,color='red')
            canvas_emg.draw()  
            window.refresh()
            print('Canvas drawn...')

            i += 1
            if buffer_count < buffer_max:
                buffer_count += 1
            else:
                buffer_count = 0 # resetting for next i <= nwin?

    
        self.start_or_stop(-1)
        self.socket_send()
        self.socket_disconnect()
        self.connect_state = False
        self.socket = None # reset the socket
       
        # save the data in an array
        np.save(self.path2file + self.filename + '_noise', allEMG)
        print('Recording to check for noise and saving data complete.')


        # determine the root mean square
        shaper = np.shape(allEMG)[1]
        for j in range(64):
            noise_check[j] = np.sqrt(np.mean(np.square(allEMG[(self.grid4display-1)*64+j,self.Fsampling-1:shaper-self.Fsampling ])))/1000


        ax_emg.cla()                    # Clear axes first if required
        ax_emg.set_facecolor((0,0,0))
        ax_emg.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
        for spine_type in ['left','bottom']:
            ax_emg.spines[spine_type].set_color('#FFFFFF')
        ax_emg.set_xlabel("Channels",c='#FFFFFF',fontsize=15)
        ax_emg.set_ylabel("RMS (AU)", c='#FFFFFF',fontsize=15)
        ax_emg.set_xlim(0,65)
        ax_emg.set_ylim(0,max(noise_check))
        # new attempt for plotting
        ax_emg.bar(range(1,len(noise_check)+1),noise_check,color='white')
        ax_emg.plot(time_axis,buffer,'w',linewidth = 1)
        ax_emg.axvline(x=time_axis[buffer_count* self.nsamp], linewidth=1,color='red')
        canvas_emg.draw()  
        print('Canvas drawn...')
        window.refresh()
    
  

    def visualise_emg(self,window,fig_emg,ax_emg,canvas_emg):
        
        # how many iterations/buffer windows are required to collect 15 seconds of data
        # self.nsamp is therefore the number of samples per window
        nwin = int(np.floor(30*float(self.Fsampling)/float(self.nsamp)))
        # TO DO: remove the assumption of the number of channels per grid -> DONE: 4 x ngrids to len(self.chanarray)
        allEMG = np.zeros([len(self.chanarray),30*self.Fsampling]) # total number of channels x total number of samples required across 15 seconds 
        channels = np.arange(1, 65)
        offset = np.ones([64, self.nsamp])
        offset = offset * channels[:, np.newaxis]

        time_axis = np.linspace(0, 5, 5*int(self.Fsampling))
        buffer = np.zeros([5*int(self.Fsampling), 64])
        i = 0
        buffer_count = 0
        # we are plotting 5 seconds, but the total recording span is 15 seconds
        buffer_max = int(np.floor(5*float(self.Fsampling)/float(self.nsamp))) - 1 # accounting for Python 0-indexing

        # to initiate a recording, the 0th bit of the first byte in the configuration string must be changed from 0 to 1
        # assuming the socket is already opened?
        self.start_or_stop(1)
        self.socket_send()

        while i <= nwin - 1: # accounting for Python 0-indexing

            bytes_to_read = self.nchans * self.nsamp * 2 # 2 bytes per int16
            
            received_data_temp = self.socket.recv(bytes_to_read,socket.MSG_WAITALL)
           
            # convert the recieved data from a bytes object to a numpy array
            # unpack the data into a numpy array of int16
            # < means unpacking assumes the byte order is little-endian
            # h  means unpacted as short integers (int16)
            received_data = np.array(struct.unpack("<{}h".format((self.nchans*self.nsamp)),received_data_temp))
            received_data = received_data.reshape((self.nsamp,self.nchans)).T
            allEMG[:,self.nsamp*i:(i+1)*self.nsamp] = received_data[self.chanarray,:] # selecting the right channels from the set-up
            buffer[buffer_count*self.nsamp:(buffer_count+1)*self.nsamp,:] = (allEMG[(self.grid4display-1)*64:(self.grid4display)*64,self.nsamp*i:(i+1)*self.nsamp]*0.0005 + offset).T
            
            ax_emg.cla()                    # Clear axes first if required
            ax_emg.set_facecolor((0,0,0))
            ax_emg.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
            for spine_type in ['left','bottom']:
                ax_emg.spines[spine_type].set_color('#FFFFFF')
            ax_emg.set_xlabel("Time (s)",c='#FFFFFF',fontsize=15)
            ax_emg.set_ylabel("Channels", c='#FFFFFF',fontsize=15)
            ax_emg.set_xlim(0,5)
            ax_emg.set_ylim(0,66)
            # new attempt for plotting
            ax_emg.plot(time_axis,buffer,'w',linewidth = 1)
            ax_emg.axvline(x=time_axis[buffer_count* self.nsamp], linewidth=1,color='red')
            canvas_emg.draw()  
            window.refresh()
            print('Canvas drawn...')

            i += 1
            if buffer_count < buffer_max:
                buffer_count += 1
            else:
                buffer_count = 0 # resetting for next i <= nwin?

    
        self.start_or_stop(-1)
        self.socket_send()
        self.socket_disconnect()
        self.connect_state = False
        self.socket = None # reset the socket
       
        # save the data in an array
        np.save(self.path2file + self.filename +'_EMG', allEMG)
        print('Recording to check EMG quality and saving data complete.')


        shaper = np.shape(allEMG)[1]
        print(allEMG)
        for i in range(self.ngrids):
            grid = i + 1
            EMG_checker = np.zeros([13,shaper*5])
            for j in range(5):
                if j < 4: 
                    EMG_checker[:,j*shaper:(j+1)*shaper] =  allEMG[(i*64)+(j*13):(i*64)+(j+1)*13,:]
                else: # exclude the top left corner channel
                    EMG_checker[0:12,j*shaper:(j+1)*shaper] = allEMG[(i*64)+(j*13):(i*64)+(j+1)*13 - 1,:]


            
            channels_2 = np.arange(1, 14)
            offset_2 = np.ones([13, np.shape(EMG_checker)[1]])
            offset_2 = offset_2 * channels_2[:, np.newaxis]
            EMG_checker = EMG_checker * 0.0005 + offset_2


            #### new plot ####
            ax_emg.cla()                    # Clear axes first if required
            ax_emg.set_facecolor((0,0,0))
            ax_emg.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
            for spine_type in ['left','bottom']:
                ax_emg.spines[spine_type].set_color('#FFFFFF')
            ax_emg.set_xlabel("Samples",c='#FFFFFF',fontsize=15)
            ax_emg.set_ylabel("Grid Rows", c='#FFFFFF',fontsize=15)
            ax_emg.set_xlim(0,np.shape(EMG_checker)[1])
            ax_emg.set_ylim(0,14)
            # new attempt for plotting
            ax_emg.plot(np.arange(0,5*shaper),EMG_checker.T,'w',linewidth = 1) # no time context, so no time axis
            # each vertical line landmarks data from the next electrode in the grid row (row spanning from 0 to 12)
            ax_emg.axvline(x=shaper, linewidth=1,color='red')
            ax_emg.axvline(x=2*shaper, linewidth=1,color='red')
            ax_emg.axvline(x=3*shaper, linewidth=1,color='red')
            ax_emg.axvline(x=4*shaper, linewidth=1,color='red')
            
            canvas_emg.draw()  
            window.refresh()

            # might need to change to a more basic user request part, if too slow...
            self.rejected_channels = np.zeros([self.ngrids,13,5])

            for j in range(5):
                # now prompt for user input:
                inputchannels = simpledialog.askstring(title="Channel Rejection",
                                    prompt="Please enter channel numbers to be rejected (1-13), input with spaces between numbers:")
                print("The selected channels for rejection are:", inputchannels)
            
                if inputchannels:
                    str_chans2reject = inputchannels.split(" ")
                    
                    num_chans2reject = np.array([int(x) - 1 for x in str_chans2reject]) # since user input ranges from 1-13, but indexing ranges from 0-12
                    print(type(num_chans2reject))
                    print(num_chans2reject)
                    self.rejected_channels[i,num_chans2reject,j] =  1
        
        print('Rejected channels')
        print(self.rejected_channels)


    ########################################## MVC TRAINING SET UP ################################################

    def establish_offset(self,window):

        # how many iterations/buffer windows are required to collect 3 seconds of data
        # self.nsamp is therefore the number of samples per window
        nwin = int(np.floor(3*float(self.Fsampling)/float(self.nsamp)))
        force = np.zeros([nwin*self.nsamp])
        i = 0
        # to initiate a recording, the 0th bit of the first byte in the configuration string must be changed from 0 to 1
        # assuming the socket is already opened
        self.start_or_stop(1)
        self.socket_send()
        # TO DO: pass in the window and enable the disconnection button!!!! (done)
        window['-DQUAT_B-'].Update(disabled=False)
        window['-DQUAT_R-'].Update(disabled=False)
        window['-DQUAT_T-'].Update(disabled=False)
    
        while i <= nwin - 1: # accounting for Python 0-indexing

            bytes_to_read = self.nchans * self.nsamp * 2 # 2 bytes per int16
            received_data_temp = self.socket.recv(bytes_to_read,socket.MSG_WAITALL)
            # convert the recieved data from a bytes object to a numpy array
            # unpack the data into a numpy array of int16
            # < means unpacking assumes the byte order is little-endian
            # h  means unpacted as short integers (int16)
            received_data = np.array(struct.unpack("<{}h".format((self.nchans*self.nsamp)),received_data_temp))
            received_data = received_data.reshape((self.nsamp,self.nchans)).T
            force[i*self.nsamp: (i+1)*self.nsamp] = received_data[self.chanarray[-1]+1, :] # +1 since its the last channel on the grid, that is the reference?
            i += 1

        self.start_or_stop(-1)
        self.socket_send()
        self.socket_disconnect()
        self.connect_state = False
        self.socket = None # reset the socket
        print('Calculating force')
        self.offset_force = np.mean(force[int(self.Fsampling):])
        print(self.offset_force)
        window['-DISPOFFSET-'].Update(str(self.offset_force))
        

    def reset_offset(self,window):

        self.offset_force = 0
        window['-DISPOFFSET-'].Update('Not selected')

    def connection(self, state):
        self.connection = state
        self.button_connection_text = 'DISCONNECT FROM AMPLIFIER'

    def get_connection(self):
        return self.connection_state

    def get_button_connection_text(self):
        return self.button_connection_text

    def get_socket(self):
        return self.socket()

    def update_channels(self,new_muscle):
        self.musclearray = [self.musclearray,new_muscle]



################################## FORCE TEMPLATE UPDATE ######################################

class Force:
    # here data is given as obj constructor
    def __init__(self): 

        self.mvc = 0
        self.rampd = 0
        self.rampp = 0
        self.mvcz = 0
        self.its = float(1)
        self.fsamp = float(100)

    def update_force_plot(self,window):

        zfloat = float(0)
        online_duration = (self.rampp * self.fsamp + 2*(self.rampd*self.fsamp + self.mvcz*self.fsamp))* self.its
        time_axis =  np.linspace(zfloat,online_duration/self.fsamp,int(online_duration))

        rest_phase = np.linspace(zfloat,zfloat,int(self.mvcz*self.fsamp))
        ramp_up_phase = np.linspace(zfloat,self.mvc,int(self.rampd*self.fsamp))
        ramp_down_phase = np.linspace(self.mvc,zfloat,int(self.rampd*self.fsamp))
        plateau_phase = [self.mvc*x for x in np.linspace(1.0,1.0,int(self.rampp*self.fsamp))]
        
        force_axis = np.concatenate((rest_phase,ramp_up_phase,plateau_phase,ramp_down_phase,rest_phase))
        force_its = force_axis.copy()
    
        if self.its > 1:
            for i in range(int(self.its)-1):
                force_axis =  np.concatenate((force_axis,force_its),axis=0)
        
        plt.ioff()
        plt.figure(figsize = (5,3), facecolor='black')
        plt.plot(time_axis,force_axis,'w',linewidth = 3)
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0))
        plt.tight_layout(pad= 4, w_pad=5, h_pad=5)
        for spine_type in ['left','bottom']:
            ax.spines[spine_type].set_color('#FFFFFF')
        ax.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
        
        ax.set_xlabel('Time (s)', c='#FFFFFF',fontsize=15)
        ax.set_ylabel('MVC Force (N)', c='#FFFFFF',fontsize=15)
        return plt.gcf()

    def draw_force_plot(self,canvas, figure, loc=(0, 0)):

        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=0)
        return figure_canvas_agg

    def delete_force_plot(self,fig_agg):
        fig_agg.get_tk_widget().forget()
        plt.close('all')

    def update_mvc_target(self, state):
        self.mvc = float(state)

    def update_rampd(self, state):
        self.rampd = float(state)

    def update_rampp(self, state):
        self.rampp = float(state)

    def update_rest(self,state):
        self.mvcz = float(state)

    def update_its(self,state):
        self.its = float(state)

    def update_freq(self,state):
        self.fsamp = float(state)

################################# MVC TEMPLATE UPDATE ################################

class MVC:
    # here data is given as obj constructor
    def __init__(self): 

        self.mvc = 0
        self.rampd = float(15) # pre-set for training
        self.rampp = 0
        self.mvcz = float(10) # pre-set for training 
        # self.its = float(1) ---> only one iteration for training
        self.fsamp = float(100)

    def update_mvc_plot(self,window):

        zfloat = float(0)
        online_duration = (self.rampp * self.fsamp + 2*(self.rampd*self.fsamp + self.mvcz*self.fsamp))
        time_axis =  np.linspace(zfloat,online_duration/self.fsamp,int(online_duration))

        rest_phase = np.linspace(zfloat,zfloat,int(self.mvcz*self.fsamp))
        ramp_up_phase = np.linspace(zfloat,self.mvc,int(self.rampd*self.fsamp))
        ramp_down_phase = np.linspace(self.mvc,zfloat,int(self.rampd*self.fsamp))
        plateau_phase = [self.mvc*x for x in np.linspace(1.0,1.0,int(self.rampp*self.fsamp))]
        
        force_axis = np.concatenate((rest_phase,ramp_up_phase,plateau_phase,ramp_down_phase,rest_phase))
        force_its = force_axis.copy()
        plt.ioff()
        plt.figure(figsize = (8,3), facecolor='black')
        plt.plot(time_axis,force_axis,'w',linewidth = 3)
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0))
        plt.tight_layout(pad=4, w_pad=6, h_pad=25)
        for spine_type in ['left','bottom']:
            ax.spines[spine_type].set_color('#FFFFFF')
        ax.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=11)
        ax.set_xlabel('Time (s)', c='#FFFFFF',fontsize=11)
        ax.set_ylabel('MVC Force (N)', c='#FFFFFF',fontsize=11)
        return plt.gcf()

    def draw_mvc_plot(self,canvas, figure, loc=(0, 0)):

        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=0) 
        return figure_canvas_agg

    def delete_mvc_plot(self,fig_agg):
        fig_agg.get_tk_widget().forget()
        plt.close('all')

    def update_mvc_target(self, state):
        self.mvc = float(state)

    def update_duration(self, state):
        self.rampp = float(state)

    def update_freq(self,state):
        self.fsamp = float(state)
  


