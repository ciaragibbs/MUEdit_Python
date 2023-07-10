from muedit_decomposition import *
import PySimpleGUI as sg
import os.path
import numpy as np
import matplotlib, time
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from threading import Thread # to allow runtime of acquisition/recording and stop with button
from pytictoc import TicToc
import math, random


# Path for saving decomposed data
log_dir_emg = "decomposed_data/"
yes_no_dict = dict({'yes': 1 ,'no': 0})
# Classes
emg_obj = offline_EMG() # an object of the offline EMG decomposition class


#################### Setting up plotting in real time #################
class Canvas(FigureCanvasTkAgg):

    # translating the canvas of PySimpleGui into something compatible for Matplotlib
    def __init__(self, figure=None, master=None):
        super().__init__(figure=figure, master=master)
        self.canvas = self.get_tk_widget()
        self.canvas.pack(side='top', fill='both', expand=1)

########################################################################

# might need to modify this 
def refresh(window, start):
    while time.time()-start <10:
            window.refresh()
            


sg.theme("Black")
sg.set_options(font=("Avenir Next", 16))


######################################## GUI DESIGN #################################################
####################################### DECOMPOSITION SETUP ###########################################
decomposition_setup = [
    
    ####### DROP DOWN CHOICES ######
    # first row
     [sg.Text("DECOMPOSITION SETTINGS", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    # second row
     [sg.Text("Enter filename: ")], 
    # third row
    [sg.Input(key='-INPUT_FILENAME-', size=(10, 1), expand_x=True),
        sg.FileBrowse(file_types=(("Quattrocento Recordings", "*.otb+"), ("ALL Files", "*.*"))),
        sg.Button("Load",key='input_filename'),],

    # fourth row
    [sg.Text("Reference:"), sg.Push(),
        sg.Combo(["Force", "EMG Amplitude"], key='-ref-', size=(6, 1), font =("Avenir Next", 14), enable_events=True),],
    # fifth row
    [sg.Text("Check EMG"), sg.Push(),
        sg.Combo(["Yes","No"], key='-check-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],
    # sixth row
    [sg.Text("Contrast Function"), sg.Push(),
        sg.Combo(["logcosh","square","skew"], key='-contrast-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],
    # seventh row
    [sg.Text("Initialization"), sg.Push(),
        sg.Combo(["EMG max", "Random"], key='-init-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],
    # eighth row
    [sg.Text("CoV Filter"), sg.Push(),
        sg.Combo(["Yes", "No"], key='-cov-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],
    # ninth row
    [sg.Text("Peel Off"), sg.Push(),
        sg.Combo(["Yes", "No"], key='-peel-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],
    # tenth row
    [sg.Text("Refine MUs"), sg.Push(),
        sg.Combo(["Yes", "No"], key='-refine-', size=(6, 1),font =("Avenir Next", 14), enable_events=True),],

    [sg.Text("CHOOSING PARAMETERS", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    ##### INPUT CHOICES ######
    # eleventh row
    [sg.Text("No. Iterations",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-its-',  size=(7, 1)),
        sg.Button("Submit",key = 'its',font =("Avenir Next", 14))],
    # twelfth row
    [sg.Text("No. Windows",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-wins-',  size=(7, 1)),
        sg.Button("Submit",key = 'wins',font =("Avenir Next", 14))],
    # fourteenth row
    [sg.Text("No. Electrodes",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-elecs-',  size=(7, 1)),
        sg.Button("Submit",key = 'elecs',font =("Avenir Next", 14))],
    # fivteenth row
    [sg.Text("Threshold Target",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-threstarg-',  size=(7, 1)),
        sg.Button("Submit",key = 'threstarg',font =("Avenir Next", 14))],
    # sixteenth row
    [sg.Text("No. Extended",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-exchan-',  size=(7, 1)),
        sg.Button("Submit",key = 'exchan',font =("Avenir Next", 14))],
    [sg.Text("Channels",font =("Avenir Next", 14)),sg.Push(),],
    # seventeenth row
    # eighteenth row
    [sg.Text("Duplicate",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-dupthres-',  size=(7, 1)),
        sg.Button("Submit",key = 'dupthres',font =("Avenir Next", 14))],
    [sg.Text("Threshold",font =("Avenir Next", 14)),sg.Push(),],
    # nineteenth row 
    [sg.Text("SIL Threshold",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-silthres-',  size=(7, 1)),
        sg.Button("Submit",key = 'silthres',font =("Avenir Next", 14))],
    # twentieth row
    [sg.Text("CoV Threshold",font =("Avenir Next", 14)),sg.Push(),
        sg.Input(key='-covthres-',  size=(7, 1)),
        sg.Button("Submit",key = 'covthres',font =("Avenir Next", 14))],
    [sg.T('')],
    # twenty first row
    [sg.Button("START",font =("Avenir Next", 14),expand_x = True, key='-OFFDECOMP-')],
    # twenty second row
    [sg.T('')]

]


   
######################################## EDITION SETUP #############################################
edition_setup = [  


    # first row
    [sg.Text("EDITION SETTINGS", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    # second row
     [sg.Text("Enter filename: ")], 
    # third row
    [sg.Input(key='-INPUT_FILENAME-', size=(20, 1), expand_x=True),
        sg.Button("Submit",key='input_filename2'),],
    # fourth row
    [sg.Text("BATCH PROCESSING", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    # fifth row
    [sg.Button("1 - Remove all outliers",font =("Avenir Next", 14),expand_x = True, key='-outliers-')],
    # sixth row
    [sg.Button("2 - Revaluate all MU filters",font =("Avenir Next", 14),expand_x = True, key='-reeval-')],
    # seventh row
    [sg.Button("2 - Remove flagged MUs",font =("Avenir Next", 14),expand_x = True, key='-removflagged-')],
    # eighth row
    [sg.Button("2 - Remove duplicates within grids",font =("Avenir Next", 14),expand_x = True, key='-removdupsw-')],
    # nineth row
    [sg.Button("2 - Remove duplicates between grids",font =("Avenir Next", 14),expand_x = True, key='-removdupsb-')],
    # tenth row
    [sg.Text("VISUALISATION", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    # eleventh row
    [sg.Button("Plot MU spike trains",font =("Avenir Next", 14),expand_x = True, key='-plotspikes-')],
    # twelfth row
    [sg.Button("Plot MU firing rates",font =("Avenir Next", 14),expand_x = True, key='-plotrates-')],
    # thirteenth row
    [sg.Text("SAVE THE EDITION", expand_x=True, justification="center",font =("Avenir Next", 16,'bold'))],
    # fourteenth row
    [sg.Button("Save",font =("Avenir Next", 14,'bold'),expand_x = True, key='-saver-')],
    [sg.T('')]
    ]



####################################### COLLECT ITEMS OF TAB GROUP ##################################################
    
decomp_setup = [[sg.TabGroup([[sg.Tab('Decomposition', decomposition_setup,key='DECOMPOSITION'),
                    sg.Tab('Edition', edition_setup,key='EDITION'),]], tab_location='centertop', border_width=5,size= (300,780),enable_events=True)]]  
        
# For now will only show the name of the file that was chosen
signal_display = [ [sg.Canvas(background_color = "#000000", size = (1500,1000), key='emg-canvas',expand_x = True,expand_y = True, pad=((2,0),(0,2)))]
   ]
############################################## GLOBAL LAYOUT ######################################################
layout = [[sg.Column(decomp_setup, element_justification='top', size= (320,900),justification='center',key='-COL1'),
        sg.VSeperator(),
        sg.Column(signal_display,element_justification='top', size= (1100,900),justification='center',key='-COL2-'), 
    ]]

window = sg.Window("DECOMPOSITION AND EDITION GUI", layout, resizable=True, finalize=True, use_default_focus=False, return_keyboard_events=True)

######################################### Running Window #####################################################
acq_thread = None # acquisition thread
refresh_thread = None
fig_force = None
fig_agg = None
count = 0

###### initialising the right hand plot
fig_emg = Figure(figsize=(150, 150), facecolor= 'black',dpi=100)
ax_emg = fig_emg.add_subplot()
canvas_emg = Canvas(fig_emg, window['emg-canvas'].Widget)

fig_emg.tight_layout(pad=70)
ax_emg.set_facecolor((0,0,0))
ax_emg.tick_params(axis='both', colors='#FFFFFF',pad=1,labelsize=15)
for spine_type in ['left','bottom']:
    ax_emg.spines[spine_type].set_color('#FFFFFF')
ax_emg.set_xlabel("Time (s)",c='#FFFFFF',fontsize=15)
ax_emg.set_ylabel("Channels", c='#FFFFFF',fontsize=15)
ax_emg.set_xlim(0,5)
ax_emg.set_ylim(0,66)
#####################################################################

#emgobj is the class object for decomposition

while True:
    
    event, values = window.read()


    ####################################### CONDITIONS #########################################
    if event == 'input_filename':
        
        filepath = values['-INPUT_FILENAME-'] # get the name of the file. but should include otb+
        print(filepath)
        emg_obj.save_dir = os.getcwd()
        emg_obj.open_otb_new(filepath)
        
    
    ##### DECOMPOSITION PARAMETERS ######
    if event == '-ref-':
        emg_obj.ref = values['-ref-']
    if event == '-check-':
        emg_obj.check_emg = yes_no_dict[values['-check-'].lower()]
    if event == 'contrast':
        emg_obj.contrast = values['-contrast-']
    if event == '-init-':
        emg_obj.init = values['-init-']
    if event == '-cov-':
        emg_obj.cov_filter = yes_no_dict[values['-cov-'].lower()]
    if event == '-peel-':
        emg_obj.peel_off =  yes_no_dict[values['-peel-'].lower()]
    if event == '-refine-':
        emg_obj.refineMU = yes_no_dict[values['-refine-'].lower()]

    ##### EDITION PARAMETERS #####


    if event in (sg.WIN_CLOSED, 'Exit'):
        break # exit the while loop and close the gui
        

window.close()


        



        




