
from emg_decomposition_final import EMG, offline_EMG
import glob, os
import numpy as np
import pickle 
import pandas as pd
import json

emg_obj = offline_EMG('/Users/cfg18/Documents/Decomposition Ciara Version/',1)
os.getcwd()
all_files = sorted(glob.glob('./*.otb+')) # list of file names in alphabetical order
checkpoint = 1
all_dicts = []
print(all_files)


########### Update: 12/03/24 - adding different file options for opening...
file_type = 'otb' # 'otb' 'ephys' etc.


for i in range(1): #range(len(all_files)):

    ################## FILE ORGANISATION ################################

    #emg_obj.open_otb_multi(all_files[i]) # adds signal_dict to the emg_obj

    if file_type == 'otb':
        
        emg_obj.open_otb(all_files[i]) 
    
    elif file_type == 'csv':
        # also need to add a method by which we store the acquisiton parameters
        #emg_obj.open_csv(all_files[i])
        print('To be completed')

    elif file_type == 'ephys':
         
        print('To be completed')



    emg_obj.electrode_formatter() # adds spatial context, and additional filtering
    
    if emg_obj.check_emg: # if you want to check the signal quality, perform channel rejection
        emg_obj.manual_rejection()

    #################### BATCHING #######################################

    if emg_obj.ref_exist: # if you want to use the target path to segment the EMG signal, to isolate the force plateau
        print('Target used for batching')
        emg_obj.batch_w_target()
    else:
        emg_obj.batch_wo_target() # if you don't have one, batch without the target path

    ################### CONVOLUTIVE SPHERING #############################
    
    emg_obj.signal_dict['diff_data'] = []
    tracker = 0
    nwins = int(len(emg_obj.plateau_coords)/2)
    for g in range(int(emg_obj.signal_dict['nelectrodes'])):
            
            extension_factor = int(np.round(emg_obj.ext_factor/np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]))
            # these two arrays are holding extended emg data PRIOR to the removal of edges
            emg_obj.signal_dict['extend_obvs_old'] = np.zeros([nwins, np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor), np.shape(emg_obj.signal_dict['batched_data'][tracker])[1] + extension_factor -1 - emg_obj.differential_mode ])
            emg_obj.decomp_dict['whitened_obvs_old'] = emg_obj.signal_dict['extend_obvs_old'].copy()
            # these two arrays are the square and inverse of extended emg data PRIOR to the removal of edges
            emg_obj.signal_dict['sq_extend_obvs'] = np.zeros([nwins,np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor),np.shape(emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor)])
            emg_obj.signal_dict['inv_extend_obvs'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # dewhitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
            emg_obj.decomp_dict['dewhiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # whitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
            emg_obj.decomp_dict['whiten_mat'] = emg_obj.signal_dict['sq_extend_obvs'].copy()
            # these two arrays are holding extended emg data AFTER the removal of edges
            emg_obj.signal_dict['extend_obvs'] = emg_obj.signal_dict['extend_obvs_old'][:,:,int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove)-1):-int(np.round(emg_obj.signal_dict['fsamp']*emg_obj.edges2remove))].copy()
            emg_obj.decomp_dict['whitened_obvs'] = emg_obj.signal_dict['extend_obvs'].copy()

            for interval in range (nwins): 
                
                # initialise zero arrays for separation matrix B and separation vectors w
                emg_obj.decomp_dict['B_sep_mat'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],emg_obj.its])
                emg_obj.decomp_dict['w_sep_vect'] = np.zeros([np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],1])
                emg_obj.decomp_dict['MU_filters'] = np.zeros([nwins,np.shape(emg_obj.decomp_dict['whitened_obvs'][interval])[0],emg_obj.its])
                emg_obj.decomp_dict['SILs'] = np.zeros([nwins,emg_obj.its])
                emg_obj.decomp_dict['CoVs'] = np.zeros([nwins,emg_obj.its])
                emg_obj.decomp_dict['tracker'] =  np.zeros([1,emg_obj.its])
                emg_obj.decomp_dict['masked_mu_filters'] = []   # initialise empty list for the MU filters, because at each interval the removed MUs might not be the same
                
                emg_obj.convul_sphering(g,interval,tracker)
                
    #################### FAST ICA ########################################
                emg_obj.fast_ICA_and_CKC(g,interval,tracker)

                tracker = tracker + 1

    ##################### POSTPROCESSING #################################
            
            emg_obj.post_process_EMG(g)

    ####################### SAVING DECOMPOSITION OUTPUT INTO DATAFRAMES ########################
            
            # creating a decomposition dictionary as output, for use in the online tracking
            decomposition_dict = {'chan_name': emg_obj.signal_dict['electrodes'][g],'muscle_name': emg_obj.signal_dict['muscles'][g],
                                'coordinates': [emg_obj.coordinates[g]], 'ied': emg_obj.ied[g],'emg_mask': [emg_obj.rejected_channels[g]],
                                'mu_filters': [emg_obj.decomp_dict['masked_mu_filters']],'inv_extend_obvs': [emg_obj.signal_dict['inv_extend_obvs']],
                                'pulse_trains': [emg_obj.mu_dict['pulse_trains'][g]],'discharge_times': [emg_obj.mu_dict['discharge_times'][g]],
                                'clusters': 1 }

            # convert to data frame
            #decomposition_dict = pd.DataFrame(decomposition_b4df)
            all_dicts.append(decomposition_dict)

    # save dicitionaries into data file
    with open('decomposition_data.pkl','wb') as file:
         pickle.dump(all_dicts, file)
            
    
    if emg_obj.dup_bgrids and sum(emg_obj.mus_in_array) > 0:    
         emg_obj.post_process_across_arrays()
    
    print('Completed processing of the recorded EMG signal')
    print('Reformatting file for saving...')

    # creating a parameters data frame for saving processing parameters
    # assuming that all data collected in the same trial is recorded at the same frequency?
    parameters_dict = {
        'file_path': all_files[0], 'n_its':  emg_obj.its, 'ref_exist': emg_obj.ref_exist, 'fsamp': emg_obj.signal_dict['fsamp'],
        'target_thres': emg_obj.target_thres, 'nwins': nwins, 'check_EMG': emg_obj.check_emg,
        'drawing_mode': emg_obj.drawing_mode, 'differential_mode': emg_obj, 'peel_off': emg_obj.peel_off,
        'refine_MU': emg_obj.refine_mu, 'sil_thr': emg_obj.sil_thr, 'ext_factor': emg_obj.ext_factor,
        'edges': emg_obj.edges2remove, 'dup_thr': emg_obj.dup_thr, 'cov_filter': emg_obj.cov_filter, 
        'cov_thr': emg_obj.cov_thr, 'original_data': emg_obj.data, 'path': emg_obj.signal_dict['path'], 'target': emg_obj.signal_dict['target'] }
    
    # save parameters to json file
    with open('decompsotion_paraneters.pkl','wb') as file:
         pickle.dump(parameters_dict,file)

    print('Decomposed data saved.')


            
            


