import numpy as np
import scipy
from scipy import signal
from numpy import linalg
from scipy.fft import fft
import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import numba
from numba import jit

##################################### FILTERING TOOLS #######################################################

def notch_filter(signal,fsamp,to_han = False):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid 
    
    IMPORTANT!!! There is a small difference in the output of this function and that in MATLAB, best guess currently is differences in inbuilt FFT implementation."""

    bandwidth_as_index = int(round(4*(np.shape(signal)[1]/fsamp)))
    # width of the notch filter's effect, when you intend for it to span 4Hz, but converting to indices using the frequency resolution of FT
    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])


    for chan in range(np.shape(signal)[0]):

        if to_han:
            hwindow = scipy.signal.hann(np.shape(signal[chan,:])[0])
            final_signal = signal[chan,:]* hwindow
        else:
            final_signal = signal[chan,:]

        fourier_signal = np.fft.fft(final_signal)
        fourier_interf = np.zeros(len(fourier_signal),dtype = 'complex_')
        interf2remove = np.zeros(len(fourier_signal),dtype=np.int)
        window = fsamp
        tracker = 0
    
        for interval in range(0,len(fourier_signal)-window+ 1,window): # so the last interval will start at len(fourier_emg) - window
            
            # range(start, stop, step)
            median_freq = np.median(abs(fourier_signal[interval+1:interval+window+1])) # so interval + 1: interval + window + 1
            std_freq = np.std(abs(fourier_signal[interval+1:interval+window+1]))
            # interference is defined as when the magnitude of a given frequency component in the fourier spectrum
            # is greater than 5 times the std, relative to the median magnitude
            label_interf = list(np.where(abs(fourier_signal[interval+1:interval+window+1]) > median_freq+5*std_freq)[0]) # np.where gives tuple, element access to the array
            # need to shift these labels to make sure they are not relative to the window only, but to the whole signal
            label_interf = [x + interval + 1 for x in label_interf] # + 1 since we are already related to a +1 shifted array?
    
            if label_interf: # if a list exists
                for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                    
                    temp_shifted_list = [x + i for x in label_interf]
                    interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                    tracker = tracker + len(label_interf)
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove >= 0 , interf2remove <= len(fourier_signal)/2))[0]
        fourier_interf[interf2remove[indexf2remove]] = fourier_signal[interf2remove[indexf2remove]]
        corrector = int(len(fourier_signal) - np.floor(len(fourier_signal)/2)*2)  # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - np.fft.ifft(fourier_interf).real
      

    return filtered_signal


   

def bandpass_filter(signal,fsamp, emg_type = 'surface'):

    """ Generic band-pass filter implementation and application to EMG signal  - assuming that you will iterate this function over each grid """

    """IMPORTANT!!! There is a difference in the default padding length between Python and MATLAB. For MATLAB -> 3*(max(len(a), len(b)) - 1),
    for Python scipy -> 3*max(len(a), len(b)). So I manually adjusted the Python filtfilt to pad by the same amount as in MATLAB, if you don't the results will not match across
    lanugages. """   

    if emg_type == 0:
        lowfreq = 20
        highfreq = 500
        order = 2
    elif emg_type == 1:
        lowfreq = 100
        highfreq = 4400
        order = 3

    # get the coefficients for the bandpass filter
    nyq = fsamp/2
    lowcut = lowfreq/nyq
    highcut = highfreq/nyq
    [b,a] = scipy.signal.butter(order, [lowcut,highcut],'bandpass') # the cut off frequencies should be inputted as normalised angular frequencies

    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])
    # construct and apply filter
    for chan in range(np.shape(signal)[0]):
        
        filtered_signal[chan,:] = scipy.signal.filtfilt(b,a,signal[chan,:],padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return filtered_signal



def moving_mean1d(v,w):
    """ Moving average filter that replicates the method of movmean in MATLAB
    v is a 1 dimensional vector to be filtered via a moving average
    w is the window length of this filter """

    u = v.copy()
    w_temp = w
    n = len(v)-1

    if w_temp % 2 != 0:

        w = int(np.ceil(w_temp/2))
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-i:])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
        
            u[i] = np.mean(v[i - w + 1:i + w])

    else:

        w = int(w_temp/2)
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-(i+1):])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
            u[i] = np.mean(v[i - w:i + w ])

    return u
    

################################# CONVOLUTIVE SPHERING TOOLS ##########################################################

def extend_emg(extended_template, signal, ext_factor):

    """ Extension of EMG signals, for a given window, and a given grid. For extension, R-1 versions of the original data are stacked, with R-1 timeshifts.
    Structure: [channel1(k), channel2(k),..., channelm(k); channel1(k-1), channel2(k-1),...,channelm(k-1);...;channel1(k - (R-1)),channel2(k-(R-1)), channelm(k-(R-1))] """

    # signal = self.signal_dict['batched_data'][tracker][0:] (shape is channels x temporal observations)

    nchans = np.shape(signal)[0]
    nobvs = np.shape(signal)[1]
    for i in range(ext_factor):

        extended_template[nchans*i :nchans*(i+1), i:nobvs +i] = signal
  
    return extended_template


def whiten_emg(signal):
    
    """ Whitening the EMG signal imposes a signal covariance matrix equal to the identity matrix at time lag zero. Use to shrink large directions of variance
    and expand small directions of variance in the dataset. With this, you decorrelate the data. """

    # get the covariance matrix of the extended EMG observations

    cov_mat = np.cov(np.squeeze(signal),bias=True)
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eigh(cov_mat) 
    # in MATLAB: eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)
    sorted_evalues = np.sort(evalues)[::-1]
    penalty = np.mean(sorted_evalues[len(sorted_evalues)//2:]) # int won't wokr for odd numbers
    penalty = max(0, penalty)


    rank_limit = np.sum(evalues > penalty)-1
    if rank_limit < np.shape(signal)[0]:

        hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2

    # use the rank limit to segment the eigenvalues and the eigenvectors
    evectors = evectors[:,evalues > hard_limit]
    evalues = evalues[evalues>hard_limit]
    diag_mat = np.diag(evalues)
    # CHECK:np.linalg.solve can be faster than np.linalg.inv -> DONE: no difference here
    whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ np.transpose(evectors)
    dewhitening_mat = evectors @ np.sqrt(diag_mat) @ np.transpose(evectors)
    whitened_emg =  np.matmul(whitening_mat, signal).real 

    return whitened_emg, whitening_mat, dewhitening_mat



###################################### DECOMPOSITION TOOLS ##################################################################

# orthogonalisation update on 5th feb 20:24
@numba.njit(fastmath = True)
def ortho_gram_schmidt(w,B):

    """ This is the recommended method of orthogonalisation in Negro et.al 2016,
    documented in Hyvärinen et.al 2000 (fast ICA) """
    basis_projection = np.zeros(np.shape(w))
    for i in range(B.shape[1]):
        w_history = B[:, i]
        if np.all(w_history == 0):
            continue
        # Estimate the independent components one by one. When we have estimated p independent components, 
        # or p vectors w1; ...; wp; we run the one-unit fixed- point algorithm for wp11; 
        # and after every iteration step subtract from wp11 the “projections”  of the previously estimated p vectors, and then
        # renormalize 
        basis_projection = basis_projection + np.divide(np.dot(w, w_history),np.dot(w_history,w_history)) * w_history
    w = w - basis_projection
    return w

@numba.njit
def square(x):
    return np.square(x)

@numba.njit
def skew(x):
    return x**3/3

@numba.njit
def exp(x):
    return np.exp(-np.square(x)/2)

@numba.njit
def logcosh(x):
    return np.log(np.cosh(x))

@numba.njit
def dot_square(x):
    return 2*x

@numba.njit
def dot_skew(x):
    return 2*(np.square(x))/3

@numba.njit
def dot_exp(x):
    return -1*(np.exp(-np.square(x)/2)) + np.dot((np.square(x)), np.exp(-np.square(x)/2))

@numba.njit
def dot_logcosh(x):
    return np.tanh(x)


@numba.njit(fastmath=True)
def fixed_point_alg(w_n, B, Z,cf, dot_cf, its = 500,ortho_type='ord_deflation'):

    """ Update function for source separation vectors. The code user can select their preferred contrast function using a string input:
    1) square --> x^2
    2) logcosh --> log(cosh(x))
    3) exp  --> e(-x^2/2)
    4) skew --> -x^3/3 
    e.g. skew is faster, but less robust to outliers relative to logcosh and exp
    Upon meeting a threshold difference between iterations of the algorithm, separation vectors are discovered 
    
    The maximum number of iterations (its) and the contrast function type (cf) are already specified, unless alternative input is provided. """
   
    assert B.ndim == 2
    assert Z.ndim == 2
    assert w_n.ndim == 1
    assert its in [500]

    counter = 0
    its_tolerance = 0.0001
    sep_diff = np.ones(its)
    B_T_B = B @ B.T
    Z_meaner = Z.shape[1]

    while sep_diff[counter] > its_tolerance or counter >= its:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n.copy()
        # use update function to get new, current separation vector
        wTZ = w_n_1.T @ Z 
        A = dot_cf(wTZ).mean()
        w_n = Z @ cf(wTZ).T / Z_meaner  - A * w_n_1

        # orthogonalise separation vectors
        if ortho_type == 'ord_deflation':
            w_n -= np.dot(B_T_B, w_n)
        elif ortho_type == 'gram_schmidt':
            # as recommended in Negro et.al 2016
            w_n = ortho_gram_schmidt(w_n,B)

        # normalise separation vectors
        w_n /= np.linalg.norm(w_n)
        counter += 1
        sep_diff[counter] = np.abs(w_n @ w_n_1 - 1)

    return w_n

# updadted get spikes on feb 19th 11:26am
def get_spikes(w_n,Z, fsamp=2048, std_thr = 3):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1 ) # peaks variable holds the indices of all peaks
    source_pred /=  np.mean(maxk(source_pred[peaks], 10))
    
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        # remove outliers from the spikes cluster with a std-based threshold
        spikes = spikes[source_pred[spikes] <= np.mean(source_pred[spikes]) + std_thr*np.std(source_pred[spikes])]
    else:
        spikes = peaks

    return source_pred, spikes


def min_cov_isi(w_n,B,Z,fsamp,cov_n,spikes_n): 
    
    cov_n_1 = 2 * cov_n
    while cov_n < cov_n_1:

        cov_n_1 = cov_n.copy()
        spikes_n_1 = spikes_n.copy()
        # w_n = np.expand_dims(w_n,axis=1)
        w_n_1 = w_n.copy()
        _ , spikes = get_spikes(w_n,Z,fsamp)
        # determine the interspike interval
        ISI = np.diff(spikes/fsamp)
        # determine the coefficient of variation
        CoV = np.std(ISI)/np.mean(ISI)
        # update the sepearation vector by summing all the spikes
        w_n = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
       

    # if you meet the CoV minimisation condition, but with a single-spike-long train, use the updated
    # separation vector and recompute the spikes

    if len(spikes) < 2:
        _ , spikes = get_spikes(w_n,Z,fsamp)

    return w_n_1, spikes_n_1


################################ VALIDATION TOOLS ########################################

# updated silohuette measure on feb 5th 18:20pm

def maxk(signal, k):
    return np.partition(signal, -k, axis=-1)[..., -k:]

def get_silohuette(w_n,Z,fsamp):

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1 ) # this is approx a value of 20, which is in time approx 10ms
    source_pred /=  np.mean(maxk(source_pred[peaks], 10))
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        # indices of the spike and noise clusters (the spike cluster should have a larger value)
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        print()
        noise_ind = np.argmin(kmeans.cluster_centers_)
        # get the points that correspond to each of these clusters
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        noise = peaks[np.where(kmeans.labels_ == noise_ind)]
        # calculate the centroids
        spikes_centroid = kmeans.cluster_centers_[spikes_ind]
        noise_centroid = kmeans.cluster_centers_[noise_ind]
        # difference between the within-cluster sums of point-to-centroid distances for spikes (i.e. spikes spoints to spikes cluster centre)
        intra_sums = (((source_pred[spikes]- spikes_centroid)**2).sum()) 
        # difference between the between-cluster sums of point-to-centroid distance for spikes (i.e. spikes points to noise cluster centre)
        inter_sums = (((source_pred[spikes] - noise_centroid)**2).sum())
        sil = (inter_sums - intra_sums) / max(intra_sums, inter_sums)  

    else:

        sil = 0

    return source_pred, spikes, sil


def peel_off(Z,spikes,fsamp):

    windowl = round(0.05*fsamp)
    waveform = np.zeros([windowl*2+1])
    firings = np.zeros([np.shape(Z)[1]])
    firings[spikes] = 1 # make the firings binary
    EMGtemp = np.empty(Z.shape) # intialise a temporary EMG signal

    for i in range(np.shape(Z)[0]): # iterating through the (extended) channels
        temp = cutMUAP(spikes,windowl,Z[i,:])
        waveform = np.mean(temp,axis=0)
        EMGtemp[i,:] =  scipy.signal.convolve(firings, waveform, mode = 'same',method='auto')

    Z -= EMGtemp; # removing the EMG representation of the source spearation vector from the signal, avoid picking up replicate content in future iterations
    return Z
############################## POST PROCESSING #####################################################


def gausswin(M, alpha=2.5):
    
    """ Python equivalent of the in-built gausswin function MATLAB (since there is no open-source Python equivalent) """
    
    n = np.arange(-(M-1) / 2, (M-1) / 2 + 1,dtype=np.float128)
    w = np.exp((-1/2) * (alpha * n / ((M-1) / 2)) ** 2)
    return w

def cutMUAP(MUPulses, length, Y):

    """ Direct converion of MATLAB code in-lab. Extracts consecutive MUAPs out of signal Y and stores
    them row-wise in the out put variable MUAPs.
    Inputs: 
    - MUPulses: Trigger positions (in samples) for rectangualr window used in extraction of MUAPs
    - length: radius of rectangular window (window length = 2*len +1)
    - Y: Single signal channel (raw vector containing a single channel of a recorded signals)
    Outputs:
    - MUAPs: row-wise matrix of extracted MUAPs (algined signal intervals of length 2*len+1)"""
 
    while len(MUPulses) > 0 and MUPulses[-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]

    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = gausswin(2 * edge_len) # gives the same output as the in-built gausswin function in MATLAB
    # create the filtering window 
    win = np.ones(2 * length + 1)
    win[:edge_len] = tmp[:edge_len]
    win[-edge_len:] = tmp[edge_len:]
    MUAPs = np.empty((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length, 1) - (MUPulses[k] - length)
        end = MUPulses[k] + length- min(MUPulses[k] + length, len(Y))
        MUAPs[k, :] = win * np.concatenate((np.zeros(start), Y[max(MUPulses[k] - length, 1):min(MUPulses[k] + length, len(Y))+1], np.zeros(end)))

    return MUAPs



def batch_process_filters(dewhit_mat, mu_filters,inv_extend_obvs,extend_obvs,plateau,exfactor,diff,orig_sig_size,fsamp):

    """ dis_time: the distribution of spiking times for every identified motor unit, but at this point we don't check to see
    whether any of these MUs are repeats"""

    # Pulse trains has shape no_mus x original signal duration
    # dewhitening matrix has shape no_windows x exten chans x exten chans
    # mu filters has size no_windows x exten chans x (maximum of) x no iterations  --> less if iterations failed to reach SIL threshold

    mu_count = 0
    for win in range (np.shape(dewhit_mat)[1]):
        mu_count += np.shape(mu_filters[win])[1]
    
    pulse_trains = np.zeros([mu_count, orig_sig_size]) 
    discharge_times = [None] * mu_count
    mu_batch_count = 0

    for win_1 in range(np.shape(dewhit_mat)[1]):
        for exchan in range(np.shape(mu_filters)[1]):
            for win_2 in range(np.shape(dewhit_mat)[1]):

                dewhit_filters = (np.matmul(dewhit_mat, mu_filters[win_1][:,exchan])).T
                #pulse_trains[mu_batch_count, plateau[win_2*2]:plateau[(win_2+1)*2 - 1]+exfactor-1-diff] = np.matmul(np.matmul(dewhit_filters, iReSIG{nwin2}), eSIG{nwin2})

            pulse_trains[mu_batch_count,:] = pulse_trains[mu_batch_count,:]/ np.max(pulse_trains[mu_batch_count,:])
            pulse_trains[mu_batch_count,:] = np.multiply( pulse_trains[mu_batch_count,:],abs(pulse_trains[mu_batch_count,:])) 
            peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu_batch_count,:]), distance = np.round(fsamp*0.01) ) 
            
            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu_batch_count,peaks].reshape(-1,1)) 
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            discharge_times[mu_count] = peaks[np.where(kmeans.labels_ == spikes_ind)] 
            print(f"Batch processing MU#{mu_batch_count} out of {mu_count} MUs")
            mu_batch_count += mu_batch_count
    
    return pulse_trains, discharge_times

def xcorr(x,y):
    norm_x = x/np.linalg.norm(x)
    norm_y = y/np.linalg.norm(y)
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr

def remove_duplicates(pulse_trains, discharge_times, exfactor, tol = 0.3):

    spike_trains = np.zeros([np.shape(pulse_trains)[0],np.shape(pulse_trains)[1]])
    lag_limit = exfactor * 2
    # Making binary spike trains for each established MU
    for i in range(np.shape(pulse_trains)[0]):
        spike_trains[i,discharge_times[i]] = 1
    # With these binary trains, you can readily identify duplicate MUs
    while discharge_times:
        
        temp_discharge_times = [None] * len(pulse_trains)
        for mu in range(len(pulse_trains)):

            lags, corr = xcorr(spike_trains[0,:], spike_trains[mu,:])
            lim_lags = np.where(np.logical_and(lags>=-2*exfactor, lags<=2*exfactor))[0]
            corr_max, ind_max = max(corr[lim_lags])

            if corr_max > 0.2:
                temp_discharge_times[mu] = temp_discharge_times[mu] + lags[ind_max]
            else:
                temp_discharge_times[mu] = temp_discharge_times[mu]

        # Find common discharge times
        common_discharges = np.zeros(len(pulse_trains))
        for mu in range(1,len(pulse_trains)):

            common_discharges[mu] = 1

           

def remove_outliers(pulse_trains, discharge_times, fsamp, max_its = 30):

    for mu in range(np.shape(discharge_times)[0]):

        # isn't this quite a coarse way of calculating firing rate? i.e. without any smoothing?
        discharge_rates = 1/(np.diff(discharge_times[mu]) / fsamp)
        while (np.std(discharge_rates)/np.mean(discharge_rates)) > 0.4 and it < max_its:

            artifact_limit = np.mean(discharge_rates) + 3*np.std(discharge_rates)
            # identify the indices for which this limit is exceeded
            artifact_inds = np.squeeze(np.argwhere(discharge_rates > artifact_limit))
            if len(artifact_inds) > 0:

                # vectorising the comparisons between the numerator terms used to calculate the rate, for indices at rate artifacts
                diff_artifact_comp = pulse_trains[mu,discharge_times[mu][artifact_inds]] < pulse_trains[mu, discharge_times[mu][artifact_inds + 1]]
                # 0 means discharge_times[mu][artifact_inds]] was less, 1 means discharge_times[mu][artifact_inds]] was more
                less_or_more = np.argmax([diff_artifact_comp, ~diff_artifact_comp], axis=0)
                discharge_times[mu] = np.delete(discharge_times[mu], artifact_inds + less_or_more)

        discharge_rates = 1/(np.diff(discharge_times[mu]) / fsamp)
    
    return discharge_times


def refine_mus(signal,signal_mask, pulse_trains_n_1, discharge_times_n_1):

    """ signal is no_chans x time, where no_chans is the total for one grid
        signal_mask is the channels to discard
    
    signal.data(i*64-63:i*64,:), signal.EMGmask{i}, PulseT, distimenew);"""

    print("Refining MU pulse trains...")
    signal = [x for i, x in enumerate(signal) if signal_mask[i] != 1]
    nbextchan = 1500
    extension_factor = round(nbextchan/np.shape(signal)[0])
    extend_obvs = np.zeros([np.shape(signal)[0]*(extension_factor), np.shape(signal)[0] + extension_factor -1 ])
    extend_obvs = extend_emg(extend_obvs,signal,extension_factor)
    re_obvs = np.matmul(extend_obvs, extend_obvs.T)/np.shape(extend_obvs)[1]
    invre_obvs = np.linalg.pinv(re_obvs)
    pulse_trains_n = np.zeros([np.shape(pulse_trains_n_1)[0], np.shape(pulse_trains_n_1)[1]])
    discharge_times_n = [None] * len(pulse_trains_n_1)

    # recalculating the mu filters

    for mu in range(len(pulse_trains_n_1)):

        mu_filters = np.sum(extend_obvs[:,discharge_times_n_1[mu]],axis=1)
        IPTtmp = np.dot(np.dot(mu_filters.T,invre_obvs),extend_obvs)
        pulse_trains_n[mu,:] = IPTtmp[:np.shape(signal)[1]]

        pulse_trains_n[mu,:] = pulse_trains_n[mu,:]/ np.max(pulse_trains_n[mu,:])
        pulse_trains_n[mu,:] = np.multiply( pulse_trains_n[mu,:],abs(pulse_trains_n[mu,:])) 
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains_n[mu,:]))  # why no distance threshold anymore?
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains_n[mu,peaks].reshape(-1,1)) 
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        discharge_times_n[mu] = peaks[np.where(kmeans.labels_ == spikes_ind)] 

   
    print(f"Refined {len(pulse_trains_n_1)} MUs")

    return discharge_times_n
