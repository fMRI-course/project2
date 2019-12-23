import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
#from dipy.segment.mask import median_otsu
from scipy.ndimage import gaussian_filter
from matplotlib import colors
from dipy.segment.mask import median_otsu
from scipy.ndimage import gaussian_filter
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gamma
from scipy.interpolate import InterpolatedUnivariateSpline
from skimage import filters
import scipy.stats as stats
import numpy.linalg as npl
import os


#img_arte=nib.load('ds/sub_1/func/sub_1/115.nii')
#data_arte=img_arte.get_data()
#print(data_arte.shape)

def smooth_mask(data):
         """ Applies smoothing and computes mask. Applies mask to smoothed data """
         mean_data = np.mean(data,axis=-1)
         masked, mask = median_otsu(mean_data,2,1)
         smooth_data = gaussian_filter(data,[2,2,2,0])
         smooth_masked = smooth_data[mask]
         return smooth_masked.T

def masked_data(subject, run):
         data = bold_data(subject, run)
         mean_data = np.mean(data,axis=-1)
         masked, mask = median_otsu(mean_data,2,1)
         masked_data = data[mask]
         return masked_data.T

def smooth_data(subject, run):
         data = bold_data(subject, run)
         smooth_data = gaussian_filter(data,[2,2,2,0])
         return smooth_data

def find_arteries(data):
    data_1d=data.ravel()
    data_small=data[40:-40, 90:-90,10:210]
    pct_99 = np.percentile(data_small, 99.5)
    binarized_data = data_small > pct_99
    #binarized_data=np.reshape(binarized_data,(64,64,30,173))
    return(binarized_data)

def binarized_surface(binary_array):
    """ Do a 3D plot of the surfaces in a binarized image

    The function does the plotting with scikit-image and some fancy
    commands that we don't need to worry about at the moment.
    """
    # Here we use the scikit-image "measure" function
    verts, faces = measure.marching_cubes_classic(binary_array, 0)
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles\
    mesh = Poly3DCollection(verts[faces], linewidths=0, alpha=0.5)
    #ax.add_collection3d(mesh)
    #ax.set_xlim(0, binary_array.shape[0])
    #ax.set_ylim(0, binary_array.shape[1])
    #ax.set_zlim(0, binary_array.shape[2])
    #plt.show()
#m=find_arteries(data_arte)
#binarized_surface(m)


def events2neural(task_fname, tr, n_trs):
    """ Return predicted neural time course from event file

    Parameters
    ----------
    task_fname : str
        Filename of event file
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run

    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    task = np.loadtxt(task_fname)
    if task.ndim != 2 or task.shape[1] != 3:
        raise ValueError("Is {0} really a task file?", task_fname)
    time_course = np.zeros(n_trs)
    for i in range(n_trs):
        for onset, duration,amplitude in task:
            if tr*i>=onset and tr*i<=onset+2:
                time_course[i]=amplitude
                break
    #print(time_course)
    '''
    task[:, :2] = task[:, :2] / tr   #the time detected in this volumn
    print(task)
    time_course = np.zeros(n_trs)
    for onset, duration, amplitude in task:
        onset, duration = int(onset),  int(duration)
        time_course[onset:onset + duration+1] = amplitude
    '''


    return time_course

def mt_hrf(times):
    """ Return values for HRF at given times

    This is the "not_great_hrf" from the "make_an_hrf" exercise.
    Feel free to replace this function with your improved version from
    that exercise.
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def design_matrix(data):
    tr=2.5
    neural_prediction = events2neural(data,tr,n_trs)
    #plt.plot(neural_prediction)

    neural_prediction_list=[]
    for j in range(1,9):
        for i in range(1, neural_prediction.shape[-1]+1):
            if TR*i>(36*j+12):
                if(j==1):
                    low_bound=5
                else:
                    low_bound=low_bound+data_classified.shape[-1]
                data_classified=np.zeros((i-1-low_bound,1))
                data_classified=neural_prediction[low_bound:i-1]
                neural_prediction_list.append(data_classified)
                #print(data_classified.shape[0])
                break

    for i in range(8):
        for j in range(neural_prediction_list[i].shape[0]):
            if neural_prediction_list[i][j]!=1 and neural_prediction_list[i][j]!=0:
                neural_prediction_list[i][j]=1

    #difference = task_2_scan.mean(axis=-1) - rest_2_scan.mean(axis=-1)
    #print(difference.shape)
    #plt.imshow(difference[:, :, 20], cmap='gray')

    #hrf
    convol_x=[]
    for i in range(8):
        hrf_times = np.arange(0,35,tr)
        hrf_signal = mt_hrf(hrf_times)
        #plt.plot(hrf_times,hrf_signal)

        #- Convolve predicted neural time course with HRF samples
        hemodynamic_prediction = np.convolve(neural_prediction_list[i], hrf_signal)

        X = np.ones((len(neural_prediction_list[i]),2))
        #- Remove extra tail of values put there by the convolution
        hemodynamic_prediction_R = hemodynamic_prediction[:len(neural_prediction_list[i])]
    #- Plot convolved neural prediction and unconvolved neural prediction
        X[:,1]=hemodynamic_prediction_R

        #plt.figure()
        #plt.plot(neural_prediction[4:], label = 'unconvolved')
        #plt.plot(hemodynamic_prediction_R, label = 'convolved')
        #plt.legend()
        #plt.savefig('1.jpg')
        convol_x.append(X)

    return(convol_x)


#x=design_matrix(data)


def time_correction(data):

    #: Set defaults for plotting
    #plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    slice_0_ts = data[23,19,0,:]
    slice_1_ts = data[23,19,1,:]

    slice_0_times = np.arange(data.shape[-1]) * 2.5

    slice_1_times = slice_0_times + TR / 2

    interp = InterpolatedUnivariateSpline(slice_1_times,slice_1_ts,k=1)

    slice_1_ts_est = interp(slice_0_times)

    #plt.plot(slice_0_times[:10],slice_0_ts[:10], ':+')
    #plt.plot(slice_1_times[:10],slice_1_ts[:10], ':+')
    #plt.plot(slice_0_times[:10],slice_1_ts_est[:10], 'kx')
    #min_y,max_y=plt.ylim()
    #for i in range(1,10):
     #   t=slice_0_times[:10]
      #  plt.plot([t,t],[min_y,max_y],'k:')
    #plt.show()


    acquisition_order = np.zeros(64)
    acquisition_index = 0
    for i in range(0,64,2):
        acquisition_order[i] = acquisition_index
        acquisition_index += 1
    for i in range(1,64,2):
        acquisition_order[i] = acquisition_index
        acquisition_index += 1
    slice_0_times = np.arange(data.shape[-1]) * 2.5
    time_offsets = acquisition_order / 64 * TR
    new_data = data.copy()
    for z in range(data.shape[2]):
        slice_z_times = slice_0_times + time_offsets[z]
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                time_series = data[x,y,z,:]
                interp = InterpolatedUnivariateSpline(slice_z_times,time_series,k=1)
                new_series = interp(slice_0_times)
                new_data[x,y,z,:] = new_series
    return(new_data)

def mask(data):
    mean = data.mean(axis=-1)
    thresh = filters.threshold_otsu(mean)
    # The mask has True for voxels above "thresh", False otherwise
    mask = mean > thresh
    #plt.imshow(data[:,:,32,1])

    #plt.imshow(mask[:, :, 32], cmap='gray')


    #print(brain_voxel / n_voxels)
    Y = data[mask].T

    return(Y)

#x=time_correction(data)
#y=mask(x)





def classify_Y(data):
    #classification
    '''
    data_classifieds=[]
    for j in range(1,9):
        for i in range(data.shape[-1]):
            if TR*i>(36*j):
                if(j==1):
                    low_bound=0
                else:
                    low_bound=low_bound+data_classified.shape[-1]
                data_classified=np.zeros((data.shape[0],data.shape[1],data.shape[2],i))
                data_classified=data[...,low_bound:i-1]
                data_classifieds.append(data_classified)
                break
    '''
    data_classifieds=[]
    for j in range(1,9):
        for i in range(1, data.shape[-1]+1):
            if TR*i>(36*j+12):
                if(j==1):
                    low_bound=5
                else:
                    low_bound=low_bound+data_classified.shape[-1]
                data_classified=np.zeros((i-1-low_bound,1))
                data_classified=data[...,low_bound:i-1]
                data_classifieds.append(data_classified)
                #print(data_classified.shape[0])
                break
    return(data_classifieds)


def pca(data):
    vol_shape = data.shape[:-1]
    n_vols = data.shape[-1]
    N = np.prod(vol_shape)
    arr = data.reshape(N, n_vols).T
    row_means = np.outer(np.mean(arr, axis=1), np.ones(N))
    X = arr - row_means
    unscaled_covariance = X.dot(X.T)
    U, S, VT = npl.svd(unscaled_covariance)
    C = U.T.dot(X)
    C_vols = C.T.reshape(data.shape)
    plt.imshow(data[:,:,32,1])
    plt.figure()
    plt.imshow(C_vols[:,:,32,1])
    plt.show()
    return(C_vols)



def linear_regression(x,y,i_person,j_sub):
    X=design_matrix(x)
    classify_data=classify_Y(y)

    for i in range(8):
        time_correction_data=time_correction(classify_data[i])
        #pca_data=pca(time_correction_data)
        #Y=mask(pca_data)
        Y=mask(time_correction_data)

        mean = time_correction_data.mean(axis=-1)
        thresh = filters.threshold_otsu(mean)
        mask_1=mean>thresh


        #linear regression
        B= npl.pinv(X[i]).dot(Y)
        fitted = X[i].dot(B)
        E = Y - fitted
        n=X[i].shape[0]
        df=n-npl.matrix_rank(X[i])
        sigma_2 = np.sum(E ** 2, axis=0) / df
        # c and c_b_cov are the same as before, but recalculate anyway
        c = np.array([0, 1])
        c_b_cov = c.dot(npl.pinv(X[i].T.dot(X[i]))).dot(c)
        t = c.T.dot(B) / np.sqrt(sigma_2 * c_b_cov)
        #print(Y.shape[:3])
        t_3d = np.zeros((40,64,64))

        #print(t.shape)
        t_3d[mask_1] = t

        #t_3d=np.reshape(t_3d,(40,64,64,t_3d.shape[0]))
        #plt.imshow(t[:,:,32])
        #plt.imshow(t_3d[:, :, 32], cmap='gray')
        t_dist = stats.t(df)
        p = 1 - t_dist.cdf(t)
        p_3d = np.zeros((40,64,64))
        p_3d[mask_1] = p
        #print(p)
        path='data/person_'+str(i_person)+'/sub_'+str(j_sub)+'/'
        if not os.path.isdir(path):
            os.makedirs(path)
        #plt.imshow(p_3d[:, :, 32],cmap='inferno')
        #plt.savefig(path+'stimu_'+str(i)+'.jpg')
        np.save(path+'stimu_'+str(i)+'.npy',p_3d)
        #plt.show()


for i in range(1,7):
    for j in range(1,11):
        TR=2.5
        if j <10:
            img=nib.load('ds000105_R2.0.2/sub-'+str(i)+'/func/sub-'+str(i)+'_task-objectviewing_run-0'+str(j)+'_bold.nii.gz')
            data=img.get_data()
        else:
            img=nib.load('ds000105_R2.0.2/sub-'+str(i)+'/func/sub-'+str(i)+'_task-objectviewing_run-'+str(j)+'_bold.nii.gz')
            data=img.get_data()
        n_trs=data.shape[-1]
        #print(data.shape)
        x=linear_regression('onsets/'+str(j)+'.tsv',data,i,j)

'''
n_trs=121
TR=2.5
x=design_matrix('onsets/'+str(1)+'.tsv')
'''
