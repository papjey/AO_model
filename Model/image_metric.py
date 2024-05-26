# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:27:29 2023

@author: papuckaj
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import time


class NelderMead():
    '''Nelder Mead optimization algorithm; 
    size =  size of the matrix N x N; 
    order = order of zernike modes to consider; 
    seed = seed for random disturbance; 
    if_plot =  determines plotting of initial PSF and WF; 
    pad =  determines the relative resolution of the PSF as well as influences computing time; 
    spacing =  determines how often the intermediate plots should be made.; '''
    def __init__(self, size, order, seed, if_plot=False, pad=2000, spacing=20, max_error=12000e-9, wavelength=400e-9,
                 max_fev=100, tol_tilt=0.005, tol_fine=0.0007, simplex=1e-5):
        self.history = []
        self.center_hist = []
        self.spacing = spacing # for plotting interval
        self.wave = wavelength
        matrices = zernike_matrices(size, order)
        mat_scale = np.max(np.abs(np.sum(matrices, axis=0)))
        self.matrices = matrices/mat_scale
        self.zero_vector = random_vector(seed, order)*max_error
        self.pad = pad
        self.model_time = 0
        self.model_t_time = 0
        self.length = np.shape(zernike_indices(order))[0]
        self.size = size
        self.seed = seed
        self.order = order
        self.iteration = 0
        self.if_plot = if_plot
        self.max_fev = max_fev
        self.wf = 0
        self.max_error = max_error
        self.tol_tilt = tol_tilt
        self.tol_fine = tol_fine
        self.simplex = simplex
        
        #Scaling of the initial zero_vector
        correct_wf = controlled_wavefront(self.matrices, self.zero_vector)
        rms,pv,_,_ = error(crop(correct_wf))
        print("rms is:",rms)
        scale = max_error/pv
        self.zero_vector = self.zero_vector*scale
        self.wf = controlled_wavefront(self.matrices, self.zero_vector)
        
        rms,pv,_,_ = error(crop(self.wf))
        plot_wf(crop(self.wf))
        
        
    def f(self, x):
        '''This function generates surface of the mirror by using zernike matrices,
        and random coeficient vector with uniform distribution which is scale to have
        PV values of specified max_error. This coeffiecient vector "zero_vector" is 
        used to offset the initial mirror surface and to imitate SFE of the mirror.
        The function is has input of vector x which alters the zero_vector such 
        that to offset it and makes the matrices zero. To evaluate the cost of
        the function which consist of two parts: first the fourier transform is
        performed to simulate the PSF and the quality of the PSF determines the cost.
        Second, it is determined how far the PSF is from the center of the camera,
        to compensate for tip and tilt modes as these does not effect the first part
        of the cost function. Wavefront and FFT is timed to substract the time from
        the total optimization time, as in real situation this time would not
        contribute to the overall time in real situation.'''
        
        # Set coeficients of the Zernike
        start_time = time.time()
        if self.count == 1:
            fine_x = np.concatenate((self.solution[:3],x))
            c_wf = controlled_wavefront(self.matrices, self.zero_vector+fine_x)
        else:   
            c_wf = controlled_wavefront(self.matrices, self.zero_vector+x)
        
        # Record & report cost     
        # get psf image
        
        psf = pupil_transform(c_wf, self.pad, wavelength=self.wave)
        end_time = time.time()
        self.model_time += ( end_time - start_time)
           
        plot = self.iteration % self.spacing == 0 and self.if_plot
        if self.iteration == 0:
            plot = True
        
        val, _, _ = cost(psf, self.wave, plot=plot) #requires normalization to be within same range of coeficients. 
        
        self.history.append(val)
        psf_val = val/self.history[0]
        
        print("Cost:",val,"Elapsed modeling time",int(self.model_time),"F_ev",self.iteration)
        self.iteration += 1
        
        return psf_val
    
    def f_t(self, x_t):
        
        start_time = time.time()
        x = x_t[0]
        y = x_t[1]
        self.solution[1] = x
        self.solution[2] = y 
        
        c_wf = controlled_wavefront(self.matrices, self.zero_vector+self.solution)
        psf = pupil_transform(c_wf, self.pad, wavelength=self.wave)
        
        end_time = time.time()
        self.model_t_time += ( end_time - start_time)
        
        
        #plot = self.iteration % self.spacing == 0 
        val_t, x_cm, y_cm = cost(psf, self.wave, plot=False) #requires normalization to be within same range of coeficients. 
        
# =============================================================================
#         This function calculate the distance from the center of the image matrix
#         and makes a vector from the distances by Pythagaros theorem. Then all
#         further vectors will be scaled by initial value to normalize it and keep
#         it under 1. 
# =============================================================================
        x_wf_cent = int(psf.shape[0]/2)
        y_wf_cent = int(psf.shape[1]/2)

        #self.history.append(val_t)
        #psf_val = val_t/self.history[0]
        cent_x = np.square(np.abs(x_wf_cent-x_cm))
        cent_y = np.square(np.abs(y_wf_cent-y_cm))
        
        center_val = np.sqrt(cent_x + cent_y)
        
        self.center_hist.append(center_val)
        print("Cost:",center_val, 'PSF cost', val_t,"Elapsed modeling time",int(self.model_t_time),"F_ev",self.iteration)
        center_val = center_val/self.center_hist[0]
        
        self.iteration += 1
        
        return center_val
    
    def nelder_mead(self):
        """Options require simplex of size (N+1,N) """
        x0 = np.zeros(self.length)
        x_t_0 = np.zeros(2)
        
        u = np.ones((1,self.length))*self.simplex
        u_t = np.ones((1,2))*self.simplex
        simplex = np.concatenate((np.eye(self.length)*self.simplex,u),axis=0)
        simplex_t = np.concatenate((np.eye(2)*self.simplex,u_t),axis=0)
        
        opts = {'maxfev': self.max_fev, 'adaptive': True,
                'initial_simplex': simplex}
        opts_t = {'maxfev': self.max_fev, 'adaptive': True,
                'initial_simplex': simplex_t}
        # 'initial_simplex': simplex
        
        mini_start_time = time.time()
        self.count = 0
        result = minimize(self.f, x0, method='Nelder-Mead', tol=self.tol_fine, options=opts)
        self.solution = result['x']
        self.solution[0] = -self.zero_vector[0]
        
        evaluation = result['fun']
        
        result_t = minimize(self.f_t, x_t_0, method='Nelder-Mead', tol=self.tol_tilt, options=opts_t)
        solution_t = result_t['x']
        print('solution', solution_t, 'zero_vect',self.zero_vector[1:3])
        
# =============================================================================
#         print('remainder', self.zero_vector+self.solution)
# =============================================================================
        evaluation = result_t['fun']
        
        mini_end_time = time.time()
        minimization_time = mini_end_time - mini_start_time - self.model_time - self.model_t_time
        
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        print('Solution: f(%s) = %.5f' % (self.solution, evaluation))
        
# =============================================================================
#         Total expected time of the whole process, 1/32 is framerate of the camera
#         minimizataion time is determined by how long the minimization takes 
#         excluding the FFT time, but including cost determination, function evaluation
#         and minimization.
# =============================================================================
        
        opt_time = result['nfev']*(1/32) + minimization_time 
        
# =============================================================================
#         This part plots the final optimized wavefront and it's psf 
# =============================================================================
        wf = controlled_wavefront(self.matrices, self.zero_vector+self.solution)
        psf = pupil_transform(wf, self.pad, plot=True, wavelength=self.wave)
        plot(psf, self.wave, lim=0.3)
        plot_wf(crop(wf))
        cost(psf.real, self.wave, plot=True)
        
        return self.solution, result, minimization_time, self.history, crop(self.wf), self.zero_vector, wf, opt_time

def crop(im, crop_frac=1, plot=False, imag=False):
    if crop_frac < 0 or crop_frac > 1:
        print('crop_frac must have a value between 0 and 1')
        return im

    center_x, center_y = im.shape[1] // 2, im.shape[0] // 2
    radius_x = (im.shape[1] - center_x) * crop_frac
    radius_y = (im.shape[0] - center_y) * crop_frac
    
    rows, cols = np.indices(im.shape)
    condition = (((rows - center_y) ** 2) / (radius_y ** 2) + ((cols - center_x) ** 2) / (radius_x ** 2)) > 1
    
    if imag:
        im[condition] = 0 + 0j
    else:
        im[condition] = 0 
    if plot:
        plt.figure(dpi=500)
        plt.imshow(im.real, cmap="viridis")
        cbar = plt.colorbar()
        cbar.set_label('Z Displacement', rotation=90)
        plt.show()
    return im

def padding(matrix, pad=0, y_pad=0, symm=True, plot=False):
    
    if symm:
        y_pad=pad
        
    top_padding = y_pad
    bottom_padding = top_padding
    left_padding = pad
    right_padding = left_padding
    
    # Calculate the new dimensions of the padded matrix
    padded_height = matrix.shape[0] + top_padding + bottom_padding
    padded_width = matrix.shape[1] + left_padding + right_padding

    # Create a new matrix with zeros and the desired dimensions
    padded_matrix = np.zeros((padded_height, padded_width), dtype=matrix.dtype)

    # Copy the original matrix into the center of the padded matrix
    padded_matrix[top_padding:top_padding + matrix.shape[0],
                  left_padding:left_padding + matrix.shape[1]] = matrix
    if plot:
        plt.figure(dpi=500)
        plt.imshow(padded_matrix.real, cmap="viridis")
        plt.show()
    return padded_matrix

def zernike_indices(order):
    
    zernike_indices = []
    
    for n in range(order):
        for m in range(-n, n + 1, 2):
            zernike_indices.append([n, m])
            
    return zernike_indices

def zernike_matrices(size, order):
    
    indices=zernike_indices(order)
    x = np.linspace(-1, 1, size)
    y = np.linspace(1, -1, size)
    x, y = np.meshgrid(x, y)
    t, r = np.arctan2(y, x), np.sqrt(x**2 + y**2)
    
    zernikeMatrices = []
    
    for i in range(len(indices)):
        zernikeMatrix = zernike(r, t, indices[i][0], indices[i][1])
        zernikeMatrices.append(zernikeMatrix)
        
    return zernikeMatrices

def zernike(r, t, n, m):
    
    if m < 0:
        zern = -zernike_radial(r, n, -m) * np.sin(-m * t)
    else:
        zern = zernike_radial(r, n, m) * np.cos(m * t)
    return zern   
        
def zernike_radial(r, n, m):
    
    
    if (n - m) % 2 == 1:
        raise ValueError('n-m must be even')

    if n < 0 or m < 0:
        raise ValueError('n and m must both be positive in radial function')
    
    if n != int(n) or m != int(m):
        raise ValueError('n and m must both be integers')
    
    if n == m:
        radial = r**n
    elif n - m == 2:
        radial = n * zernike_radial(r, n, n) - (n - 1) * zernike_radial(r, n - 2, n - 2)
    else:
        H3 = (-4 * ((m + 4) - 2) * ((m + 4) - 3)) / ((n + (m + 4) - 2) * (n - (m + 4) + 4))
        H2 = (H3 * (n + (m + 4)) * (n - (m + 4) + 2)) / (4 * ((m + 4) - 1)) + ((m + 4) - 2)
        H1 = ((m + 4) * ((m + 4) - 1) / 2) - (m + 4) * H2 + (H3 * (n + (m + 4) + 2) * (n - (m + 4))) / 8
        radial = H1 * zernike_radial(r, n, m + 4) + (H2 + H3 / r**2) * zernike_radial(r, n, m + 2)
    # Fill in NaN values that may have resulted from DIV/0 in prior line.
    # Evaluate these points directly (non-recursively) as they are scarce if present.

    if np.isnan(radial).sum() > 0:
        row, col = np.where(np.isnan(radial))
        c = 0
        while c < len(row):
            x = 0
            for k in range(int((n - m) / 2) + 1):
                x += ((-1) ** k * np.math.factorial(n - k)) / (np.math.factorial(k) * np.math.factorial((n + m) // 2 - k) * np.math.factorial((n - m) // 2 - k)) * 0 ** (n - 2 * k)
            radial[row[c], col[c]] = x
            c += 1

    return radial

def pupil_transform(poly, pad, plot=False, wavelength=500e-9):
    pup = np.exp((2 * np.pi * 1j * poly)/wavelength)
    pup_crop = padding(crop(pup, 1, plot=plot, imag=True), pad, plot=plot)
    amp = np.fft.fftshift(np.fft.fft2(pup_crop))
    psf = amp * np.conj(amp)
    psfmax = np.max(psf)
    psf /= psfmax
    return psf.real

def random_vector(seed, order):
    '''Random vector generation to simulate uneven mirror, the instances of the
    array act as coeficients to the zernike modes; Although the uniform 
    distribution is between -1 and 1 the vector has to be normalized.; To force
    the total wavefront to be within range of 1, this is done with optimization
    in mind as it is sensitive to scaling. '''
    
    np.random.seed(seed)
    indices=zernike_indices(order)
    random_array = np.random.uniform(-1, 1, np.shape(indices)[0]-1)
    random_vect = np.concatenate(([0],random_array)) # add piston mode as zero

    return random_vect

def wavefront(size, order, seed, max_error=3000e-9, in_vector=None, plot=False):
    '''creates wavefront from either random coeficient vector or input vector through in_vector.
    The function multiplies zernike mode matrices with respectful coeficients and sums all the matrices element wise.
    Resulting in final combined wavefront which is cropped to have circular pupil shape. 
    This method was used for single WF analysis and for optimization separated to random_matrices and controlled_wavefront methods'''
    
    matrices=zernike_matrices(size, order)
    random_array_normalized=random_vector(seed, order)
    
    if in_vector is not None:
        random_array_normalized = in_vector

    wavefront = np.sum(
        (matrices * 
         random_array_normalized[:, np.newaxis, np.newaxis])
        , axis=0)
    
    wavefront = crop((wavefront/np.max(np.sqrt(np.square(wavefront))))*max_error)
    _,pv,_,_ = error(wavefront)
    
    scale = (max_error/pv)
    wavefront = wavefront*scale
    
    return wavefront, matrices, scale

def controlled_wavefront(matrices, control):
    
    controlled_wavefront = np.sum(
        (matrices * 
         control[:, np.newaxis, np.newaxis])
        , axis=0) 
    
    return controlled_wavefront

def scale_vector(wf):
    scale = np.max(np.abs(wf))
    print(scale)
    return scale

def cost(im, wave=590e-9, bound=0, plot=False):
    # Ignore pixels below 50, needed to combat noise    
    # im[im < bound] = 0.0
    
    # if im.max() == 0.0:
    #     raise Exception('All values are zero!')
    
    x_arr = np.arange(im.shape[0], dtype=float)
    y_arr = np.arange(im.shape[1], dtype=float)
    
    tot = im.sum()
    
    y_cm = im.sum(axis=0).dot(y_arr)/tot
    x_cm = im.sum(axis=1).dot(x_arr)/tot
    
    # Shift position arrays to cm
    x_arr -= x_cm
    y_arr -= y_cm
    
    # Square
    x_arr *= x_arr
    y_arr *= y_arr
    
    # Radial distance squared
    r2 = x_arr[:,None]+y_arr
    
    # Moment of intertia
    moi = (r2 * im).sum()
    c = moi/im.sum()
    
    # Draw on figure (used only for debug!)
    if plot:
        
        intensity = 0.8
        im[:,int(y_cm)] = intensity
        im[int(x_cm),:] = intensity
        
        # Equivalent radius of fully filled in circle
        r = (moi * 4/np.pi)**0.25/4
        
        x1 = int(x_cm - r)
        x2 = int(x_cm + r)
        y1 = int(y_cm - r)
        y2 = int(y_cm + r)
        
        offset = int(r*3)
        x_l_lim = y_cm-offset
        x_u_lim = y_cm+offset
        y_u_lim = x_cm+offset
        y_l_lim = x_cm-offset
        
        
        if not (0 < x1 < im.shape[0] or 0 < x2 < im.shape[0] or
                0 < y1 < im.shape[1] or 0 < y2 < im.shape[1]):
            raise Exception('Out of bounds! Lights are on!')
        
        im[x1,y1:y2] = intensity
        im[x2,y1:y2] = intensity
        im[x1:x2,y1] = intensity
        im[x1:x2+1,y2] = intensity
         
        plt.figure(num=2, dpi=500)
        plt.pcolormesh(im, shading='auto', cmap='viridis')
        plt.xlim([x_l_lim, x_u_lim])
        plt.ylim([y_u_lim, y_l_lim]) # for some reason the plot pf just the PSF is inverted in Y axis thus it reuquires bound inversion to keep the psf in the same orientation
        plt.title(f'Cost_value: {c:.0f}')
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()
        plt.clf()
        
        x = np.linspace(-1, 1, np.shape(im)[0])
        y = np.linspace(1, -1, np.shape(im)[1])
        
        formatted_wave = '{:.0f}'.format(wave * 1e9)
        plt.figure(num=3, dpi=500)
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, im, shading='auto', cmap='viridis')
        plt.title('PSF at {} nm'.format(formatted_wave))
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.show()
        plt.clf()
    return c, x_cm, y_cm

def plot(psf, wave, lim=0.05):
    
    formatted_wave = '{:.0f}'.format(wave * 1e9)
    
    x = np.linspace(-1, 1, np.shape(psf)[0])
    y = np.linspace(1, -1, np.shape(psf)[1])
        
    plt.figure(num=2, dpi=500)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, psf, shading='auto', cmap='viridis')
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.title('PSF at {} nm'.format(formatted_wave))
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.show()
    plt.clf()
    
def plot_conv(cost_history, val=0, val_0=0, modes=0):
    plt.figure(num=0, dpi=500)
    plt.title('Optimization convergence, {} modes'.format('{:2.0f}'.format(modes)))
    plt.xlabel('Evaluation count')
    plt.ylabel('Cost value')
    for i in range(np.shape(cost_history)[0]):
        plt.plot(cost_history[i], label='WF RMS {} nm, opt. time {} s'.format('{:3.0f}'.format(val[i]*1e9), '{:.0f}'.format(val_0[i])))
    plt.legend()
    plt.savefig('Optimization convergence.png')
        
    plt.show()
    plt.clf()
    
def plot_wf(im):
    
    rms,_,_,_ = error(im)
    print(rms)
    f_pv = '{:.0f}'.format(rms*1e9)
    plt.figure(num=4, dpi=500)
    plt.imshow(im, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label('Z Displacement', rotation=90)
    plt.title('Wavefront, RMS {} nm'.format(f_pv))
    plt.gca().set_aspect('equal')
    plt.show()

def error(wf, init_wf=0):

    rms = np.sqrt(np.mean(np.square(wf)))
    rms_init = np.sqrt(np.mean(np.square(init_wf)))
    
    pv_wf = np.max(wf) - np.min(wf)
    pv_wf_init = np.max(init_wf) - np.min(init_wf)
    
    #impr_rms = (rms_init / rms) * 100
    
    return rms, pv_wf, pv_wf_init, rms_init

class History():
    
    def __init__(self):
        self.cost_history = []
        self.rms_impr_hist = np.empty((0))
        self.pv_impr_hist = np.empty((0))
        self.pv_hist = np.empty((0))
        self.init_pv_hist = np.empty((0))
        self.opt_time_hist = []
        self.rms_init_hist = np.empty((0))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the History")
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return self.cost_history, self.rms_impr_hist, self.pv_impr_hist, self.init_pv_hist, self.pv_hist, self.opt_time_hist
            
    def record_hist(self, init_wf, wavefront, cost_hist_run, opt_time):    
        
        rms_impr, pv_wf, init_wf_pv, rms_init = error(wavefront, init_wf)
        
        self.cost_history.append(cost_hist_run)
        self.rms_impr_hist = np.append(self.rms_impr_hist, rms_impr)
        self.init_pv_hist = np.append(self.init_pv_hist, init_wf_pv)
        self.pv_hist = np.append(self.pv_hist, pv_wf)
        self.opt_time_hist.append(opt_time)
        self.rms_init_hist = np.append(self.rms_init_hist, rms_init)