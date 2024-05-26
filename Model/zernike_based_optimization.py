import time
import numpy as np
# import matplotlib.pyplot as plt
import utils as ut

size = 300
pad = 1000
order = 4
seed = np.array((42,32,11,89,24,73))
wave = 600e-9 #np.array((400e-9,600e-9,800e-9))

plot = True
custom = True

#%% Single wavefront analysis

error_PV = 12000e-9
if custom:
    ind = ut.zernike_indices(order)
    vector = np.zeros((np.shape(ind)[0],))
    vector[4] = 0#12000e-9 # 1200 nm defocus
    vector[7] = 0 # Vertical coma
    vector[1] = 12000e-9
else:
    vector=None
    
rand_wavefront, matrices, _ = ut.wavefront(size, order, seed[0], max_error=error_PV, in_vector=vector, plot=plot)

psf = ut.pupil_transform(rand_wavefront, pad, plot=plot, wavelength=wave)

ut.plot(rand_wavefront, psf, wave=wave, lim=0.3)

#%% Optimization
import utils as ut

plot = False
f_ev = 1000
order = 4
control_modes = 4
modes = np.shape(ut.zernike_indices(order))[0] - 1 # Numbder of modes minus the piston
with ut.History() as Hist:
    for i in range(np.shape(seed)[0]):
    
        a = ut.NelderMead(size, order, seed[i], if_plot=plot, wavelength=wave,
                      max_fev=f_ev)
    
        solution, result, opt_time, cost_hist_run, init_wf, zero_vector, wf, opt_time = a.nelder_mead()
    
        Hist.record_hist(init_wf, wf, cost_hist_run, opt_time)
        
    ut.plot_conv(Hist.cost_history, val=Hist.rms_impr_hist, val_0=Hist.opt_time_hist, modes=modes)


