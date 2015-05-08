import pyfits
import emcee
import numpy as np
import os

#Se leen los pixeles que se determinaron relevantes
with open('pixels_bono.txt','r') as f:
	raw_data = f.readlines()

pixels_x = []
pixels_y = []
for line in raw_data:
	coords = line.rsplit(",")
	pixels_x.append(int(coords[0]))
	pixels_y.append(int(coords[1]))
	
#Se carga el archivo .fits
fits_data = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
fits_data = fits_data[0].data

#Se cargan los intervalos de tiempo del archivo intervals.csv
with open('intervals.csv','r') as f:
	time = f.readlines()
time.pop(0)
time = [float(i) for i in time]

file_name = 'likelihood_paso.txt'

if os.path.isfile(file_name):
	os.remove(file_name)

with open(file_name, 'a') as myfile:
	myfile.write('x y likelihood f g h n t0\n')


#Modelo Paso
def my_model(x, f, g, h, n, t0):
	model = [f + g*t + h*(1 + (2.0/np.pi)*np.arctan(n*(t-t0))) for t in x]
	return model

def lnprior(theta):
	f, g, h, n, t0 = theta
	if -10.0 < f < 40.0 and -0.5 < g < 0.5 and -10 < h < 100 and -2 < n < 2 and -200 < t0 < 200:
		return 0.0
	return -np.inf

def lnlike(theta, x, y):
	f, g, h, n, t0 = theta
	model = my_model(x, f, g, h, n, t0)
	return -0.5*(np.sum((y-model)**2))

def lnprob(theta, x, y):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y)

def likelihood(y_obs, y_model):
    chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
    return -chi_squared

ndim = 5
nwalkers = 10
nsteps = 2000
pos = [[np.random.random()*10, np.random.random()*0.1, np.random.random()*100, np.random.random()*0.5, np.random.random()*100] for i in range(nwalkers)]

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, pixel_data))
	sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state( ))
	samples_fc = sampler.flatchain
	
	f_values = []
	g_values = []
	h_values = []
	n_values = []
	t0_values = []

	i = 0

	for element in samples_fc:
		if i > 100:
			f_values.append(element[0])
			g_values.append(element[1])
			h_values.append(element[2])
			n_values.append(element[3])
			t0_values.append(element[4])
		else:
			i = i + 1

	freq_f, bins_f = np.histogram(f_values, bins=100)
	freq_g, bins_g = np.histogram(g_values, bins=100)
	freq_h, bins_h = np.histogram(h_values, bins=100)
	freq_n, bins_n = np.histogram(n_values, bins=100)
	freq_t0, bins_t0 = np.histogram(t0_values, bins=100)

	best_f = bins_f[np.argmax(freq_f)]
	best_g = bins_g[np.argmax(freq_g)]
	best_h = bins_h[np.argmax(freq_h)]
	best_n = bins_n[np.argmax(freq_n)]
	best_t0 = bins_t0[np.argmax(freq_t0)]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(likelihood(pixel_data, my_model(time, best_f, best_g, best_h, best_n, best_t0))) + " " + str(best_f) + " " + str(best_g) + " " + str(best_h) + " " + str(best_n) + " " + str(best_t0) +  "\n")
