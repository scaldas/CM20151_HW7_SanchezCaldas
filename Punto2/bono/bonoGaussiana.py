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

file_name = 'likelihood_gaussiana.txt'

if os.path.isfile(file_name):
	os.remove(file_name)

with open(file_name, 'a') as myfile:
	myfile.write('x y likelihood c d sigma miu kappa\n')

#Modelo Gaussiano
def my_model(x, c, d, sigma, miu, kappa):
	helper = np.sqrt(2*np.pi)
	model = [c + d*t + (kappa/(sigma*helper))*np.exp((-1.0/2.0)*((t-miu)/sigma)**2) for t in x]
	return model

def lnprior(theta):
	c, d, sigma, miu, kappa = theta
	if -100.0 < c < 100.0 and -1 < d < 1 and -5 < sigma < 5 and 200 < miu < 300 and -100 < kappa < 100:
		return 0.0
	return -np.inf

def lnlike(theta, x, y):
	c, d, sigma, miu, kappa = theta
	model = my_model(x, c, d, sigma, miu, kappa)
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
pos = [[np.random.random(), np.random.random(), np.random.random(), 250, np.random.random()] for i in range(nwalkers)]

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, pixel_data))
	sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state( ))
	samples_fc = sampler.flatchain
	
	c_values = []
	d_values = []
	sigma_values = []
	miu_values = []
	kappa_values = []

	i = 0

	for element in samples_fc:
		if i > 100:
			c_values.append(element[0])
			d_values.append(element[1])
			sigma_values.append(element[2])
			miu_values.append(element[3])
			kappa_values.append(element[4])
		else:
			i = i + 1

	freq_c, bins_c = np.histogram(c_values, bins=100)
	freq_d, bins_d = np.histogram(d_values, bins=100)
	freq_sigma, bins_sigma = np.histogram(sigma_values, bins=100)
	freq_miu, bins_miu = np.histogram(miu_values, bins=100)
	freq_kappa, bins_kappa = np.histogram(kappa_values, bins=100)

	best_c = bins_c[np.argmax(freq_c)]
	best_d = bins_d[np.argmax(freq_d)]
	best_sigma = bins_sigma[np.argmax(freq_sigma)]
	best_miu = bins_miu[np.argmax(freq_miu)]
	best_kappa = bins_kappa[np.argmax(freq_kappa)]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(likelihood(pixel_data, my_model(time, best_c, best_d, best_sigma, best_miu, best_kappa))) + " " + str(best_c) + " " + str(best_d) + " " + str(best_sigma) + " " + str(best_miu) + " " + str(best_kappa) +  "\n")
