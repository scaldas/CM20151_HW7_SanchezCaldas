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

file_name = 'likelihood_lineal.txt'

if os.path.isfile(file_name):
	os.remove(file_name)

with open(file_name, 'a') as myfile:
	myfile.write('x y likelihood a b\n')

#Modelo Lineal
def my_model(x, a, b):
	model = [a + b*t for t in x]
	return model

def lnprior(theta):
	a, b = theta
	if 0 < a < 100.0 and -1 < b < 1:
		return 0.0
	return -np.inf

def lnlike(theta, x, y):
	a, b = theta
	model = my_model(x, a, b)
	return -0.5*(np.sum((y-model)**2))

def lnprob(theta, x, y):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y)

def likelihood(y_obs, y_model):
    chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
    return -chi_squared

ndim = 2
nwalkers = 10
nsteps = 2000
pos = [[np.random.random()*10, np.random.random()*0.1] for i in range(nwalkers)]

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, pixel_data))
	sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state( ))
	samples_fc = sampler.flatchain
	
	a_values = []
	b_values = []

	i = 0

	for element in samples_fc:
		if i > 100:
			a_values.append(element[0])
			b_values.append(element[1])
		else:
			i = i + 1

	freq_a, bins_a = np.histogram(a_values, bins=100)
	freq_b, bins_b = np.histogram(b_values, bins=100)

	best_a = bins_a[np.argmax(freq_a)]
	best_b = bins_b[np.argmax(freq_b)]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(likelihood(pixel_data, my_model(time, best_a, best_b))) + " " + str(best_a) + " " + str(best_b) + "\n")
