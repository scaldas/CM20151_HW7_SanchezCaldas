import pyfits
import numpy as np
import matplotlib.pyplot as plt

f_lineal = open('likelihood_lineal.txt','r')
f_gaussiano = open('likelihood_gaussiana.txt','r')
f_paso = open('likelihood_paso.txt','r')

raw_lineal = f_lineal.readlines()
raw_gaussiana = f_gaussiano.readlines()
raw_paso = f_paso.readlines()

raw_lineal.pop(0)
raw_gaussiana.pop(0)
raw_paso.pop(0)

f_lineal.close()
f_gaussiano.close()
f_paso.close()

#Se cargan los intervalos de tiempo del archivo intervals.csv
with open('intervals.csv','r') as f:
	time = f.readlines()
time.pop(0)
time = [float(i) for i in time]

fits_data = pyfits.open('hmi.m_45s.magnetogram.subregion_x1y1.fits')
fits_data = fits_data[0].data

#Modelo Lineal
def lineal_model(x, a, b):
	model = [a + b*t for t in x]
	return model

def gaussian_model(x, c, d, sigma, miu, kappa):
    helper = np.sqrt(2*np.pi)
    model = [c + d*t + (kappa/(sigma*helper))*np.exp((-1.0/2.0)*((t-miu)/sigma)**2) for t in x]
    return model

def step_model(x, f, g, h, n, t0):
	model = [f + g*t + h*(1 + (2.0/np.pi)*np.arctan(n*(t-t0))) for t in x]
	return model

print(len(raw_lineal))
for i in range(0, len(raw_lineal)):
	likelihoods = []

	info_lineal = raw_lineal[i].rsplit(" ")
	likelihood_lineal = float(info_lineal[2])
	likelihoods.append(likelihood_lineal)

	info_gaussiana = raw_gaussiana[i].rsplit(" ")
	likelihood_gaussiana = float(info_gaussiana[2])
	likelihoods.append(likelihood_gaussiana)

	info_paso = raw_paso[i].rsplit(" ")
	likelihood_paso = float(info_paso[2])
	likelihoods.append(likelihood_paso)

	max_likelihood = max(likelihoods)

	if max_likelihood == likelihood_lineal:
		B_fit = lineal_model(time, float(info_lineal[3]), float(info_lineal[4]))
		pixel_data = fits_data[:,info_lineal[0],info_lineal[1]]
		B_fit = lineal_model(time, float(info_lineal[3]), float(info_lineal[4]))
		real, = plt.plot(time, pixel_data, 'go', markersize=3)
		fit, = plt.plot(time, B_fit, 'k')
		plt.title('Mejor Fit para el Pixel (' + info_lineal[0] + ',' + info_lineal[1] + '): Lineal\n Parametros: a = ' + "{0:.2f}".format(float(info_lineal[3])) + ', b = ' + "{0:.2f}".format(float(info_lineal[4])))
		plt.ylabel('B (Unidades Arbitrarias)')
		plt.xlabel('t (min)')
		plt.legend([real, fit], ['Observado','Fit'], loc=4)
		plt.savefig('./graficas/Pixel' + str(info_lineal[0]) + '-' + str(info_lineal[1]) + '.png')
		plt.clf()
	else:
		if max_likelihood == likelihood_gaussiana:
			B_fit = lineal_model(time, float(info_gaussiana[3]), float(info_gaussiana[4]))
			pixel_data = fits_data[:,info_gaussiana[0],info_gaussiana[1]]
			B_fit = gaussian_model(time, float(info_gaussiana[3]), float(info_gaussiana[4]), float(info_gaussiana[5]), float(info_gaussiana[6]), float(info_gaussiana[7]))
			real, = plt.plot(time, pixel_data, 'go', markersize=3)
			fit, = plt.plot(time, B_fit, 'k')
			plt.title('Mejor Fit para el Pixel (' + info_gaussiana[0] + ',' + info_gaussiana[1] + '): Gaussiano\n Parametros: c = ' + "{0:.2f}".format(float(info_gaussiana[3])) + ', d = ' + "{0:.2f}".format(float(info_gaussiana[4])) + ', sigma = ' + "{0:.2f}".format(float(info_gaussiana[5])) + '\nmiu = ' + "{0:.2f}".format(float(info_gaussiana[6])) + ', kappa = ' + "{0:.2f}".format(float(info_gaussiana[7]))) 
			plt.ylabel('B (Unidades Arbitrarias)')
			plt.xlabel('t (min)')
			plt.legend([real, fit], ['Observado','Fit'], loc=4)
			plt.savefig('./graficas/Pixel' + str(info_gaussiana[0]) + '-' + str(info_gaussiana[1]) + '.png', bbox_inches='tight')
			plt.clf()	
		else:
			pixel_data = fits_data[:,info_paso[0],info_paso[1]]
			B_fit = step_model(time, float(info_paso[3]), float(info_paso[4]), float(info_paso[5]), float(info_paso[6]), float(info_paso[7]))
			real, = plt.plot(time, pixel_data, 'go', markersize=3)
			fit, = plt.plot(time, B_fit, 'k')
			plt.title('Mejor Fit para el Pixel (' + info_paso[0] + ',' + info_paso[1] + '): Paso\n Parametros: f = ' + "{0:.2f}".format(float(info_paso[3])) + ', g = ' + "{0:.2f}".format(float(info_paso[4])) + ', h = ' + "{0:.2f}".format(float(info_paso[5])) + '\nn = ' + "{0:.2f}".format(float(info_paso[6])) + ', t0 = ' + "{0:.2f}".format(float(info_paso[7]))) 
			plt.ylabel('B (Unidades Arbitrarias)')
			plt.xlabel('t (min)')
			plt.legend([real, fit], ['Observado','Fit'], loc=4)
			plt.savefig('./graficas/Pixel' + str(info_paso[0]) + '-' + str(info_paso[1]) + '.png', bbox_inches='tight')
			plt.clf()