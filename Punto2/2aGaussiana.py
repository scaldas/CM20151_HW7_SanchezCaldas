import numpy as np 
import pyfits, os

#Se leen los pixeles que se determinaron relevantes
with open('pixels.txt','r') as f:
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

def likelihood(y_obs, y_model):
	chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
	return -chi_squared 

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	c_walk = np.empty((0)) 
	d_walk = np.empty((0)) 
	sigma_walk = np.empty((0)) 
	miu_walk = np.empty((0)) 
	kappa_walk = np.empty((0)) 
	l_walk = np.empty((0))

	c_walk = np.append(c_walk, np.random.random()) 
	d_walk = np.append(d_walk, np.random.random())
	sigma_walk = np.append(sigma_walk, np.random.random())
	miu_walk = np.append(miu_walk, 250) 
	kappa_walk = np.append(kappa_walk, np.random.random()) 

	B_init = my_model(time, c_walk[0], d_walk[0], sigma_walk[0], miu_walk[0], kappa_walk[0])
	l_walk = np.append(l_walk, likelihood(pixel_data, B_init))

	n_iterations = 20000
	for i in range(n_iterations):
		c_prime = np.random.normal(c_walk[i], 1) 
		d_prime = np.random.normal(d_walk[i], 0.1)
		sigma_prime = np.random.normal(sigma_walk[i], 0.1)
		miu_prime = np.random.normal(miu_walk[i], 0.1)
		kappa_prime = np.random.normal(kappa_walk[i], 1)

		B_init = my_model(time, c_walk[i], d_walk[i], sigma_walk[i], miu_walk[i], kappa_walk[i])
		B_prime = my_model(time, c_prime, d_prime, sigma_prime, miu_prime, kappa_prime)
        
		l_init = likelihood(pixel_data, B_init)
		l_prime = likelihood(pixel_data, B_prime)
    
		alpha = l_prime - l_init
		if(alpha >= 0.0):
			c_walk  = np.append(c_walk, c_prime)
			d_walk  = np.append(d_walk, d_prime)
			sigma_walk  = np.append(sigma_walk, sigma_prime)
			miu_walk  = np.append(miu_walk, miu_prime)
			kappa_walk = np.append(kappa_walk, kappa_prime)
			l_walk = np.append(l_walk, l_prime)
		else:
			beta = np.random.random()
			if(beta <= alpha):
				c_walk = np.append(c_walk, c_prime)
				d_walk = np.append(d_walk, d_prime)
				sigma_walk = np.append(sigma_walk, sigma_prime)
				miu_walk = np.append(miu_walk, miu_prime)
				kappa_walk = np.append(kappa_walk, kappa_prime)
				l_walk = np.append(l_walk, l_prime)
			else:
				c_walk = np.append(c_walk, c_walk[i])
				d_walk = np.append(d_walk, d_walk[i])
				sigma_walk = np.append(sigma_walk, sigma_walk[i])
				miu_walk = np.append(miu_walk, miu_walk[i])
				kappa_walk = np.append(kappa_walk, kappa_walk[i])
				l_walk = np.append(l_walk, l_init)
	
	max_likelihood = np.argmax(l_walk)
	best_c = c_walk[max_likelihood]
	best_d = d_walk[max_likelihood]
	best_sigma = sigma_walk[max_likelihood]
	best_miu = miu_walk[max_likelihood]
	best_kappa = kappa_walk[max_likelihood]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(l_walk[max_likelihood]) + " " + str(best_c) + " " + str(best_d) + " " + str(best_sigma) + " " + str(best_miu) + " " + str(best_kappa) +  "\n")
