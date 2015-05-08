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

file_name = 'likelihood_lineal.txt'

if os.path.isfile(file_name):
	os.remove(file_name)

with open(file_name, 'a') as myfile:
	myfile.write('x y likelihood a b\n')


#Modelo B(t) = a + bt
def my_model(x, a, b):
	model = [a + b*t for t in x]
	return model

def likelihood(y_obs, y_model):
	chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
	return -chi_squared 

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	a_walk = np.empty((0)) 
	b_walk = np.empty((0)) 
	l_walk = np.empty((0))

	a_walk = np.append(a_walk, np.random.random()*10) 
	b_walk = np.append(b_walk, np.random.random()*0.1)

	B_init = my_model(time, a_walk[0], b_walk[0])
	l_walk = np.append(l_walk, likelihood(pixel_data, B_init))

	n_iterations = 20000
	for i in range(n_iterations):
		a_prime = np.random.normal(a_walk[i], 1)
		b_prime = np.random.normal(b_walk[i], 0.1)

		B_init = my_model(time, a_walk[i], b_walk[i])
		B_prime = my_model(time, a_prime, b_prime)

		l_init = likelihood(pixel_data, B_init)
		l_prime = likelihood(pixel_data, B_prime)
		
		alpha = l_prime - l_init
		if(alpha >= 0.0):
			a_walk  = np.append(a_walk, a_prime)
			b_walk  = np.append(b_walk, b_prime)
			l_walk = np.append(l_walk, l_prime)
		else:
			beta = np.random.random()
			if(beta <= alpha):
				a_walk = np.append(a_walk, a_prime)
				b_walk = np.append(b_walk, b_prime)
				l_walk = np.append(l_walk, l_prime)
			else:
				a_walk = np.append(a_walk, a_walk[i])
				b_walk = np.append(b_walk, b_walk[i])
				l_walk = np.append(l_walk, l_init)

	max_likelihood = np.argmax(l_walk)
	best_a = a_walk[max_likelihood]
	best_b = b_walk[max_likelihood]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(l_walk[max_likelihood]) + " " + str(best_a) + " " + str(best_b) + "\n")
