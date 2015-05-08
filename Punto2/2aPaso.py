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

file_name = 'likelihood_paso.txt'

if os.path.isfile(file_name):
	os.remove(file_name)

with open(file_name, 'a') as myfile:
	myfile.write('x y likelihood f g h n t0\n')


#Modelo Paso
def my_model(x, f, g, h, n, t0):
	model = [f + g*t + h*(1 + (2.0/np.pi)*np.arctan(n*(t-t0))) for t in x]
	return model

def likelihood(y_obs, y_model):
	chi_squared = (1.0/2.0)*sum((y_obs-y_model)**2)
	return -chi_squared 

for j in range(0,len(pixels_x)):
	pixel_data = fits_data[:,pixels_x[j],pixels_y[j]]
	f_walk = np.empty((0)) 
	g_walk = np.empty((0)) 
	h_walk = np.empty((0)) 
	n_walk = np.empty((0)) 
	t0_walk = np.empty((0)) 
	l_walk = np.empty((0))

	f_walk = np.append(f_walk, np.random.random()*10) 
	g_walk = np.append(g_walk, np.random.random()*0.1)
	h_walk = np.append(h_walk, np.random.random()*100)
	n_walk = np.append(n_walk, np.random.random()*0.5) 
	t0_walk = np.append(t0_walk, ((350.0-80.0)/2.0)) 

	B_init = my_model(time, f_walk[0], g_walk[0], h_walk[0], n_walk[0], t0_walk[0])
	l_walk = np.append(l_walk, likelihood(pixel_data, B_init))

	n_iterations = 20000 #Numero de iteraciones
	for i in range(n_iterations):
		f_prime = np.random.normal(f_walk[i], 1) 
		g_prime = np.random.normal(g_walk[i], 0.01)
		h_prime = np.random.normal(h_walk[i], 10)
		n_prime = np.random.normal(n_walk[i], 0.1)
		t0_prime = np.random.normal(t0_walk[i], 10)

		B_init = my_model(time, f_walk[i], g_walk[i], h_walk[i], n_walk[i], t0_walk[i])
		B_prime = my_model(time, f_prime, g_prime, h_prime, n_prime, t0_prime)

		l_init = likelihood(pixel_data, B_init)
		l_prime = likelihood(pixel_data, B_prime)
    
		alpha = l_prime - l_init
		if(alpha >= 0.0):
			f_walk  = np.append(f_walk, f_prime)
			g_walk  = np.append(g_walk, g_prime)
			h_walk  = np.append(h_walk, h_prime)
			n_walk  = np.append(n_walk, n_prime)
			t0_walk  = np.append(t0_walk, t0_prime)
			l_walk = np.append(l_walk, l_prime)
		else:
			beta = np.random.random()
			if(beta <= alpha):
				f_walk = np.append(f_walk, f_prime)
				g_walk = np.append(g_walk, g_prime)
				h_walk = np.append(h_walk, h_prime)
				n_walk = np.append(n_walk, n_prime)
				t0_walk = np.append(t0_walk, t0_prime)
				l_walk = np.append(l_walk, l_prime)
			else:
				f_walk = np.append(f_walk, f_walk[i])
				g_walk = np.append(g_walk, g_walk[i])
				h_walk = np.append(h_walk, h_walk[i])
				n_walk = np.append(n_walk, n_walk[i])
				t0_walk = np.append(t0_walk, t0_walk[i])
				l_walk = np.append(l_walk, l_init)
	
	max_likelihood = np.argmax(l_walk)
	best_f = f_walk[max_likelihood]
	best_g = g_walk[max_likelihood]
	best_h = h_walk[max_likelihood]
	best_n = n_walk[max_likelihood]
	best_t0 = t0_walk[max_likelihood]

	with open(file_name, 'a') as myfile:
		myfile.write(str(pixels_x[j]) + " " + str(pixels_y[j]) + " " + str(l_walk[max_likelihood]) + " " + str(best_f) + " " + str(best_g) + " " + str(best_h) + " " + str(best_n) + " " + str(best_t0) +  "\n")
