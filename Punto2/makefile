all: ./graficas/*.png

./graficas/*.png: likelihood_lineal.txt likelihood_gaussiana.txt likelihood_paso.txt
	python 2b.py

likelihood_lineal.txt: 2aLineal.py intervals.csv
	python 2aLineal.py

likelihood_gaussiana.txt: 2aGaussiana.py intervals.csv
	python 2aGaussiana.py
	
likelihood_paso.txt: 2aPaso.py intervals.csv
	python 2aPaso.py

clean: 
	rm ./graficas/*.png

