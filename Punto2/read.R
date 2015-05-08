#install.packages("lubridate")
#install.packages("BBmisc")
#install.packages("stringi")
library(lubridate)
library(BBmisc)
library(stringi)

#Cambiar la ruta absoluta a aquella donde se tenga el repositorio
dat <- read.csv('/Users/caldasrivera/Dropbox/UniAndes/Semestres Academicos/Septimo Semestre/Metodos Computacionales/Tareas/Tarea7/CM20151_HW7_SanchezCaldas/Punto2/times.csv',header=F)
dat$duration <- dat$V2
dat <- dropNamed(dat,"V1")
dat <- dropNamed(dat,"V2")
dat$duration <-stri_sub(dat$duration,1,19)
dat$durationdo <- parse_date_time(dat$duration, "%Y.%m.%d_%H:%M:%S"); 
dat <- dropNamed(dat,"duration")
dat$durationdo <- dat$durationdo - head(dat$durationdo,1)
dat$durationdo <- as.numeric(dat$durationdo)
#Se imprimen minutos
dat$durationdo <- dat$durationdo/60
#Cambiar la ruta absoluta a aquella donde se tenga el repositorio
write.csv(dat, "/Users/caldasrivera/Dropbox/UniAndes/Semestres Academicos/Septimo Semestre/Metodos Computacionales/Tareas/Tarea7/CM20151_HW7_SanchezCaldas/Punto2/intervals.csv", row.names=FALSE)
