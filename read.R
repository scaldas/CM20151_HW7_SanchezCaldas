#install.packages("lubridate")
#install.packages("BBmisc")
#install.packages("stringi")
library(lubridate)
library(BBmisc)
library(stringi)

dat <- read.csv('~/Desktop/HW7/P2/times.csv',header=F)
dat$duration <- dat$V2
dat <- dropNamed(dat,"V1")
dat <- dropNamed(dat,"V2")
dat$duration <-stri_sub(dat$duration,1,19)
dat$durationdo <- parse_date_time(dat$duration, "%Y.%m.%d_%H:%M:%S"); 
dat <- dropNamed(dat,"duration")
dat$durationdo <- dat$durationdo - head(dat$durationdo,1)
dat$durationdo <- as.numeric(dat$durationdo)
dat$durationdo <- dat$durationdo/60
write.csv(dat, "~/Desktop/HW7/P2/dat.csv", row.names=FALSE, header=F)
