# PCA
knitr::opts_chunk$set(echo = TRUE)
library(knitr)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(lattice)
library(grid)
library(gridExtra)
library(mosaic)
library(quantmod)
library(foreach)

head(Half_Data_Sheet)
kable(sort(colnames(Half_Data_Sheet)))


H = Half_Data_Sheet[,c(6:66)]

H = scale(H, center=TRUE, scale=FALSE)

plot(H)

v_random = rnorm(2)
v_random = v_random/sqrt(sum(v_random^2))

plot(H, pch=19, col=rgb(0.3,0.3,0.3,0.3))
segments(0, 0, v_random[1], v_random[2], col='red', lwd=4)

slope = v_random[2]/v_random[1]
abline(0, slope)

v_random = rnorm(2)
v_random = v_random/sqrt(sum(v_random^2))

par(mfrow=c(1,2))
plot(H, pch=19, col=rgb(0.3,0.3,0.3,0.3),
     xlim=c(-2.5,2.5), ylim=c(-2.5,2.5))
slope = v_random[2]/v_random[1]
abline(0, slope)

alpha = H %>% v_random
H_hat = alpha %>% v_random
points(H_hat, col='blue', pch=4)
segments(0, 0, v_random[1], v_random[2], col='red', lwd=4)   !!!
  
  hist(alpha, 25, xlim=c(-3,3), main=round(var(alpha), 2))

pc_H = prcomp(H, rank=1)

pc_H$rotation
v_random

summary(pc_H)

pc_H$x