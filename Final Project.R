# Lasso Linear

wlx = model.matrix(`W.L` ~ . - Win_Margin - Away - Home - Year, data=Final_Data) 
wly = Final_Data$`W.L`

wllasso = gamlr(wlx, wly, family = "binomial")
plot(wllasso)

AICc(wllasso)
plot(wllasso$lambda, AICc(wllasso))
plot(log(wllasso$lambda), AICc(wllasso))

wlbeta = coef(wllasso) 

log(wllasso$lambda[which.min(AICc(wllasso))])
sum(wlbeta!=0)

wlcvl = cv.gamlr(wlx, wly, nfold=10, family="binomial", verb=TRUE)

plot(wlcvl, bty="n")

wlb.min = coef(wlcvl, select="min")
log(wlcvl$lambda.min)
sum(wlb.min!=0)

wlb.1se = coef(wlcvl)
log(wlcvl$lambda.1se)
sum(wlb.1se!=0)

plot(wlcvl, bty="n", ylim=c(0, 10))
lines(log(wllasso$lambda),AICc(wllasso), col="green", lwd=2)
legend("top", fill=c("blue","green"), legend=c("CV","AICc"), bty="n")

library(glmnet)

best_lambda <- wlcvl$lambda.min

best_model <- glmnet(wlx, wly, lambda = best_lambda)
coef(best_model)


# Stepwise Selection

library(tidyverse)
library(mosaic)
library(foreach)
library(modelr)
library(rsample)


hockey_split = initial_split(Final_Data, prop = 0.8)
hockey_train = training(hockey_split)
hockey_test = testing(hockey_split)

lm_medium = lm(data = Final_Data, `W.L` ~ Point_Differential + Total_goal_differential + `Net.PP.` + `Net.PK.` + Shot_Differential)

lm0 = lm(`W.L` ~ 1, data=Final_Data)
lm_forward = step(lm0, direction='forward', scope=~(.))

lm_back = lm(data = Final_Data, Final_Data$W.L ~ (Point_Differential + Total_goal_differential + `Net.PP.` + `Net.PK.` + Shot_Differential +
                        `GF.GP` + `GA.GP` + `PIM` + `Net.Pen.60`)^2)


lm_step = step(lm_medium, scope=~(.)^2)

getCall(lm_step)
coef(lm_step)

rmse(lm_medium, hockey_test)
rmse(lm_back, hockey_test)
rmse(lm_forward, hockey_test)
rmse(lm_step, hockey_test)


# Linear and Probit Models



library(rms)
library(aod)
library(ggplot2)
library(tidyverse)
library(rsample)
library(caret)
library(modelr)
library(parallel)
library(foreach)

olsreg <- lm(`W.L` ~ . - Away - Home - Year - Win_Margin, data = Final_Data)
summary(olsreg)

probit <- glm(`W.L` ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`, family = binomial(link = "probit"), data = Final_Data)
summary(probit)

coef(olsreg)

ProbitScalar <- mean(dnorm(predict(probit, type = "link")))
ProbitScalar * coef(probit)

polsreg <- predict(olsreg)
summary(polsreg)

pprobit <- predict(probit, type = "response")
summary(pprobit)

matrix = table(true = Final_Data$W.L, pred = round(fitted(probit)))

sum(diag(matrix))/sum(matrix)

probit0 <- update(probit, formula = `W.L` ~ 1)
McFadden <- 1-as.vector(logLik(probit)/logLik(probit0))
McFadden


# Ordered Probit

library(MASS)

playoff.plr <- polr(as.factor(`Win_Margin`) ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`, method = "probit", data = Final_Data)
playoff.plr 
summary(playoff.plr)

summary(update(playoff.plr, method = "probit"))

summary(update(playoff.plr, method = "cloglog"))

predict(playoff.plr, Final_Data, type = "p")
addterm(playoff.plr, ~.^2, test = "Chisq")
playoff.plr2 <- stepAIC(playoff.plr, ~.^2)
playoff.plr2$anova
anova(playoff.plr, playoff.plr2)

playoff.plr <- update(playoff.plr, Hess=TRUE)
pr <- profile(playoff.plr)
confint(pr)
plot(pr)
pairs(pr)



# Testing Accuracy


test = predict(playoff.plr)
(err.table = table(Final_Data$Win_Margin, test))
1 - sum(diag(err.table)) / sum(err.table)


phat_test_data = predict(logit, Final_Data)
yhat_test_data = ifelse(phat_test_data > 0.5, 1, 0)
confusion_out = table(y = Final_Data$`W.L`, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)


# Logit

logit <- glm(`W.L` ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`, family = binomial(link = "logit"), data = Final_Data)
summary(logit)

matrixl = table(true = Final_Data$W.L, pred = round(fitted(logit)))

sum(diag(matrixl))/sum(matrixl)


# K-Fold

library(tidyverse)
library(rsample)
library(caret)
library(modelr)
library(parallel)
library(foreach)

hockey_split = initial_split(Final_Data, prop = 0.8)
hockey_train = training(hockey_split)
hockey_test = testing(hockey_split)

K_folds = 10

Final_Data = Final_Data %>%
  mutate(fold_id = rep(1:K_folds, length=nrow(Final_Data)) %>% sample)

head(Final_Data)

rmse_cv = foreach(fold = 1:K_folds, .combine='c') %do% {
  logit = glm(`W.L` ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`,
                  family = binomial(link = "logit"), data=filter(Final_Data, fold_id != fold))
  modelr::rmse(logit, data=filter(Final_Data, fold_id == fold))
}

rmse_cv
mean(rmse_cv) 
sd(rmse_cv)/sqrt(K_folds)




probit_step <- glm(`W.L` ~ `FOW.` + `Net.PK.` + `Win..2.Goal.Game` + `Pen.Drawn.60` + `Win..3.Goal.Game` + `GA.in.P2` + `Win..Lead.2P` + `Wins.Lead.1P`, family = binomial(link = "probit"), data = Final_Data)
summary(probit_step)

logit_step <- glm(`W.L` ~ `FOW.` + `Net.PK.` + `Win..2.Goal.Game` + `Pen.Drawn.60` + `Win..3.Goal.Game` + `GA.in.P2` + `Win..Lead.2P` + `Wins.Lead.1P`, family = binomial(link = "logit"), data = Final_Data)
summary(logit_step)

playoff.plr.step <- polr(as.factor(`Win_Margin`) ~ `FOW.` + `Net.PK.` + `Win..2.Goal.Game` + `Pen.Drawn.60` + `Win..3.Goal.Game` + `GA.in.P2` + `Win..Lead.2P` + `Wins.Lead.1P`, method = "probit", data = Final_Data)
playoff.plr.step 
summary(playoff.plr.step)

summary(update(playoff.plr.step, method = "probit"))

test = predict(playoff.plr.step)
(err.table = table(Final_Data$Win_Margin, test))
sum(diag(err.table)) / sum(err.table)


phat_test_data = predict(logit_step, Final_Data)
yhat_test_data = ifelse(phat_test_data > 0.5, 1, 0)
confusion_out = table(y = Final_Data$`W.L`, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)