# K-folds probit, logit, stepwise

Final_Data_probit = Final_Data

probit_step <- glm(`W.L` ~ `FOW.` + `Net.PK.` + `Win..2.Goal.Game` + `Pen.Drawn.60` + `Win..3.Goal.Game` + `GA.in.P2` + `Win..Lead.2P` + `Wins.Lead.1P`, family = binomial(link = "probit"), data = Final_Data)
summary(probit_step)

probit_split = initial_split(Final_Data, prop = 0.8)
probit_train = training(probit_split)
probit_test = testing(probit_split)

num_folds = 10

control = trainControl(method = "repeatedcv", number = num_folds, repeats = 100)
probit_cv = train(probit_step, data = probit_train, method = "glm", family = binomial(link = "probit"), trControl = control)
probit_cv_pred = predict(probit_cv, probit_test)

yhat_test_data = ifelse(probit_cv_pred > 0.5, 1, 0)
confusion_out = table(y = probit_test$W.L, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)


logit_step <- glm(`W.L` ~ `FOW.` + `Net.PK.` + `Win..2.Goal.Game` + `Pen.Drawn.60` + `Win..3.Goal.Game` + `GA.in.P2` + `Win..Lead.2P` + `Wins.Lead.1P`, family = binomial(link = "logit"), data = Final_Data)
summary(logit_step)


logit_split = initial_split(Final_Data, prop = 0.8)
logit_train = training(logit_split)
logit_test = testing(logit_split)

num_folds = 10

control = trainControl(method = "repeatedcv", number = num_folds, repeats = 100)
logit_cv = train(logit_step, data = logit_train, method = "glm", family = binomial(link = "logit"), trControl = control)
logit_cv_pred = predict(logit_cv, logit_test)

yhat_test_data = ifelse(logit_cv_predict > 0.5, 1, 0)
confusion_out = table(y = logit_test$W.L, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)

# K-folds probit, logit, lasso

Final_Data_probit = Final_Data

probit_lasso <- glm(`W.L` ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`, family = binomial(link = "probit"), data = Final_Data)
summary(probit_lasso)

probit_split = initial_split(Final_Data, prop = 0.8)
probit_train = training(probit_split)
probit_test = testing(probit_split)

num_folds = 10

control = trainControl(method = "repeatedcv", number = num_folds, repeats = 100)
probit_cv = train(probit_lasso, data = probit_train, method = "glm", family = binomial(link = "probit"), trControl = control)
probit_cv_pred = predict(probit_cv, probit_test)

yhat_test_data = ifelse(probit_cv_pred > 0.5, 1, 0)
confusion_out = table(y = probit_test$W.L, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)


logit_lasso <- glm(`W.L` ~ `RW` + Total_goal_differential + `Net.PK.` + Shot_Differential + `GA.in.P2` + `W..SF` + `Win..3.Goal.Game` + `Win..Lead.2P`, family = binomial(link = "logit"), data = Final_Data)
summary(logit_step)


logit_split = initial_split(Final_Data, prop = 0.8)
logit_train = training(logit_split)
logit_test = testing(logit_split)

num_folds = 10

control = trainControl(method = "repeatedcv", number = num_folds, repeats = 100)
logit_cv = train(logit_lasso, data = logit_train, method = "glm", family = binomial(link = "logit"), trControl = control)
logit_cv_pred = predict(logit_cv, logit_test)

yhat_test_data = ifelse(logit_cv_predict > 0.5, 1, 0)
confusion_out = table(y = logit_test$W.L, yhat = yhat_test_data)
confusion_out
sum(diag(confusion_out))/sum(confusion_out)