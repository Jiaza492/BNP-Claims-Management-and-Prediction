#################################
setwd("~/Documents/KSMC/")
train = read.csv("train.csv",header = TRUE,na.strings = '')
test = read.csv("test.csv",header = TRUE, na.strings = '')
# test.csv need to be repaired.
train_1 = na.omit(train)
write.csv(train_1,"train_1.csv")

##################################
## Data imputation 
#summary(as.factor(train_1$target))
#summary(train)
# store summary into summary_list
sum.list = function(train){
  summary_list = list()
  for(i in 1:dim(train)[2]){
    sum = as.matrix(summary(train[,i]))
    summary_list = c(summary_list, list(sum = sum))
  }
  return(summary_list)
}

# here we first use Amelia to impute the missing data
library(Amelia)
dummy = c(4,23,25,31,32,39,48,53,57,63,67,72,73,75,76,80,92,108,111,113,114,126,130)# in test
dummy_var = colnames(train_1[,-1])[dummy]
imputs = amelia(train, idvars = c("ID","target",dummy_var), empri = 0.01 * nrow(train))
train = imputs$imputations$imp3
imputs_test = amelia(test, idvars = c("ID",dummy_var), empri = 0.01 * nrow(test))
test = imputs_test$imputations$imp1
rm(imputs)
rm(inputs_test)

# impute test set
repair = function(dumm_var, test){
  for(i in dumm_var){
    #sum = sum_list[i+1]
    na.ind = which(is.na(test[,i])==TRUE)
    if(class(test[,i]) == "factor"){
      print(i)
      l = levels(test[,i])
      prob = as.vector(table(test[,i]))/length(test[,i])
      print(prob)
      print(length(na.ind))
      print(table(test[,i]))
      #print(names)
      #col = test[,i]
      if(length(na.ind) > 0){
        for(j in 1:length(na.ind)){
          r = sample(length(l),1,prob = prob)
          #print("repair")
          test[na.ind[j],i] = l[r]
        }
      }
      #print(col)
      #test[,i] = col
    }
  }
  return(test)
}

#sum_list = sum.list(train)

# refill test set
test_1 = repair(dummy_var, test)
write.csv(test_1,"test_1.csv")

# refill train set
train_re = repair(sum_list, train[,-1])
train_rr = data.frame("ID" = train$ID,train_re)
write.csv(train_rr,"train_re.csv")

######################################
## Factorization
train_1 = read.csv("train_re.csv",header = TRUE)
train_1 = train_1[,-1]

test_1 = read.csv("test_1.csv",header = TRUE)
test_1 = test_1[,-1]

# factorize some numerical variables
test_1$v38 = as.factor(test_1$v38)
test_1$v62 = as.factor(test_1$v62)
test_1$v72 = as.factor(test_1$v72)
test_1$v129 = as.factor(test_1$v129)

# make test and train set onto same feature levels
flat.levels = function(train, test, dummy_var){
  for(i in 1:length(dummy_var)){
    train[,dummy_var[i]] = factor(train[,dummy_var[i]],levels=levels(as.factor(test[,dummy_var[i]])))
  }
  return(train)
}

dummy = c(4,23,25,31,32,39,48,53,57,63,67,72,73,75,76,80,92,108,111,113,114,126,130)# in test
dummy_var = colnames(train_1[,-1])[dummy]

train_1 = flat.levels(train_1, test_1, dummy_var)
train_1 = na.omit(train_1)

###############################################
# log loss function
MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- length(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(length(act))      
  return(ll);
}

###############################################
## Regularized logistic model

# ## Standarized
# v_names = colnames(train_1)
# numerical = colnames(train_1[,c(-1,-2)])[-(dummy-1)]
# stand = function(dt, numerical){
#   for(i in 1:length(numerical)){
#     var = sd(dt[,numerical[i]])
#     dt[, numerical[i]] = (dt[,numerical[i]]-mean(dt[,numerical[i]]))
#   }
#   return(dt)
# }
# train_1.sta = stand(train_1,numerical)

# transform categorical feature into dummy variables
library(caret)

dv = dummyVars(~.,data = train_1[,c(-1,-2, -24)])
train_log = data.frame(predict(dv,train_1[,c(-1,-2, -24)]))

library(glmnet)
library(doMC)
registerDoMC(cores=8)
# samples 
ind = c(1:nrow(train_log))
which.train = sample.int(nrow(train_log), 100000)
trains.ind = ind[which.train]
tests.ind = ind[-which.train]

# Cross Validation
fit_log = cv.glmnet(x = as.matrix(train_log[trains.ind,]), y = as.matrix(train_1[trains.ind,]$target), family = "binomial", alpha = 1, type.measure="class",parallel = TRUE)
plot(fit_log)

# Choose best model then fit model
model_log = glmnet(x = as.matrix(train_log[trains.ind,]), y = as.matrix(train_1[trains.ind,]$target), family = "binomial", alpha = 1, lambda = fit_log$lambda.min, standardize = FALSE)

# predictions
model_pre = predict(model_log, newx=as.matrix(train_log[tests.ind,]),type="response")
model_log.loss = MultiLogLoss(as.numeric(train_1[tests.ind,]$target), model_pre)

# result of selected variables
var.names =c("intercept",colnames(train_log))
var.select = var.names[which(coef(model_log,lambda = fit_log$lambda.min)!= 0)]
var_coef = data.frame("coef" = model_log$beta[which(abs(coef(model_log)) != 0)-1,])
write.csv(var_coef,"coef_log.csv")

# significant variables
var_sig = var.select[-1]

# store fit_log
saveRDS(fit_log, file = "fit_log", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)


###############################################
## KNN
library(class)
library(caret)
# sample a smaller size
ind = sample.int(nrow(train_log), 20000)
which.train = sample.int(20000, 10000)
trains.ind = ind[which.train]
tests.ind = ind[-which.train]

# fit knn model
ks = seq(from=10,to=500,by=20)
loss = c()
for(i in 1:length(ks)){
  knnTrain = knn3(x = train_log[trains.ind,var_sig],y = as.factor(train_1[trains.ind,]$target), k = ks[i], prob = TRUE)
  fit_knn = predict(knnTrain, newdata = train_log[tests.ind,var_sig], type="prob")
  loss = rbind(loss, MultiLogLoss(as.numeric(train_1[tests.ind,]$target), fit_knn[,2]))
  print(loss)
}
loss = cbind(ks,loss)
colnames(loss) = c("k","log_loss")
write.csv(loss,"knn_loss.csv")

###############################################
## RandomForest
# reduce feature levels
library(rminer)
OVER_LIMIT = c("v22","v56","v113","v125")
OVER_LIMIT = c(23,57,114,126) # in test_1

reduce_it = function(dat, f_to_r, level_limit){
  dt = dat$dt
  dt_train = dat$dt_train
  sum = table(dt[,f_to_r])
  sorted = sort(sum,decreasing = TRUE)
  rm_levels = rownames(sorted)[level_limit:length(sorted)]
  #print(rm_levels)
  #print(temp)
  temp = delevels(dt[,f_to_r], levels = rm_levels, label = "Others")
  temp_train = delevels(dt_train[,f_to_r], levels = rm_levels, label = "Others")
  #print(table(temp))
  #temp = factor(dt[,f_to_r], levels = new_levels)
  #print(temp)
  print(table(temp))
  print(table(temp_train))
  dt[,f_to_r] = temp
  dt_train[,f_to_r] = temp_train
  return(list(dt = dt, dt_train = dt_train))
}


# samples
ind = sample.int(nrow(train_log), 20000)
which.train = sample.int(20000, 10000)
trains.ind = ind[which.train]
tests.ind = ind[-which.train]

# if we decide to reduce feature
var.select.sta = c("v9","v14","v20","v21","v24","v28","v30","v31","v38","v39","v40","v44","v47","v50","v56","v59","v66","v78","v82","v89","v98","v100","v101","v102","v109","v110","v112","v117","v118","v119","v129","v131" )
dat = list(dt = test_1[,var.select.sta], dt_train = train_1[,var.select.sta])
dat = reduce_it(dat,"v56",20)
train_rf = dat$dt_train
test_rf = dat$dt

# if we decide to use dummy variables matrix
# var.select.sta = var.select[-1]
# library(caret)
# dv = dummyVars(~.,data = train_1[,c(-1,-2,-24)])
# train_dum = data.frame(predict(dv,train_1[,c(-1,-2,-24)]))
# train_rf = train_dum[,var.select.sta]

# stratification
trainSet = train_log[trains.ind,var_sig]
testSet = train_log[tests.ind,var_sig]
strata = round(as.vector(table(train_1$target[trains.ind]))/10)

write.csv(train_1[trains.ind,],"train_1.csv")

## The Model
library(randomForest)
library(ROCR)

n_tree = c(21, 41, 61, 81, 101,401,801)
plot_mat = c()
result_mat = c()

for(i in 1:length(n_tree)){
  fit = randomForest(x=trainSet, y=as.factor(train_1$target[trains.ind]),strata =as.factor(train_1$target[trains.ind]) ,sampsize = strata,importance = TRUE, ntree = n_tree[i], proximity=TRUE )
  overall_error=fit$err.rate[length(fit$err.rate[,1]),1]
  
  # MSE for train
  predictions=fit$votes[,2]
  pred=prediction(predictions,train_1$target[trains.ind])
  train_mse = MultiLogLoss(as.numeric(train_1$target[trains.ind]), predictions)
  
  #First calculate the AUC value
  perf_AUC=performance(pred,"auc")
  AUC=perf_AUC@y.values[[1]]
  
  #Then, plot the actual ROC curve
  perf_ROC=performance(pred,"tpr","fpr")
  plot_mat = cbind(plot_mat,perf_ROC)
  
  # prediction
  rf_pre = predict(fit, newdata = testSet,type = "vote",norm.votes = TRUE)
  rf_pre1 = predict(fit, newdata = testSet)
  pre_err = sum(abs(as.numeric(rf_pre1) - as.numeric(train_1$target[tests.ind])))/nrow(train_1[-trains.ind,])
  
  # MSE for test
  predictions=rf_pre[,2]
  pred=prediction(predictions,train_1$target[tests.ind])
  test_mse = MultiLogLoss(as.numeric(train_1$target[tests.ind]), predictions)
  
  
  # print errors'
  r = c(n_tree[i],overall_error, pre_err, train_mse, test_mse, AUC)
  print(r)
  result_mat = rbind(result_mat,r)
}

# store result rf
colnames(result_mat) = c("n_tree","train_OOB","test_OOB","train_LOSS","test_LOSS", "AUC")
write.csv(result_mat,"result_rf_dummy.csv")

# store ROC
saveRDS(plot_mat, file = "roc", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)

