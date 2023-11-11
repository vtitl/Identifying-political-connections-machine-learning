# Political connections
rm(list=ls())

# Change path to dataset & load libraries
pathData="C:/Dropbox/research/papers/identifyin-pol-conn/"
setwd(pathVita)

library(haven)
library(SuperLearner)
library(varhandle)  
library(caret)
library(dplyr)
library(stargazer)


# Set a seed for reproducibility in this random sampling.
set.seed(1)

###############################################################################################
########################## 1- DATA PREPARATION ################################################
#year = "2011"
year = "2018"

data<-read.csv(file=paste("data_",year,".csv", sep=""))

colnames(data)

### Here one choose the type of connection to be predicted
type = "all"
#type = "donors"
#type = "ceo_donors"
#type = "connections"

data$connection = 0

data$donor[is.na(data$donor)] = 0
data$donor_po[is.na(data$donor_po)] = 0
data$personal_connection[is.na(data$personal_connection)] = 0

#### ALL
if (type=="all"){
  data$connection[data$donor==1] = 1
  data$connection[data$donor_po==1] = 1
  data$connection[data$personal_connection==1] = 1
}

#### ONLY DONORS 
if (type=="donors"){
  data$connection[data$donor==1] = 1
}
#### ONLY CEO DONORS
if (type=="ceo_donors"){
  data$connection[data$donor_po==1] = 1
}
#### ONLY PER. CONNECTIONS
if (type=="connections"){
  data$connection[data$personal_connection==1] = 1
}

table(data$connection)

data$prague = 0
data$prague[grepl("Praha",data$city,ignore.case = TRUE)] = 1

# Convert strings to factors
data$mainactivity <- as.factor(data$mainactivity)

data$missingIndicator = as.numeric(rowSums(is.na(data))>0)

data[is.na(data)] <- 0
data$connection<- factor(data$connection)


stargazer(data, type = "text", title="Descriptive statistics", digits=1, out="tableSummaryStats.txt")
# TABLE 1

########################## 3- MODELS setting #################################################
# List of models we will try
models <- c("SL.glm", "SL.xgboost", "SL.randomForest", "SL.glmnet", "SL.glmnet")
# RF, ridge regression (SL.glmnet with alpha=0), lasso (SL.glmnet with alpha=1), Support Vector Machines (SVM), 

# list to store data
results_cm <- list(0)
colnames(results) <- models

###############################################################################################
########################## 5- ESTIMATION by size #############################################
# We run for different training data sizes
sizes_steps_1 <- seq(100,1000,100)
sizes_steps_2 <- seq(2000,10000,1000)
sizes = c(sizes_steps_1,sizes_steps_2)
results_cm_by_size <- list(0)

# We will use 10 folds
cvFolds = 10

for(i in (1:length(sizes))) {
  size <- sizes[i]

  # balancing the dataset (SMOTE)
  data_pos <- subset(data,connection==1)
  data_neg <- subset(data,connection==0)
  
  smoted_data <- data_neg[sample(length(data_pos[,1]),replace = FALSE),]
  
  table(smoted_data$connection)
  smoted_data <- rbind(smoted_data,data_pos)
  table(smoted_data$connection)
  ###################
  
  y = smoted_data$connection
  x = subset(smoted_data, select = -c(connection))
  
  # less than 10,000 connected firms, we need to adjust the sampling to sampling with replacement
  if ((size>=4000 & type == "ceo_donors") | (size>=9000 & type == "connections")){
    sampling = sample(length(y), size,replace=TRUE)
  }else{
    sampling = sample(length(y), size)
  }
  y = y[sampling]
  x = x[sampling,]
  
  # to store results temporary 
  results <- matrix(data=NA, nrow = 1, ncol= length(models))
  results <- data.frame(results)
  results_cm <- list(0)
  colnames(results) <- models
  
  ridgeDone = 0
  for (j in (1:length(models))) {
    model = models[j]
    
    if (model=="SL.glmnet" & ridgeDone==0){
        learners    = create.Learner("SL.glmnet", params = list(alpha = 0))
        sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                          family = binomial(), SL.library = learners$names,
                          control = list(saveFitLibrary = TRUE),
                          cvControl = list(V = cvFolds, shuffle = FALSE))
        ridgeDone=1
    }
    else if (model=="SL.nnet" | model=="SL.glm"){
      x$mainactivity = as.numeric(x$mainactivity)
      
      sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                        family = binomial(), SL.library = model,
                        control = list(saveFitLibrary = TRUE),
                        cvControl = list(V = cvFolds, shuffle = FALSE))
    }
    else{
      sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                        family = binomial(), SL.library = model,
                        control = list(saveFitLibrary = TRUE),
                        cvControl = list(V = cvFolds, shuffle = FALSE))
    }
    
    convPreds = ifelse(sl$library.predict>=0.5,1,0)
    trueValues = sl$Y
  
    cm = confusionMatrix(factor(convPreds),factor(trueValues))  

    results_cm_by_size[[(i-1)*length(models)+j]] = cm

  }
  
  print(sizes[i])
}

# saving results
saveFile = paste("Resultsestimations/results_CMs_by_size_accuracyBalanced_",year,"_",type,".Rdata", sep="")
save(results_cm_by_size,file=saveFile)
