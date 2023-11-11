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
library(rpart.plot)
library(xgboost)
library(randomForest)
library("Ckmeans.1d.dp")


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


################################################
### Importance and partial dep. etc.
size = 1000

data_pos <- subset(data,connection==1)
data_neg <- subset(data,connection==0)

smoted_data <- data_neg[sample(length(data_pos[,1]),replace = FALSE),]

table(smoted_data$connection)
smoted_data <- rbind(smoted_data,data_pos)
table(smoted_data$connection)

###################

y = smoted_data$connection
x = subset(smoted_data, select = -c(connection))


xVector = matrix(unlist(x), ncol = length(x))
colnames(xVector) = colnames(x)

colnames(xVector)[colnames(xVector) == "age"] <- "Age"
colnames(xVector)[colnames(xVector) == "sum_procurement_contracts"] <- "Value of public procurement"
colnames(xVector)[colnames(xVector) == "oprevtheurlastavailyr"] <- "Operating Revenue"
colnames(xVector)[colnames(xVector) == "lastyear"] <- "Last year of submitted reports"
colnames(xVector)[colnames(xVector) == "shareholdersfundstheurlastavaily"] <- "Shareholders’ funds"
colnames(xVector)[colnames(xVector) == "nacerev2mainsection"] <- "Main activity"
colnames(xVector)[colnames(xVector) == "numberofcurrentdirectorsmanagers"] <- "Number of director managers"
colnames(xVector)[colnames(xVector) == "operatingplebittheurlastavailyr"] <- "Operat. profit and loss"
colnames(xVector)[colnames(xVector) == "prague"] <- "Based in Prague"
colnames(xVector)[colnames(xVector) == "nacecode"] <- "Detailed sector"
colnames(xVector)[colnames(xVector) == "returnoncapitalemployedlastavail"] <- "Return on capital"
colnames(xVector)[colnames(xVector) == "financialexpensestheurlastavaily"] <- "Financial expenses"
colnames(xVector)[colnames(xVector) == "solvencyratiolastavailyr"] <- "Solvency ratio"
colnames(xVector)[colnames(xVector) == "numberofemployeeslastavailyr"] <- "Number of employees"
colnames(xVector)[colnames(xVector) == "profitmarginlastavailyr"] <- "Profit margin"
colnames(xVector)[colnames(xVector) == "cashflowtheurlastavailyr"] <- "Cash flow"
colnames(xVector)[colnames(xVector) == "totalassetsmileurlastavailyr"] <- "Total assets"
colnames(xVector)[colnames(xVector) == "noofrecordedshareholders"] <- "Number of recorded shareholders"
colnames(xVector)[colnames(xVector) == "plbeforetaxtheurlastavailyr"] <- "Profit before tax"
colnames(xVector)[colnames(xVector) == "plforperiodmileurlastavailyr"] <- "Profit and loss"
colnames(xVector)[colnames(xVector) == "sum_eu_subsidies"] <- "Value of EU subsidies"
colnames(xVector)[colnames(xVector) == "noofrecordedsubsidiaries"] <- "Number of subsidiaries"

xgb_train = xgb.DMatrix(data = xVector, label = matrix(unlist(as.numeric(unfactor(y))), ncol = 1))
model = xgb.train(data = xgb_train, max.depth = 3, nrounds = 100)
importance_matrix = xgb.importance(colnames(xgb_train), model = model)

### feature importance all variables
(gg <- xgb.ggplot.importance(importance_matrix,n_clusters=1))
gg+ theme(legend.position = "none")+ theme(text = element_text(size = 18))   
ggsave("graphs/featureImportanceAllVaribles.pdf")
# export the graph of feature importance 

### Logit
x = subset(smoted_data, select = -c(connection))
x$mainactivity = as.numeric(x$mainactivity)

model = "SL.glm"
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = model,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))


regressionResult = sl$AllSL[[1]]$fitLibrary$SL.glm_All$object
saveFile = paste("Resultsestimations/logitResult.Rdata", sep="")
save(regressionResult,file=saveFile)

dfRegression=tidy(regressionResult)
write.csv(dfRegression, "Resultsestimations/logitResult.csv")
# export of results of Logit - Table S1

###############################################################################################
################### Confusion matrices - Table S2
# Logit
x$nacecode = as.numeric(x$nacecode)
x$conscode = as.numeric(x$conscode)
x$legalstatus = as.numeric(x$legalstatus)
x$nationallegalform = as.numeric(x$nationallegalform)
x$nacerev2mainsection = as.numeric(x$nacerev2mainsection)
x$categoryofcompany = as.numeric(x$categoryofcompany)
x$mainexchange = as.numeric(x$mainexchange)
#x$publiclyquoted = as.numeric(x$publiclyquoted)
#x$bvdindependenceindicator = as.numeric(x$bvdindependenceindicator)
x$mainactivity = as.numeric(x$mainactivity)
#x$secondaryactivity = as.numeric(x$secondaryactivity)
model = "SL.glm"
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = model,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))


convPreds = ifelse(sl$library.predict>=0.5,1,0)
trueValues = sl$Y

cmLogit = confusionMatrix(factor(convPreds),factor(trueValues))  
as.table(cmLogit) 

# LASSO
model = "SL.glmnet"
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = model,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))

convPreds = ifelse(sl$library.predict>=0.5,1,0)
trueValues = sl$Y

cmLasso = confusionMatrix(factor(convPreds),factor(trueValues))  
as.table(cmLasso) 

# Ridge
learners    = create.Learner("SL.glmnet", params = list(alpha = 0))
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = learners$names,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))
convPreds = ifelse(sl$library.predict>=0.5,1,0)
trueValues = sl$Y

cmRidge = confusionMatrix(factor(convPreds),factor(trueValues))  
as.table(cmRidge) 

# Random forest
model = "SL.randomForest"
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = model,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))
convPreds = ifelse(sl$library.predict>=0.5,1,0)
trueValues = sl$Y

cmRF = confusionMatrix(factor(convPreds),factor(trueValues))  
as.table(cmRF) 

# Boosting
model = "SL.xgboost"
sl = CV.SuperLearner(Y = as.numeric(as.character(y)), X = x, 
                     family = binomial(), SL.library = model,
                     control = list(saveFitLibrary = TRUE),
                     cvControl = list(V = cvFolds, shuffle = FALSE))
convPreds = ifelse(sl$library.predict>=0.5,1,0)
trueValues = sl$Y

cmBoosting = confusionMatrix(factor(convPreds),factor(trueValues))  
as.table(cmBoosting)

###### PARTIAL DEP ############################################################################
###############################################################################################
#### We choose sample as before
size = 1000

set.seed(123)
# balance dataset here (SMOTE)
data_pos <- subset(data,connection==1)
data_neg <- subset(data,connection==0)

smoted_data <- data_neg[sample(length(data_pos[,1]),replace = FALSE),]
smoted_data <- rbind(smoted_data,data_pos)

y = smoted_data$connection
x = subset(smoted_data, select = -c(connection))

sampling = sample(length(y), size)

y = y[sampling]
x = x[sampling,]


### Let us train random forest model and plot PARTIAL DEPENDENCE PLOT
rfModel <- randomForest(x=x,y=y,ntree=1000,keep.forest = TRUE,importance = TRUE)

pd <- rfModel %>% partial(pred.var = c("age", "sum_procurement_contracts"), n.trees = 1000)

plotPartialDep <- plot_ly(x = pd$age, y = pd$sum_procurement_contracts, z = pd$yhat, type = 'mesh3d')
plotPartialDep <- plotPartialDep %>% layout(scene = list(xaxis = list(title = "Age"),
                                                         yaxis = list(title = "Value of public contracts"),
                                                         zaxis = list(title = "Likelihood of political connection")))
plotPartialDep
# save manually after "Show in New Window" in RStudio