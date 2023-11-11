# Political connections

# Change path to dataset & load libraries

pathVita="C:/Users/Titl0001/Dropbox/research/papers/identifyin-pol-conn/"
setwd(pathVita)

rm(list=ls())

library(haven)
library(SuperLearner)
library(varhandle)  
library(caret)
library(dplyr)
library(DMwR)


# Set a seed for reproducibility in this random sampling.
set.seed(1)


type = "all"
#type = "donors"
#type = "ceo_donors"
#type = "connections"

#year = "2011"
year = "2018"

loadFile = paste("Resultsestimations/results_CMs_by_size_accuracyBalanced_",year,"_",type,".Rdata", sep="")
load(file=loadFile)

###############################################################################################
####################################### 6- GRAPHS #############################################
models <- c("SL.glm", "SL.xgboost", "SL.randomForest", "SL.glmnet", "SL.glmnet")
sizes_steps_1 <- seq(100,1000,100)
sizes_steps_2 <- seq(2000,10000,1000)

sizes = c(sizes_steps_1,sizes_steps_2)

data_graph = data.frame(model = character(), size = integer(), value = double(), lower = double(), upper = double(), 
                        sensitivity = double(), specificity = double())
names(data_graph)

ridgeDone=0

# This creates dataframe with data that can be fed to ggplot below
for (j in (1:length(models))) {
  model = models[j]
  if (model=="SL.randomForest"){
    model =  "Random forest"
  }
  else if (model=="SL.nnet"){
    model =  "Neural network"
  }
  else if (model=="SL.xgboost"){
    model =  "Boosting"
  }
  else if (model=="SL.glm"){
    model =  "Logit"
  }
  else if (model=="SL.glmnet" & ridgeDone==0){
    model = "Ridge regression"
    ridgeDone=1 
  }
  else{
    model = "LASSO"
  }
  
  for(i in (1:length(sizes))) {
    order = (i-1)*length(models)+j
    print(order)
    # accuracy
    accuracy = results_cm_by_size[[order]]$overall[1]
    lower = results_cm_by_size[[order]]$overall[3]
    upper = results_cm_by_size[[order]]$overall[4]
    
    sensitivity = results_cm_by_size[[order]]$byClass[1]
    specificity = results_cm_by_size[[order]]$byClass[2]
    
    # Pos Pred Value? is that what we want???
    
    size = sizes[i]
    
    new_row = data.frame("model" = model,"size" = size, "value" = accuracy, "lower" =lower, "upper" = upper, 
                         "sensitivity" = sensitivity, "specificity" = specificity)
    data_graph <- rbind(data_graph, new_row)
  }
}
# The following ggplots are Figures 2-7 and S3-11 (depending on the type of connections and year set)
### accuracy with CIs
p<-ggplot(data=data_graph, aes(x=size, y=value, colour=model)) + geom_point() + geom_line() + 
  scale_x_continuous(breaks=c(seq(0,1000,1000),seq(2000,10000,2000))) + ylim(0.65, 1) + 
  theme(axis.text = element_text(size = 14),axis.title = element_text(size = 18), legend.text = element_text(size = 14)) +
  ylab("Accuracy") + xlab("No. of connected and non-connected firms in the training sample") + theme(legend.title=element_blank())
p<-p+geom_ribbon(aes(ymin=lower, ymax=upper), linetype=2, alpha=0.1)
# for instance 5000 means in the training
p
ggsave(paste("Graphs/methods-accuracy-cv-clean-",year,"-",type,".png",sep=""), width = 30, height = 20, units = "cm")

### sensitivity
graph_sensitivity = ggplot(data=data_graph, aes(x=size, y=sensitivity, colour=model)) + geom_point() + geom_line() + 
  scale_x_continuous(breaks=c(seq(0,1000,1000),seq(2000,10000,2000))) + ylim(0.65, 1) + 
  theme(axis.text = element_text(size = 14),axis.title = element_text(size = 18), legend.text = element_text(size = 14)) +
  ylab("True positive rate") + xlab("No. of connected and non-connected firms in the training sample") + theme(legend.title=element_blank())
graph_sensitivity
ggsave(paste("Graphs/methods-sensitivity-cv-clean-",year,"-",type,".png",sep=""), width = 30, height = 20, units = "cm")

### specificity
graph_specificity = ggplot(data=data_graph, aes(x=size, y=specificity, colour=model)) + geom_point() + geom_line() + 
  scale_x_continuous(breaks=c(seq(0,500,500),seq(1000,10000,1000))) + ylim(0.65, 1) + 
  theme(axis.text = element_text(size = 14),axis.title = element_text(size = 18), legend.text = element_text(size = 14)) +
  ylab("True negative rate") + xlab("No. of connected and non-connected firms in the training sample") + theme(legend.title=element_blank())
graph_specificity
ggsave(paste("Graphs/methods-specificity-cv-clean-",year,"-",type,".png",sep=""), width = 30, height = 20, units = "cm")
