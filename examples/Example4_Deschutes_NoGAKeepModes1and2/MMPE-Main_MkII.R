# MMPE-Main: R SCRIPT CONTAINING PROTOTYPE MULTI-MODEL PREDICTION ANALTYICS ENGINE


# MMPE main program and all custom R functions called by it are Copyright 2017, 2018, 2019 White Rabbit R&D LLC, Corvallis, Oregon USA.
# Perpetual and non-exclusive license to use the MMPE is granted to Elyon International Inc of Vancouver, Washington USA and the National Water and Climate 
# Center, Natural Resources Conservation Service, US Department of Agriculture, Portland, Oregon USA.
# Rights, conditions, limitations, and caveats around the MMPE, and all White Rabbit R&D LLC's work leading to it and otherwise associated with it, and its 
# use, are as per the 28 November 2016 contract between White Rabbit R&D LLC and Elyon International Inc and other relevant statements of purpose and 
# limitation including in particular the initial 2017-phase White Rabbit R&D LLC project report and its disclaimer as accepted by the NWCC.


### Purpose of main program is to obtain data and run options, manage required activities, and store the outcomes. For inverse runs (model-building using
### historical data) the required activities include shipping information off to modules (custom external R functions) for model training, feature creation 
### and optimal selection, testing/validation, and most of the diagnostics, combining outcomes to form an ensemble, performing some simple optimization 
### around the ensemble and neural network configurations, and performing some additional graphical diagnostics.  For forward runs (running a previously 
### calibrated and saved set of models using new predictor data) the required activities include loading the existing built models and running them using 
### current data.



###############################################################################################################################################################
#####   PRELIMINARIES   #######################################################################################################################################
###############################################################################################################################################################


###############################################################################################################################################################

# CLEAN UP WORKSPACE AND BEGIN STOPWATCH

graphics.off()                                                                      # close all plot windows
rm(list=ls())                                                                       # remove all variables, data frames, etc
cat("\014")                                                                         # clear console in Rstudio
start_time <- Sys.time()                                                            # start timing the run


###############################################################################################################################################################

# READ IN RUN CONTROL PARAMETERS

# read MMPE_RunControlFile.txt, convert table to matrix to make input cleaner to manipulate

RunControlParameters <- read.table("MMPE_RunControlFile.txt")  
RunControlParameters <- as.matrix(RunControlParameters)                                

# begin transcribing matrix of run control values to individual variables as needed

RunTypeFlag <- RunControlParameters[1]                # block containing basic run parameters                                         
errorlog_flag <- RunControlParameters[2]

GeneticAlgorithmFlag <- RunControlParameters[3]                   # inverse run: block of flags and hyperparameters controlling feature creation & selection
MaxModes <- as.numeric(RunControlParameters[4])
DisplayFlag <- RunControlParameters[5]
MinNumVars <- as.numeric(RunControlParameters[6])
GAPopSize <- as.numeric(RunControlParameters[7])
GANumGens <- as.numeric(RunControlParameters[8])
ManualPCSelection <- as.numeric(strtoi(strsplit(RunControlParameters[9], ",")[[1]]))

AutoEnsembleFlag <- RunControlParameters[10]                      # inverse run: block controlling ensemble generation
EnsembleFlag_PCR <- RunControlParameters[11]
EnsembleFlag_PCR_BC <- RunControlParameters[12] 
EnsembleFlag_PCQR <- RunControlParameters[13]       
EnsembleFlag_PCANN <- RunControlParameters[14]        
EnsembleFlag_PCANN_BC <- RunControlParameters[15]     
EnsembleFlag_PCRF <- RunControlParameters[16]         
EnsembleFlag_PCRF_BC <- RunControlParameters[17]      
EnsembleFlag_PCMCQRNN <- RunControlParameters[18]     
EnsembleFlag_PCSVM <- RunControlParameters[19]        
EnsembleFlag_PCSVM_BC <- RunControlParameters[20]     

AutoANNConfigFlag <- RunControlParameters[21]                     # inverse run: block controlling configuration & parallelization of neural nets
ANN_config1_cutoff <- as.numeric(RunControlParameters[22])
mANN_config_selection <- as.numeric(RunControlParameters[23])
MCQRNN_config_selection <- as.numeric(RunControlParameters[24])
ANN_monotone_flag <- RunControlParameters[25] 
ANN_parallel_flag <- RunControlParameters[26]          
num_cores <- as.numeric(RunControlParameters[27])                 

SVM_config_selection <- as.numeric(RunControlParameters[28])      # inverse run: block controlling SVM configuration
fixedgamma <- as.numeric(RunControlParameters[29])                

PC1form_plot_flag <- RunControlParameters[30]                     # inverse run: block controlling generation of some outputs
PC12form_plot_flag <- RunControlParameters[31]       

PCSelection_Frwrd_LR <- as.numeric(strtoi(strsplit(RunControlParameters[32], ",")[[1]]))             # forward run: block specifying which PCA modes to retain
PCSelection_Frwrd_QR <- as.numeric(strtoi(strsplit(RunControlParameters[33], ",")[[1]]))       
PCSelection_Frwrd_mANN <- as.numeric(strtoi(strsplit(RunControlParameters[34], ",")[[1]]))     
PCSelection_Frwrd_MCQRNN <- as.numeric(strtoi(strsplit(RunControlParameters[35], ",")[[1]]))    
PCSelection_Frwrd_RF <- as.numeric(strtoi(strsplit(RunControlParameters[36], ",")[[1]]))     
PCSelection_Frwrd_SVM <- as.numeric(strtoi(strsplit(RunControlParameters[37], ",")[[1]]))    

VariableSelection_Frwrd_LR <- as.numeric(strtoi(strsplit(RunControlParameters[38], " ")[[1]]))       # forward run: block specifying which input variables to retain
VariableSelection_Frwrd_QR <- as.numeric(strtoi(strsplit(RunControlParameters[39], " ")[[1]]))        
VariableSelection_Frwrd_mANN <- as.numeric(strtoi(strsplit(RunControlParameters[40], " ")[[1]]))      
VariableSelection_Frwrd_MCQRNN <- as.numeric(strtoi(strsplit(RunControlParameters[41], " ")[[1]]))    
VariableSelection_Frwrd_RF <- as.numeric(strtoi(strsplit(RunControlParameters[42], " ")[[1]]))        
VariableSelection_Frwrd_SVM <- as.numeric(strtoi(strsplit(RunControlParameters[43], " ")[[1]]))       

ANN_monotone_flag_Frwrd <- RunControlParameters[44]                                                  # forward run: housekeeping for monotonic neural nets

Ensemble_flag_frwrd <- RunControlParameters[45]                                                      # forward run: block specifying ensemble creation options
Ensemble_type_frwrd <- RunControlParameters[46]


###############################################################################################################################################################

# OPEN LOG FILE IF REQUESTED

if (errorlog_flag == "Y") {
  logfile <- file("error_messages_log.txt", open="wt")                               
  sink(logfile,type="message")           
}



###############################################################################################################################################################
####  PERFORM INVERSE RUN  ####################################################################################################################################
###############################################################################################################################################################


if (RunTypeFlag == "BUILD") {
  
  
###############################################################################################################################################################

# LOAD R LIBRARIES & ANNOUNCE CUSTOM FUNCTIONS IN EXTERNAL FILES
  
library(akima)           # contains interpolation function used for plotting certain results

source("PCR-Module_v11.R")
source("PCQR-Module_v7.R")
source("PCANN-Module_v17.R")
source("PCRF-Module_v3.R")
source("PCMCQRNN-Module_v5.R")
source("PCSVM-Module_v6.R")
source("Diagnostics-Module_v6.R")
source("AppendEnsemble-Module_v1.R")
source("InitializeEnsemble-Module_v1.R")
source("FinalizeEnsemble-Module_v1.R")
source("PCA-Module_v5.R")
source("PCA-Graphics-Module_v3.R")
source("GA-PredictorSelection-Module_v15.R")
  
  
###############################################################################################################################################################
  
# READ IN DATA AND PERFORM SOME INITIAL PROCESSING

# Create PCA-format Z-row by N-column candidate data matrix from external input file   
  
dat <- read.table("MMPEInputData_ModelBuildingMode.txt",header=TRUE)
# dat <- read.table("SyntheticTestData.txt",header=TRUE)
year <- dat[ ,1]  # first column of input data file contains the year
obs <- dat[ ,2]   # second column of input data file contains the predictand
dimension <- dim(dat)
N <-dimension[1]  # number of samples (years)
NumCol <- dimension[2]
datmat <- dat[ ,3:NumCol]
datmat <- t(datmat)
Z <- NumCol - 2   # number of  input variables in candidate pool
rm(dimension,NumCol)
  
# Perform some error-trapping
  
# if (GeneticAlgorithmFlag == "N") {
#   if ((max(ManualPCSelection) >= Z) || (max(ManualPCSelection) > 4)) {
#     print("INPUT ERROR - stopping run")
#     print ("Specified number of PCA modes to retain must be < total number of input variables in candidate pool and < 5")
#     quit(save="ask")
#   }
# }
if (GeneticAlgorithmFlag == "Y") {
  if (MaxModes >= Z || MaxModes > 4) {
    print("INPUT ERROR - stopping run")
    print ("Specified number of PCA modes to retain must be < total number of input variables in candidate pool and < 5")
    quit(save="ask")
  }
}
  
# Find category cutoffs between below-normal, normal, and above-normal flows
  
Qcrit <- numeric(3)  # terciles of observational flow series so each category is equiprobable; used in skill calculations
Qcrit[1] <- quantile(obs, 1/3)
Qcrit[2] <- quantile(obs, 2/3)
Qcrit[3] <- quantile(obs, 1)
  
  
###############################################################################################################################################################

# PERFORM LINEAR REGRESSION ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {
  
  PCSelection <- ManualPCSelection
  
  # perform PCA on all variables in input data matrix
  PCA(datmat,PCSelection)
  
  # Perform linear regression on user-selected PCs
  PCR(PCSelection)

}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {
  
  # Use genetic algorithm to find optimal predictors
  plot.new()
  dev.new()
  model_flag <- "PCR"
  PredictorOptimization(model_flag)   # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCR.txt")
  
  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)
  
  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1  
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]    
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCRmodel,LOOCV_Rsqrd_PCRmodel,PCR_model_summary,PCRmodel,prd_PCR,prd_PCR_LOOCV,res_PCR,res_PCR_LOOCV,y10_PCR,y10_PCR_BCbased,y30_PCR,y30_PCR_BCbased,y70_PCR,y70_PCR_BCbased,y90_PCR,y90_PCR_BCbased,Ymod_PCR,Ymod_PCR_BC)
  
  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  PCR(PCSelection)
  
}

# Find, plot, and save PCA-related information for PCR, tidy up workspace:

PCAgraphics("PCR: PCA eigenspectrum","PCR: PC time series","PCR: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCR_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCR_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCR_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCR_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCR_StdevOfEachRetainedInputVariate.csv")
PC1_PCR <- PC1
if (exists("PC2") == TRUE) {PC2_PCR <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# PERFORM QUANTILE REGRESSION ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {

  PCSelection <- ManualPCSelection

  # perform PCA on all variables in input data matrix
  PCA(datmat,PCSelection)

  # Perform quantile regression on user-selected PCs
  PCQR(PCSelection)

}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {

  # Use genetic algorithm to find optimal predictors
  dev.new()                             # without this, plot autogenerated by rbga.bin in PredictorOptimization function appears to overwrite PCR ordination diagram
  model_flag <- "PCQR"
  PredictorOptimization(model_flag)     # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCQR.txt")

  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)

  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCQRmodel,LOOCV_Rsqrd_PCQRmodel,PCQR_model_summary,PCQRmodel,prd_PCQR,prd_PCQR_LOOCV,res_PCQR,res_PCQR_LOOCV,y10_PCQR,y30_PCQR,y70_PCQR,y90_PCQR,Ymod_PCQR)

  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  PCQR(PCSelection)

}

# Find, plot, and save PCA-related information for PCQR, tidy up workspace:

PCAgraphics("PCQR: PCA eigenspectrum","PCQR: PC time series","PCQR: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCQR_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCQR_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCQR_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCQR_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCQR_StdevOfEachRetainedInputVariate.csv")
PC1_PCQR <- PC1
if (exists("PC2") == TRUE) {PC2_PCQR <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# PERFORM RANDOM FORESTS MODELING ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {
  
  PCSelection <- ManualPCSelection
  
  # perform PCA on all variables in input data matrix
  PCA(datmat,PCSelection)
  
  # Perform RF regression on user-selected PCs
  PCRF(PCSelection)
  
}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {
  
  # Use genetic algorithm to find optimal predictors
  dev.new()                               # without this, plot autogenerated by rbga.bin in PredictorOptimization function appears to overwrite previous ordination diagram
  model_flag <- "PCRF"
  PredictorOptimization(model_flag)       # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCRF.txt")
  
  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)
  
  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCRFmodel,LOOCV_Rsqrd_PCRFmodel,PCRF_model_summary,PCRFmodel,prd_PCRF,prd_PCRF_LOOCV,res_PCRF,res_PCRF_LOOCV,y10_PCRF,y30_PCRF,y70_PCRF,y90_PCRF,Ymod_PCRF)
  
  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  dev.new()                               # without this, plot autogenerated by random forests algorithm appears to overwrite GA convergence plot for RF 
  PCRF(PCSelection)
  
}

# Find, plot, and save PCA-related information for PCRF, tidy up workspace:

PCAgraphics("PCRF: PCA eigenspectrum","PCRF: PC time series","PCRF: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCRF_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCRF_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCRF_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCRF_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCRF_StdevOfEachRetainedInputVariate.csv")
PC1_PCRF <- PC1
if (exists("PC2") == TRUE) {PC2_PCRF <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# PERFORM SUPPORT VECTOR MACHINE MODELING ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {
  
  PCSelection <- ManualPCSelection
  
  # perform PCA on all variables in input data matrix
  PCA(datmat,PCSelection)
  
  # Perform SVM regression on user-selected PCs
  PCSVM(PCSelection,SVM_config_selection,fixedgamma)
  
}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {
  
  # Use genetic algorithm to find optimal predictors
  dev.new()                               # without this, plot autogenerated by rbga.bin in PredictorOptimization function appears to overwrite previous ordination diagram
  model_flag <- "PCSVM"
  PredictorOptimization(model_flag)       # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCSVM.txt")
  
  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)
  
  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCSVMmodel,LOOCV_Rsqrd_PCSVMmodel,PCSVM_model_summary,PCSVMmodel,prd_PCSVM,prd_PCSVM_LOOCV,res_PCSVM,res_PCSVM_LOOCV,y10_PCSVM,y30_PCSVM,y70_PCSVM,y90_PCSVM,Ymod_PCSVM)
  
  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  dev.new()                               # without this, plot autogenerated by random forests algorithm appears to overwrite GA convergence plot for RF 
  PCSVM(PCSelection,SVM_config_selection,fixedgamma)
  
}

# Find, plot, and save PCA-related information for PCSVM, tidy up workspace:

PCAgraphics("PCSVM: PCA eigenspectrum","PCSVM: PC time series","PCSVM: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCSVM_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCSVM_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCSVM_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCSVM_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCSVM_StdevOfEachRetainedInputVariate.csv")
PC1_PCSVM <- PC1
if (exists("PC2") == TRUE) {PC2_PCSVM <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# PERFORM MONOTONE ARTIFICIAL NEURAL NETWORK MODELING ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {
  
  # perform PCA on all variables in input data matrix
  PCSelection <- ManualPCSelection
  PCA(datmat,PCSelection)
  
  # if fitting ANN using user-specified neural network configuration...
  if (AutoANNConfigFlag == "N") {
    PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
  } 
  
  # if fitting ANN using automated configuration selection based primarily on LOOCV RMSE and R^2 and secondarily on AIC...
  
  if (AutoANNConfigFlag == "Y") {
    
    mANN_config_selection <- 1
    PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
    RMSE_benchmark <- mean(c(LOOCV_RMSE_PCRmodel,LOOCV_RMSE_PCQRmodel,LOOCV_RMSE_PCRFmodel,LOOCV_RMSE_PCSVMmodel))
    Rsqrd_benchmark <- mean(c(LOOCV_Rsqrd_PCRmodel,LOOCV_Rsqrd_PCQRmodel,LOOCV_Rsqrd_PCRFmodel,LOOCV_Rsqrd_PCSVMmodel))
    RMSE_deficit <- 100*(LOOCV_RMSE_PCANNmodel - RMSE_benchmark) / RMSE_benchmark
    Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCANNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
    AIC_config1 <- AIC_ANN
    
    if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {  # test whether default ANN configuration is adequate - if not, try alternative
      
      mANN_config_selection <- 2
      PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
      RMSE_deficit <- 100*(LOOCV_RMSE_PCANNmodel - RMSE_benchmark) / RMSE_benchmark
      Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCANNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
      
      # keep this alternative configuration if (1) it does the trick, or (2) if it doesn't, then if it at least provides better bang for the buck than default - if not, go back to default
      
      if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {
        if (AIC_ANN > AIC_config1) {
          mANN_config_selection <- 1
          PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
        }
      }
    }
  }
}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {
  
  # Use genetic algorithm to find optimal predictors
  dev.new()                               # without this, plot autogenerated by rbga.bin in PredictorOptimization function appears to overwrite PCR ordination diagram
  model_flag <- "PCANN"
  PredictorOptimization(model_flag)       # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCANN.txt")
  
  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)
  
  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCANNmodel,LOOCV_Rsqrd_PCANNmodel,PCANNmodel,prd_PCANN,prd_PCANN_LOOCV,res_PCANN,res_PCANN_LOOCV,y10_PCANN,y10_PCANN_BCbased,y30_PCANN,y30_PCANN_BCbased,y70_PCANN,y70_PCANN_BCbased,y90_PCANN,y90_PCANN_BCbased,Ymod_PCANN,Ymod_PCANN_BC)
  
  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  # if fitting ANN using user-specified neural network configuration...
  if (AutoANNConfigFlag == "N") {PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)} 
  # if fitting ANN using automated configuration selection based primarily on LOOCV RMSE and R^2 and secondarily on AIC...
  if (AutoANNConfigFlag == "Y") {
    mANN_config_selection <- 1
    PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
    RMSE_benchmark <- mean(c(LOOCV_RMSE_PCRmodel,LOOCV_RMSE_PCQRmodel,LOOCV_RMSE_PCRFmodel,LOOCV_RMSE_PCSVMmodel))
    Rsqrd_benchmark <- mean(c(LOOCV_Rsqrd_PCRmodel,LOOCV_Rsqrd_PCQRmodel,LOOCV_Rsqrd_PCRFmodel,LOOCV_Rsqrd_PCSVMmodel))
    RMSE_deficit <- 100*(LOOCV_RMSE_PCANNmodel - RMSE_benchmark) / RMSE_benchmark
    Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCANNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
    AIC_config1 <- AIC_ANN
    if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {  # test whether default ANN configuration is adequate - if not, try alternative
      mANN_config_selection <- 2
      PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
      RMSE_deficit <- 100*(LOOCV_RMSE_PCANNmodel - RMSE_benchmark) / RMSE_benchmark
      Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCANNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
      # keep this alternative configuration if (1) it does the trick, or (2) if it doesn't, then if it at least provides better bang for the buck than default - if not, go back to default
      if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {
        if (AIC_ANN > AIC_config1) {
          mANN_config_selection <- 1
          PCANN(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
        }
      }
    }
  }
  
}

# Find, plot, and save PCA-related information for PCANN, tidy up workspace:

PCAgraphics("PCANN: PCA eigenspectrum","PCANN: PC time series","PCANN: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCANN_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCANN_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCANN_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCANN_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCANN_StdevOfEachRetainedInputVariate.csv")
PC1_PCANN <- PC1
if (exists("PC2") == TRUE) {PC2_PCANN <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# PERFORM MONOTONE COMPOSITE QUANTILE REGRESSION NEURAL NETWORK MODELING ON PCS

# If manual predictor selection enabled (GeneticAlgorithmFlag <- "N"):

if (GeneticAlgorithmFlag == "N") {
  
  # perform PCA on all variables in input data matrix
  PCSelection <- ManualPCSelection
  PCA(datmat,PCSelection)
  
  # if fitting MCQRNN using user-specified neural network configuration...
  if (AutoANNConfigFlag == "N") {
    PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
  } 
  
  # if fitting MCQRNN using automated configuration selection based primarily on LOOCV RMSE and R^2 and secondarily on AIC...
  
  if (AutoANNConfigFlag == "Y") {
    
    MCQRNN_config_selection <- 1
    PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
    RMSE_benchmark <- mean(c(LOOCV_RMSE_PCRmodel,LOOCV_RMSE_PCQRmodel,LOOCV_RMSE_PCRFmodel,LOOCV_RMSE_PCSVMmodel))
    Rsqrd_benchmark <- mean(c(LOOCV_Rsqrd_PCRmodel,LOOCV_Rsqrd_PCQRmodel,LOOCV_Rsqrd_PCRFmodel,LOOCV_Rsqrd_PCSVMmodel))
    RMSE_deficit <- 100*(LOOCV_RMSE_PCMCQRNNmodel - RMSE_benchmark) / RMSE_benchmark
    Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCMCQRNNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
    AIC_config1 <- AIC_MCQRNN
    
    if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {  # test whether default MCQRNN configuration is adequate - if not, try alternative
      
      MCQRNN_config_selection <- 2
      PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
      RMSE_deficit <- 100*(LOOCV_RMSE_PCMCQRNNmodel - RMSE_benchmark) / RMSE_benchmark
      Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCMCQRNNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
      
      # keep this alternative configuration if (1) it does the trick, or (2) if it doesn't, then if it at least provides better bang for the buck than default - if not, go back to default
      
      if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {
        if (AIC_MCQRNN > AIC_config1) {
          MCQRNN_config_selection <- 1
          PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
        }
      }
    }
  }
}

# If automatic predictor selection enabled (GeneticAlgorithmFlag <- "Y"):

if (GeneticAlgorithmFlag == "Y") {
  
  # Use genetic algorithm to find optimal predictors
  dev.new()                               # without this, plot autogenerated by rbga.bin in PredictorOptimization function appears to overwrite PCR ordination diagram
  model_flag <- "PCMCQRNN"
  PredictorOptimization(model_flag)       # call external function containing the GA call and the objective function including model fitting to candidate predictor set
  write(RunSummary, file = "GA_RunSummary_PCMCQRNN.txt")
  
  # Pull numerical information out of character-format GA output
  Optimal_S_char <- str_sub(RunSummary, start = -2-2*(Z+MaxModes-1)+1, end = -2)
  Optimal_S_charsplit <- strsplit(Optimal_S_char, " ")[[1]]
  Optimal_S_int <- strtoi(Optimal_S_charsplit)
  
  # Recreate optimal predictor set (optimal combination of input variables and retained PCs), tidy up workspace
  Optimal_Sdat <- Optimal_S_int[1:Z]
  datmatR <- datmat[Optimal_Sdat == 1, ]
  PCswitchpositions <- Optimal_S_int[(Z+1):(Z+MaxModes-1)]
  PCSelection <- integer(MaxModes)
  PCSelection[1] <- 1
  if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
  if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
  if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
  PCSelection <- PCSelection[PCSelection != 0]
  rm(RunSummary,Optimal_S_char,Optimal_S_charsplit,Optimal_Sdat,PCswitchpositions,A,E,PC1,PC2,PC3,PC4,lambda,LOOCV_RMSE_PCMCQRNNmodel,LOOCV_Rsqrd_PCMCQRNNmodel,PCMCQRNNmodel,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,res_PCMCQRNN,res_PCMCQRNN_LOOCV,y10_PCMCQRNN,y10_PCMCQRNN_BCbased,y30_PCMCQRNN,y30_PCMCQRNN_BCbased,y70_PCMCQRNN,y70_PCMCQRNN_BCbased,y90_PCMCQRNN,y90_PCMCQRNN_BCbased,Ymod_PCMCQRNN,Ymod_PCMCQRNN_BC)
  
  # Run model forward with the optimal predictor set
  PCA(datmatR,PCSelection)
  # if fitting MCQRNN using user-specified neural network configuration...
  if (AutoANNConfigFlag == "N") {PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)} 
  # if fitting MCQRNN using automated configuration selection based primarily on LOOCV RMSE and R^2 and secondarily on AIC...
  if (AutoANNConfigFlag == "Y") {
    MCQRNN_config_selection <- 1
    PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
    RMSE_benchmark <- mean(c(LOOCV_RMSE_PCRmodel,LOOCV_RMSE_PCQRmodel,LOOCV_RMSE_PCRFmodel,LOOCV_RMSE_PCSVMmodel))
    Rsqrd_benchmark <- mean(c(LOOCV_Rsqrd_PCRmodel,LOOCV_Rsqrd_PCQRmodel,LOOCV_Rsqrd_PCRFmodel,LOOCV_Rsqrd_PCSVMmodel))
    RMSE_deficit <- 100*(LOOCV_RMSE_PCMCQRNNmodel - RMSE_benchmark) / RMSE_benchmark
    Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCMCQRNNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
    AIC_config1 <- AIC_MCQRNN
    if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {  # test whether default MCQRNN configuration is adequate - if not, try alternative
      MCQRNN_config_selection <- 2
      PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
      RMSE_deficit <- 100*(LOOCV_RMSE_PCMCQRNNmodel - RMSE_benchmark) / RMSE_benchmark
      Rsqrd_deficit <- -100*(LOOCV_Rsqrd_PCMCQRNNmodel - Rsqrd_benchmark) / Rsqrd_benchmark
      # keep this alternative configuration if (1) it does the trick, or (2) if it doesn't, then if it at least provides better bang for the buck than default - if not, go back to default
      if ((RMSE_deficit > ANN_config1_cutoff) || (Rsqrd_deficit > ANN_config1_cutoff)) {
        if (AIC_MCQRNN > AIC_config1) {
          MCQRNN_config_selection <- 1
          PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
        }
      }
    }
  }
  
}

# Find, plot, and save PCA-related information for PCMCQRNN, tidy up workspace:

PCAgraphics("PCMCQRNN: PCA eigenspectrum","PCMCQRNN: PC time series","PCMCQRNN: PCA ordination diagram")
eigenvalue_table <- data.frame(perc_var_expl)
write.csv(eigenvalue_table, file = "PCMCQRNN_eigenspectrum.csv")
eigenvector_table <- data.frame(E)
write.csv(eigenvector_table, file = "PCMCQRNN_eigenvector.csv")
PCtimeseries_table <- data.frame(A)
write.csv(PCtimeseries_table, file = "PCMCQRNN_PCtimeseries.csv")
MeanOfEachVariate_table <- data.frame(MeanOfEachVariate)
write.csv(MeanOfEachVariate_table, file = "PCMCQRNN_MeanOfEachRetainedInputVariate.csv")
StdevOfEachVariate_table <- data.frame(StdevOfEachVariate)
write.csv(StdevOfEachVariate_table, file = "PCMCQRNN_StdevOfEachRetainedInputVariate.csv")
PC1_PCMCQRNN <- PC1
if (exists("PC2") == TRUE) {PC2_PCMCQRNN <- PC2}
rm(names,pointlabels,E,A,PC1,PC2,PC3,PC4,lambda,perc_var_expl,E1,E2,PCSelection,datmatR)


###############################################################################################################################################################

# FORM INITIAL MULTI-MODEL ENSEMBLE FROM PROBABILISTIC MEMBERS

# This is only an initial result if automatic ensemble generation is selected; it is the final result otherwise

INITIALIZE_ENSEMBLE()
if (EnsembleFlag_PCR == "Y" && AutoEnsembleFlag == "N") {
  APPEND_ENSEMBLE(y90_PCR,y70_PCR,prd_PCR,prd_PCR_LOOCV,y30_PCR,y10_PCR,Ymod_PCR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCR_BC == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCQR == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCRF == "Y" && AutoEnsembleFlag == "N") {
  APPEND_ENSEMBLE(y90_PCRF,y70_PCRF,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF,y10_PCRF,Ymod_PCRF,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCRF_BC == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCSVM == "Y" && AutoEnsembleFlag == "N") {
  APPEND_ENSEMBLE(y90_PCSVM,y70_PCSVM,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM,y10_PCSVM,Ymod_PCSVM,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCSVM_BC == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCANN == "Y" && AutoEnsembleFlag == "N") {
  APPEND_ENSEMBLE(y90_PCANN,y70_PCANN,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN,y10_PCANN,Ymod_PCANN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCANN_BC == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
if (EnsembleFlag_PCMCQRNN == "Y" || AutoEnsembleFlag == "Y") {
  APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
}
FINALIZE_ENSEMBLE()


###############################################################################################################################################################

# IF AUTOMATIC ENSEMBLE GENERATION ENABLED, CHECK ENSEMBLE FOR NON-PHYSICAL (NEGATIVE) VALUES AND ADJUST ENSEMBLE COMPOSITION ACCORDINGLY IF NEEDED

if (AutoEnsembleFlag == "Y") {

  min_index <- 0
  min_index_2ndpass <- 0
  LR_BC_exclusion_flag <- "No"
  QR_exclusion_flag <- "No"
  RF_BC_exclusion_flag <- "No"
  SVM_BC_exclusion_flag <- "No"
  ANN_BC_exclusion_flag <- "No"
  MCQRNN_exclusion_flag <- "No"
  
  # initial pass: use minimum value of the 90% exceedance probability flow as indicator; if there's a problem, identify which model contributes the most to it and remove that member from the ensemble
  if (min(y90_ensemble) < 0) {
    min_vals <- c(min(y90_PCR_BCbased),min(y90_PCQR),min(y90_PCANN_BCbased),min(y90_PCRF_BCbased),min(y90_PCMCQRNN),min(y90_PCSVM_BCbased))
    min_index <- which(min_vals == min(min_vals) )
    if (min_index == 1) {  # omit PCR_BC
      LR_BC_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    if (min_index == 2) {   # omit PCQR
      QR_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    if (min_index == 3) {   # omit PCANN_BC
      ANN_BC_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    if (min_index == 4) {   # omit PCRF_BC
      RF_BC_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    if (min_index == 5) {   # omit PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
      MCQRNN_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    if (min_index == 6) {   # omit PCSVM_BC
      SVM_BC_exclusion_flag <- "Yes"
      INITIALIZE_ENSEMBLE()
      APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      FINALIZE_ENSEMBLE()
    }
    
  }

    # second pass: if there's still a problem, identify which model contributes the most to it and remove that member from the ensemble
    if (min_index > 0 && min(y90_ensemble) < 0) {
      if (min_index == 1) {min_vals_2ndpass <- c(min(y90_PCQR),min(y90_PCANN_BCbased),min(y90_PCRF_BCbased),min(y90_PCMCQRNN),min(y90_PCSVM_BCbased))}
      if (min_index == 2) {min_vals_2ndpass <- c(min(y90_PCR_BCbased),min(y90_PCANN_BCbased),min(y90_PCRF_BCbased),min(y90_PCMCQRNN),min(y90_PCSVM_BCbased))}
      if (min_index == 3) {min_vals_2ndpass <- c(min(y90_PCR_BCbased),min(y90_PCQR),min(y90_PCRF_BCbased),min(y90_PCMCQRNN),min(y90_PCSVM_BCbased))}
      if (min_index == 4) {min_vals_2ndpass <- c(min(y90_PCR_BCbased),min(y90_PCQR),min(y90_PCANN_BCbased),min(y90_PCMCQRNN),min(y90_PCSVM_BCbased))}
      if (min_index == 5) {min_vals_2ndpass <- c(min(y90_PCR_BCbased),min(y90_PCQR),min(y90_PCANN_BCbased),min(y90_PCRF_BCbased),min(y90_PCSVM_BCbased))}
      if (min_index == 6) {min_vals_2ndpass <- c(min(y90_PCR_BCbased),min(y90_PCQR),min(y90_PCANN_BCbased),min(y90_PCRF_BCbased),min(y90_PCMCQRNN))}
      
      min_index_2ndpass <- which(min_vals_2ndpass == min(min_vals_2ndpass) )
      
      if (min_index == 1 && min_index_2ndpass == 1) {  # omit PCR_BC and PCQR
        QR_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 1 && min_index_2ndpass == 2) {   # omit PCR_BC and PCANN_BC
        ANN_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 1 && min_index_2ndpass == 3) {   # omit PCR_BC and PCRF_BC
        RF_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 1 && min_index_2ndpass == 4) {   # omit PCR_BC and PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
        MCQRNN_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 1 && min_index_2ndpass == 5) {   # omit PCR_BC and PCSVM_BC
        PCSVM_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }

      if (min_index == 2 && min_index_2ndpass == 1) {  # omit PCQR and PCR_BC
        LR_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 2 && min_index_2ndpass == 2) {   # omit PCQR and PCANN_BC
        ANN_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 2 && min_index_2ndpass == 3) {   # omit PCQR and PCRF_BC
        RF_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 2 && min_index_2ndpass == 4) {   # omit PCQR and PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
        MCQRNN_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 2 && min_index_2ndpass == 5) {   # omit PCQR and PCSVM_BC
        PCSVM_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      
      if (min_index == 3 && min_index_2ndpass == 1) {  # omit PCANN_BC and PCR_BC
        LR_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 3 && min_index_2ndpass == 2) {  # omit PCANN_BC and PCQR
        QR_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 3 && min_index_2ndpass == 3) {   # omit PCANN_BC and PCRF_BC
        RF_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 3 && min_index_2ndpass == 4) {   # omit PCANN_BC and PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
        MCQRNN_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 3 && min_index_2ndpass == 5) {   # omit PCANN_BC and PCSVM_BC
        PCSVM_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      
      if (min_index == 4 && min_index_2ndpass == 1) {  # omit PCRF_BC and PCR_BC
        LR_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 4 && min_index_2ndpass == 2) {  # omit PCRF_BC and PCQR
        QR_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 4 && min_index_2ndpass == 3) {   # omit PCRF_BC and PCANN_BC
        ANN_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 4 && min_index_2ndpass == 4) {   # omit PCRF_BC and PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
        MCQRNN_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 4 && min_index_2ndpass == 5) {   # omit PCRF_BC and PCSVM_BC
        PCSVM_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      
      if (min_index == 5 && min_index_2ndpass == 1) {  # omit PCMCQRNN and PCR_BC (this should not happen if lower = 0 in MCQRNN)
        LR_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 5 && min_index_2ndpass == 2) {  # omit PCMCQRNN and PCQR (this should not happen if lower = 0 in MCQRNN)
        QR_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 5 && min_index_2ndpass == 3) {   # omit PCMCQRNN and PCANN_BC (this should not happen if lower = 0 in MCQRNN)
        ANN_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 5 && min_index_2ndpass == 4) {   # omit PCMCQRNN and PCRF_BC (this should not happen if lower = 0 in MCQRNN)
        RF_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,prd_PCSVM,prd_PCSVM_LOOCV,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 5 && min_index_2ndpass == 5) {   # omit PCMCQRNN and PCSVM_BC (this should not happen if lower = 0 in MCQRNN)
        PCSVM_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      
      if (min_index == 6 && min_index_2ndpass == 1) {  # omit PCSVM_BC and PCR_BC
        LR_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 6 && min_index_2ndpass == 2) {  # omit PCSVM_BC and PCQR
        QR_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 6 && min_index_2ndpass == 3) {   # omit PCSVM_BC and PCANN_BC
        ANN_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 6 && min_index_2ndpass == 4) {   # omit PCSVM_BC and PCRF_BC
        RF_BC_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,prd_PCMCQRNN,prd_PCMCQRNN_LOOCV,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      if (min_index == 6 && min_index_2ndpass == 5) {   # omit PCSVM_BC and PCMCQRNN (this should not happen if lower = 0 in MCQRNN)
        MCQRNN_exclusion_flag <- "Yes"
        INITIALIZE_ENSEMBLE()
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,prd_PCR,prd_PCR_LOOCV,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,prd_PCQR,prd_PCQR_LOOCV,y30_PCQR,y10_PCQR,Ymod_PCQR,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,prd_PCANN,prd_PCANN_LOOCV,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,prd_PCRF,prd_PCRF_LOOCV,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
        FINALIZE_ENSEMBLE()
      }
      
  }

  Standard_Ensemble_Members <- c('PCR_BC excluded?','PCQR excluded?','PCANN_BC excluded?','PCRF_BC excluded?','PCMCQRN excluded?','PCSVM_BC excluded?')
  Exclusion_List <- c(LR_BC_exclusion_flag,QR_exclusion_flag,ANN_BC_exclusion_flag,RF_BC_exclusion_flag,MCQRNN_exclusion_flag,SVM_BC_exclusion_flag)
  AutomatedEnsembleMemberExclusions.output <<- data.frame(Standard_Ensemble_Members,Exclusion_List)
  write.csv(AutomatedEnsembleMemberExclusions.output, file = "AutomatedEnsembleMemberExclusions.csv")
  rm(LR_BC_exclusion_flag,QR_exclusion_flag,ANN_BC_exclusion_flag,RF_BC_exclusion_flag,MCQRNN_exclusion_flag,SVM_BC_exclusion_flag,Standard_Ensemble_Members,Exclusion_List)

}


###############################################################################################################################################################

# COMPLETE DIAGNOSTICS FOR ENSEMBLE AND EACH MEMBER

DIAGNOSTICS(year,N,prd_PCR,res_PCR,prd_PCR_LOOCV,res_PCR_LOOCV,obs,y90_PCR,y70_PCR,y30_PCR,y10_PCR,"PCR",Qcrit,Ymod_PCR)
assign("PCR_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCR_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCR_StandardPerformanceReportingSuite.data, file = "PCR_StandardPerformanceReportingSuite.csv")
write.csv(PCR_AdditionalReportingSuite.data, file = "PCR_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCR,res_PCR,prd_PCR_LOOCV,res_PCR_LOOCV,obs,y90_PCR_BCbased,y70_PCR_BCbased,y30_PCR_BCbased,y10_PCR_BCbased,"PCR (Box-Cox bounds)",Qcrit,Ymod_PCR_BC)
assign("PCR_BC_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCR_BC_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCR_BC_StandardPerformanceReportingSuite.data, file = "PCR_BC_StandardPerformanceReportingSuite.csv")
write.csv(PCR_BC_AdditionalReportingSuite.data, file = "PCR_BC_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCQR,res_PCQR,prd_PCQR_LOOCV,res_PCQR_LOOCV,obs,y90_PCQR,y70_PCQR,y30_PCQR,y10_PCQR,"PCQR",Qcrit,Ymod_PCQR)
assign("PCQR_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCQR_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCQR_StandardPerformanceReportingSuite.data, file = "PCQR_StandardPerformanceReportingSuite.csv")
write.csv(PCQR_AdditionalReportingSuite.data, file = "PCQR_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCANN,res_PCANN,prd_PCANN_LOOCV,res_PCANN_LOOCV,obs,y90_PCANN,y70_PCANN,y30_PCANN,y10_PCANN,"PCANN",Qcrit,Ymod_PCANN)
assign("PCANN_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCANN_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCANN_StandardPerformanceReportingSuite.data, file = "PCANN_StandardPerformanceReportingSuite.csv")
write.csv(PCANN_AdditionalReportingSuite.data, file = "PCANN_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCANN,res_PCANN,prd_PCANN_LOOCV,res_PCANN_LOOCV,obs,y90_PCANN_BCbased,y70_PCANN_BCbased,y30_PCANN_BCbased,y10_PCANN_BCbased,"PCANN (Box-Cox bounds)",Qcrit,Ymod_PCANN_BC)
assign("PCANN_BC_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCANN_BC_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCANN_BC_StandardPerformanceReportingSuite.data, file = "PCANN_BC_StandardPerformanceReportingSuite.csv")
write.csv(PCANN_BC_AdditionalReportingSuite.data, file = "PCANN_BC_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCRF,res_PCRF,prd_PCRF_LOOCV,res_PCRF_LOOCV,obs,y90_PCRF,y70_PCRF,y30_PCRF,y10_PCRF,"PCRF",Qcrit,Ymod_PCRF)
assign("PCRF_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCRF_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCRF_StandardPerformanceReportingSuite.data, file = "PCRF_StandardPerformanceReportingSuite.csv")
write.csv(PCRF_AdditionalReportingSuite.data, file = "PCRF_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCRF,res_PCRF,prd_PCRF_LOOCV,res_PCRF_LOOCV,obs,y90_PCRF_BCbased,y70_PCRF_BCbased,y30_PCRF_BCbased,y10_PCRF_BCbased,"PCRF (Box-Cox bounds)",Qcrit,Ymod_PCRF_BC)
assign("PCRF_BC_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCRF_BC_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCRF_BC_StandardPerformanceReportingSuite.data, file = "PCRF_BC_StandardPerformanceReportingSuite.csv")
write.csv(PCRF_BC_AdditionalReportingSuite.data, file = "PCRF_BC_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCMCQRNN,res_PCMCQRNN,prd_PCMCQRNN_LOOCV,res_PCMCQRNN_LOOCV,obs,y90_PCMCQRNN,y70_PCMCQRNN,y30_PCMCQRNN,y10_PCMCQRNN,"PCMCQRNN",Qcrit,Ymod_PCMCQRNN)
assign("PCMCQRNN_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCMCQRNN_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCMCQRNN_StandardPerformanceReportingSuite.data, file = "PCMCQRNN_StandardPerformanceReportingSuite.csv")
write.csv(PCMCQRNN_AdditionalReportingSuite.data, file = "PCMCQRNN_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCSVM,res_PCSVM,prd_PCSVM_LOOCV,res_PCSVM_LOOCV,obs,y90_PCSVM,y70_PCSVM,y30_PCSVM,y10_PCSVM,"PCSVM",Qcrit,Ymod_PCSVM)
assign("PCSVM_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCSVM_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCSVM_StandardPerformanceReportingSuite.data, file = "PCSVM_StandardPerformanceReportingSuite.csv")
write.csv(PCSVM_AdditionalReportingSuite.data, file = "PCSVM_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_PCSVM,res_PCSVM,prd_PCSVM_LOOCV,res_PCSVM_LOOCV,obs,y90_PCSVM_BCbased,y70_PCSVM_BCbased,y30_PCSVM_BCbased,y10_PCSVM_BCbased,"PCSVM (Box-Cox bounds)",Qcrit,Ymod_PCSVM_BC)
assign("PCSVM_BC_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("PCSVM_BC_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(PCSVM_BC_StandardPerformanceReportingSuite.data, file = "PCSVM_BC_StandardPerformanceReportingSuite.csv")
write.csv(PCSVM_BC_AdditionalReportingSuite.data, file = "PCSVM_BC_AdditionalReportingSuite.csv")

DIAGNOSTICS(year,N,prd_ensemble,res_ensemble,prd_ensemble_LOOCV,res_ensemble_LOOCV,obs,y90_ensemble,y70_ensemble,y30_ensemble,y10_ensemble,"ensemble",Qcrit,Ymod_ensemble)
assign("ensemble_StandardPerformanceReportingSuite.data",reporting_metrics.output)
assign("ensemble_AdditionalReportingSuite.data",other_metrics.output)
rm(reporting_metrics.output, other_metrics.output)
write.csv(ensemble_StandardPerformanceReportingSuite.data, file = "ensemble_StandardPerformanceReportingSuite.csv")
write.csv(ensemble_AdditionalReportingSuite.data, file = "ensemble_AdditionalReportingSuite.csv")


###############################################################################################################################################################

# COLLECT AND SAVE CALIBRATION-PERIOD PREDICTIONS

best_estimates_all_models <- data.frame(year,obs,prd_PCR,prd_PCQR,prd_PCANN,prd_PCRF,prd_PCMCQRNN,prd_PCSVM,prd_ensemble)
PCR_predictions <- data.frame(year,obs,prd_PCR,y90_PCR,y70_PCR,y30_PCR,y10_PCR)
PCR_predictions_BC <- data.frame(year,obs,prd_PCR,y90_PCR_BCbased,y70_PCR_BCbased,y30_PCR_BCbased,y10_PCR_BCbased)
PCQR_predictions <- data.frame(year,obs,prd_PCQR,y90_PCQR,y70_PCQR,y30_PCQR,y10_PCQR)
PCANN_predictions <- data.frame(year,obs,prd_PCANN,y90_PCANN,y70_PCANN,y30_PCANN,y10_PCANN)
PCANN_predictions_BC <- data.frame(year,obs,prd_PCANN,y90_PCANN_BCbased,y70_PCANN_BCbased,y30_PCANN_BCbased,y10_PCANN_BCbased)
PCRF_predictions <- data.frame(year,obs,prd_PCRF,y90_PCRF,y70_PCRF,y30_PCRF,y10_PCRF)
PCRF_predictions_BC <- data.frame(year,obs,prd_PCRF,y90_PCRF_BCbased,y70_PCRF_BCbased,y30_PCRF_BCbased,y10_PCRF_BCbased)
PCMCQRNN_predictions <- data.frame(year,obs,prd_PCMCQRNN,y90_PCMCQRNN,y70_PCMCQRNN,y30_PCMCQRNN,y10_PCMCQRNN)
PCSVM_predictions <- data.frame(year,obs,prd_PCSVM,y90_PCSVM,y70_PCSVM,y30_PCSVM,y10_PCSVM)
PCSVM_predictions_BC <- data.frame(year,obs,prd_PCSVM,y90_PCSVM_BCbased,y70_PCSVM_BCbased,y30_PCSVM_BCbased,y10_PCSVM_BCbased)
ensemble_predictions <- data.frame(year,obs,prd_ensemble,y90_ensemble,y70_ensemble,y30_ensemble,y10_ensemble)

residuals_all_models <- data.frame(year,obs,res_PCR,res_PCQR,res_PCANN,res_PCRF,res_PCMCQRNN,res_PCSVM,res_ensemble,res_PCR_LOOCV,res_PCQR_LOOCV,res_PCANN_LOOCV,res_PCRF_LOOCV,res_PCMCQRNN_LOOCV,res_PCSVM_LOOCV,res_ensemble_LOOCV)

write.csv(best_estimates_all_models, file = "BestEstimatesAllModels.csv")
write.csv(PCR_predictions, file = "PCR_predictions.csv")
write.csv(PCR_predictions_BC, file = "PCR_predictions_BC.csv")
write.csv(PCQR_predictions, file = "PCQR_predictions.csv")
write.csv(PCANN_predictions, file = "PCANN_predictions.csv")
write.csv(PCANN_predictions_BC, file = "PCANN_predictions_BC.csv")
write.csv(PCRF_predictions, file = "PCRF_predictions.csv")
write.csv(PCRF_predictions_BC, file = "PCRF_predictions_BC.csv")
write.csv(PCMCQRNN_predictions, file = "PCMCQRNN_predictions.csv")
write.csv(PCSVM_predictions, file = "PCSVM_predictions.csv")
write.csv(PCSVM_predictions_BC, file = "PCSVM_predictions_BC.csv")
write.csv(ensemble_predictions, file = "ensemble_predictions.csv")

write.csv(residuals_all_models, file = "residuals_all_models.csv")



###############################################################################################################################################################

# SAVE CONFIGURATION/HYPERPARAMETER INFORMATION

Hyperparameter_Name <- c('mANN_config_selection','MCQRNN_config_selection','SVM epsilon','SVM C','SVM gamma')  # Include major hyperparameters potentially optimized by code, irrespective of whether flags (e.g., AutoANNConfigFlag) were set in this specific run to optimize them
Hyperparameter_Value <- c(mANN_config_selection,MCQRNN_config_selection,optimalepsilon,optimalC,optimalgamma)
OptimizableHyperparameters.output <<- data.frame(Hyperparameter_Name,Hyperparameter_Value)
write.csv(OptimizableHyperparameters.output, file = "AutomaticallyOptimizableHyperparameters.csv")


###############################################################################################################################################################

# SAVE BOX-COX TRANSFORM-RELATED INFORMATION FOR PCR-BC, RF-BC, SVM-BC, AND ANN-BC

Postprocessed_Prediction_Bound_Information  <- c('optimal Box-Cox lambda, PCR','Box-Cox space LOOCV RMSE, PCR','optimal Box-Cox lambda, PCANN','Box-Cox space LOOCV RMSE, PCANN','optimal Box-Cox lambda, PCSVM','Box-Cox space LOOCV RMSE, PCSVM','optimal Box-Cox lambda, PCRF','Box-Cox space LOOCV RMSE, PCRF')  # Include information that would subsequently be required to build prediction bounds around predictions in forward run
Estimated_Value <- c(lambda_prd_PCR_LOOCV,LOOCV_RMSE_PCRmodel_BC,lambda_prd_PCANN_LOOCV,LOOCV_RMSE_PCANNmodel_BC,lambda_prd_PCSVM_LOOCV,LOOCV_RMSE_PCSVMmodel_BC,lambda_prd_PCRF_LOOCV,LOOCV_RMSE_PCRFmodel_BC)
BoxCoxInfo.output <<- data.frame(Postprocessed_Prediction_Bound_Information,Estimated_Value)
write.csv(BoxCoxInfo.output, file = "BoxCoxInfo.csv")


###############################################################################################################################################################

# SAVE INFORMATION FOR ANN-BC AND MCQRNN RE: WHETHER INPUTS ARE MULTIPLIED BY -1 TO REVERSE NEGATIVE FEATURE-TARGET CORRELATION IF MONOTONICITY CONSTRAINT IS SELECTED

if (ANN_monotone_flag == "Y") {
  monotonic_ANNs_reversecorrelationflag  <- c('mANN_PC1_multiply','mANN_PC2_multiply','mANN_PC3_multiply','mANN_PC4_multiply','MCQRNN_PC1_multiply','MCQRNN_PC2_multiply','MCQRNN_PC3_multiply','MCQRNN_PC4_multiply')  # Include information that would subsequently be required to run ANNs in forward run
  reversecorrelationflag_value <- c(mANN_PC1_multiply,mANN_PC2_multiply,mANN_PC3_multiply,mANN_PC4_multiply,MCQRNN_PC1_multiply,MCQRNN_PC2_multiply,MCQRNN_PC3_multiply,MCQRNN_PC4_multiply)
  MultiplyMinusOne.output <<- data.frame(monotonic_ANNs_reversecorrelationflag,reversecorrelationflag_value)
  write.csv(MultiplyMinusOne.output, file = "MonotonicityConstraintRequiredAdjustments.csv")
}


###############################################################################################################################################################

# SAVE INFORMATION NEEDED FOR POTENTIAL MANUAL CONSTRUCTION OF MULTI-MODEL ENSEMBLES AFTER MMPE RUN AND ASSOCIATED DIAGNOSTICS

Parameterlabels <- c('Qcrit[1]','Qcrit[2]','Qcrit[3]','N')
Parametervalues <- c(Qcrit,N)
QcritAndN.output <- data.frame(Parameterlabels,Parametervalues)
write.csv(QcritAndN.output, file = "QcritAndN.csv")

Ymod_PCR_1 <- Ymod_PCR[,1]
Ymod_PCR_2 <- Ymod_PCR[,2]
Ymod_PCR_3 <- Ymod_PCR[,3]
PCR_dataframeforpostprocessing <- data.frame(year,prd_PCR,res_PCR,prd_PCR_LOOCV,res_PCR_LOOCV,obs,y90_PCR,y70_PCR,y30_PCR,y10_PCR,Ymod_PCR_1,Ymod_PCR_2,Ymod_PCR_3)
write.csv(PCR_dataframeforpostprocessing,file="PCR_AdditionalDataForPostProcessing.csv")

Ymod_PCR_BC_1 <- Ymod_PCR_BC[,1]
Ymod_PCR_BC_2 <- Ymod_PCR_BC[,2]
Ymod_PCR_BC_3 <- Ymod_PCR_BC[,3]
PCR_BC_dataframeforpostprocessing <- data.frame(year,prd_PCR,res_PCR,prd_PCR_LOOCV,res_PCR_LOOCV,obs,y90_PCR_BCbased,y70_PCR_BCbased,y30_PCR_BCbased,y10_PCR_BCbased,Ymod_PCR_BC_1,Ymod_PCR_BC_2,Ymod_PCR_BC_3)
write.csv(PCR_BC_dataframeforpostprocessing,file="PCR_BC_AdditionalDataForPostProcessing.csv")

Ymod_PCQR_1 <- Ymod_PCQR[,1]
Ymod_PCQR_2 <- Ymod_PCQR[,2]
Ymod_PCQR_3 <- Ymod_PCQR[,3]
PCQR_dataframeforpostprocessing <- data.frame(year,prd_PCQR,res_PCQR,prd_PCQR_LOOCV,res_PCQR_LOOCV,obs,y90_PCQR,y70_PCQR,y30_PCQR,y10_PCQR,Ymod_PCQR_1,Ymod_PCQR_2,Ymod_PCQR_3)
write.csv(PCQR_dataframeforpostprocessing,file="PCQR_AdditionalDataForPostProcessing.csv")

Ymod_PCRF_1 <- Ymod_PCRF[,1]
Ymod_PCRF_2 <- Ymod_PCRF[,2]
Ymod_PCRF_3 <- Ymod_PCRF[,3]
PCRF_dataframeforpostprocessing <- data.frame(year,prd_PCRF,res_PCRF,prd_PCRF_LOOCV,res_PCRF_LOOCV,obs,y90_PCRF,y70_PCRF,y30_PCRF,y10_PCRF,Ymod_PCRF_1,Ymod_PCRF_2,Ymod_PCRF_3)
write.csv(PCRF_dataframeforpostprocessing,file="PCRF_AdditionalDataForPostProcessing.csv")

Ymod_PCRF_BC_1 <- Ymod_PCRF_BC[,1]
Ymod_PCRF_BC_2 <- Ymod_PCRF_BC[,2]
Ymod_PCRF_BC_3 <- Ymod_PCRF_BC[,3]
PCRF_BC_dataframeforpostprocessing <- data.frame(year,prd_PCRF,res_PCRF,prd_PCRF_LOOCV,res_PCRF_LOOCV,obs,y90_PCRF_BCbased,y70_PCRF_BCbased,y30_PCRF_BCbased,y10_PCRF_BCbased,Ymod_PCRF_BC_1,Ymod_PCRF_BC_2,Ymod_PCRF_BC_3)
write.csv(PCRF_BC_dataframeforpostprocessing,file="PCRF_BC_AdditionalDataForPostProcessing.csv")

Ymod_PCSVM_1 <- Ymod_PCSVM[,1]
Ymod_PCSVM_2 <- Ymod_PCSVM[,2]
Ymod_PCSVM_3 <- Ymod_PCSVM[,3]
PCSVM_dataframeforpostprocessing <- data.frame(year,prd_PCSVM,res_PCSVM,prd_PCSVM_LOOCV,res_PCSVM_LOOCV,obs,y90_PCSVM,y70_PCSVM,y30_PCSVM,y10_PCSVM,Ymod_PCSVM_1,Ymod_PCSVM_2,Ymod_PCSVM_3)
write.csv(PCSVM_dataframeforpostprocessing,file="PCSVM_AdditionalDataForPostProcessing.csv")

Ymod_PCSVM_BC_1 <- Ymod_PCSVM_BC[,1]
Ymod_PCSVM_BC_2 <- Ymod_PCSVM_BC[,2]
Ymod_PCSVM_BC_3 <- Ymod_PCSVM_BC[,3]
PCSVM_BC_dataframeforpostprocessing <- data.frame(year,prd_PCSVM,res_PCSVM,prd_PCSVM_LOOCV,res_PCSVM_LOOCV,obs,y90_PCSVM_BCbased,y70_PCSVM_BCbased,y30_PCSVM_BCbased,y10_PCSVM_BCbased,Ymod_PCSVM_BC_1,Ymod_PCSVM_BC_2,Ymod_PCSVM_BC_3)
write.csv(PCSVM_BC_dataframeforpostprocessing,file="PCSVM_BC_AdditionalDataForPostProcessing.csv")

Ymod_PCANN_1 <- Ymod_PCANN[,1]
Ymod_PCANN_2 <- Ymod_PCANN[,2]
Ymod_PCANN_3 <- Ymod_PCANN[,3]
PCANN_dataframeforpostprocessing <- data.frame(year,prd_PCANN,res_PCANN,prd_PCANN_LOOCV,res_PCANN_LOOCV,obs,y90_PCANN,y70_PCANN,y30_PCANN,y10_PCANN,Ymod_PCANN_1,Ymod_PCANN_2,Ymod_PCANN_3)
write.csv(PCANN_dataframeforpostprocessing,file="PCANN_AdditionalDataForPostProcessing.csv")

Ymod_PCANN_BC_1 <- Ymod_PCANN_BC[,1]
Ymod_PCANN_BC_2 <- Ymod_PCANN_BC[,2]
Ymod_PCANN_BC_3 <- Ymod_PCANN_BC[,3]
PCANN_BC_dataframeforpostprocessing <- data.frame(year,prd_PCANN,res_PCANN,prd_PCANN_LOOCV,res_PCANN_LOOCV,obs,y90_PCANN_BCbased,y70_PCANN_BCbased,y30_PCANN_BCbased,y10_PCANN_BCbased,Ymod_PCANN_BC_1,Ymod_PCANN_BC_2,Ymod_PCANN_BC_3)
write.csv(PCANN_BC_dataframeforpostprocessing,file="PCANN_BC_AdditionalDataForPostProcessing.csv")

Ymod_PCMCQRNN_1 <- Ymod_PCMCQRNN[,1]
Ymod_PCMCQRNN_2 <- Ymod_PCMCQRNN[,2]
Ymod_PCMCQRNN_3 <- Ymod_PCMCQRNN[,3]
PCMCQRNN_dataframeforpostprocessing <- data.frame(year,prd_PCMCQRNN,res_PCMCQRNN,prd_PCMCQRNN_LOOCV,res_PCMCQRNN_LOOCV,obs,y90_PCMCQRNN,y70_PCMCQRNN,y30_PCMCQRNN,y10_PCMCQRNN,Ymod_PCMCQRNN_1,Ymod_PCMCQRNN_2,Ymod_PCMCQRNN_3)
write.csv(PCMCQRNN_dataframeforpostprocessing,file="PCMCQRNN_AdditionalDataForPostProcessing.csv")


###############################################################################################################################################################

# PLOT SOME MULTI-MODEL FIGURES IF REQUESTED

if (PC1form_plot_flag == "Y") {
  
  dev.new()
  plot.new()
  plot(PC1_PCR,obs,cex=1.5,main="model forms of PC1-Q relationship",col=rgb(0.2,0.4,0.1,0.7),pch = c(17),xlab="PC1",ylab="obs (kaf)")
  points(PC1_PCQR,obs,col=rgb(0.8,0.4,0.1,0.7),pch = c(19))
  points(PC1_PCANN,obs,col="indianred4",pch = c(15))
  points(PC1_PCRF,obs,col="wheat4", pch = c(13))
  points(PC1_PCMCQRNN,obs,col="lightsteelblue3",pch = c(18))
  points(PC1_PCSVM,obs,col="plum4",pch = c(17))
  
  tempmatrix1 <- cbind(PC1_PCR,prd_PCR)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col=rgb(0.2,0.4,0.1,0.7),lwd=2)
  rm(tempmatrix1,tempmatrix2)
  tempmatrix1 <- cbind(PC1_PCQR,prd_PCQR)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col=rgb(0.8,0.4,0.1,0.7),lwd=2)
  rm(tempmatrix1,tempmatrix2)
  tempmatrix1 <- cbind(PC1_PCANN,prd_PCANN)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col="indianred4",lwd=2)
  rm(tempmatrix1,tempmatrix2)
  tempmatrix1 <- cbind(PC1_PCRF,prd_PCRF)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col="wheat4",lwd=2)
  rm(tempmatrix1,tempmatrix2)
  tempmatrix1 <- cbind(PC1_PCMCQRNN,prd_PCMCQRNN)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col="lightsteelblue3",lwd=2)
  rm(tempmatrix1,tempmatrix2)
  tempmatrix1 <- cbind(PC1_PCSVM,prd_PCSVM)
  tempmatrix2 <- tempmatrix1[order(tempmatrix1[,1]),]
  matlines(tempmatrix2[1:N,1],tempmatrix2[1:N,2],col="plum4",lwd=2)
  rm(tempmatrix1,tempmatrix2)
  legend("top", legend = c("PCR", "PCQR","PCANN","PCRF","PCMCQRNN","PCSVM"),col = c(rgb(0.2,0.4,0.1,0.7), rgb(0.8,0.4,0.1,0.7),"indianred4","wheat4","lightsteelblue3","plum4"), pch = c(17,19,15,15,18,17),
         bty = "n", pt.cex = 2, cex = 1, text.col = "black", horiz = F , inset = c(0.1, 0.1))
  
}


if (PC12form_plot_flag == "Y") {
  
  all <- c(obs,prd_PCR,prd_PCQR,prd_PCANN,prd_PCRF,prd_PCMCQRNN,prd_PCSVM)
  range <- seq(min(all),max(all),length.out = 10)
  rm(all)
  all <- c(PC1_PCR,PC1_PCQR,PC1_PCANN,PC1_PCRF,PC1_PCMCQRNN,PC1_PCSVM)
  xrange <- c(min(all),max(all))
  rm(all)
  all <- NULL
  if (exists("PC2_PCR") == TRUE) {all <- c(all,PC2_PCR)}
  if (exists("PC2_PCQR") == TRUE) {all <- c(all,PC2_PCQR)}
  if (exists("PC2_PCANN") == TRUE) {all <- c(all,PC2_PCANN)}
  if (exists("PC2_PCRF") == TRUE) {all <- c(all,PC2_PCRF)}
  if (exists("PC2_PCMCQRNN") == TRUE) {all <- c(all,PC2_PCMCQRNN)}
  if (exists("PC2_PCSVM") == TRUE) {all <- c(all,PC2_PCSVM)}
  yrange <- c(min(all),max(all))
  rm(all)
  
  if (exists("PC2_PCR") == TRUE) {
    dev.new()
    plot.new()
    akima_PCR_obs.li <- interp(PC1_PCR,PC2_PCR,obs,duplicate = "mean")
    filled.contour(akima_PCR_obs.li,color.palette=topo.colors,xlim=xrange,ylim=yrange,levels=range,plot.title=title(main="LR PCs vs observed",xlab="PC1",ylab="PC2"),key.title=title(main="kaf"))
  }
  if (exists("PC2_PCR") == TRUE) {
    dev.new()
    plot.new()
    akima_PCR.li <- interp(PC1_PCR,PC2_PCR,prd_PCR,duplicate = "mean")
    filled.contour(akima_PCR.li,color.palette=topo.colors,xlim=xrange,ylim=yrange,levels=range,plot.title=title(main="PCR",xlab="PC1",ylab="PC2"),key.title=title(main="kaf"))
  }
  if (exists("PC2_PCQR") == TRUE) {
    dev.new()
    plot.new()
    akima_PCQR.li <- interp(PC1_PCQR,PC2_PCQR,prd_PCQR,duplicate = "mean")
    filled.contour(akima_PCQR.li, color.palette = topo.colors, xlim=xrange, ylim=yrange,levels = range, plot.title = title(main = "PCQR", xlab = "PC1", ylab = "PC2"), key.title = title(main = "kaf"))
  }
  if (exists("PC2_PCANN") == TRUE) {
    dev.new()
    plot.new()
    akima_PCANN.li <- interp(PC1_PCANN,PC2_PCANN,prd_PCANN,duplicate = "mean")
    filled.contour(akima_PCANN.li, color.palette = topo.colors, xlim=xrange, ylim=yrange,levels = range, plot.title = title(main = "PCANN", xlab = "PC1", ylab = "PC2"), key.title = title(main = "kaf"))
  }
  if (exists("PC2_PCRF") == TRUE) {
    dev.new()
    plot.new()
    akima_RF.li <- interp(PC1_PCRF,PC2_PCRF,prd_PCRF,duplicate = "mean")
    filled.contour(akima_RF.li, color.palette = topo.colors, xlim=xrange, ylim=yrange,levels = range, plot.title = title(main = "PCRF", xlab = "PC1", ylab = "PC2"), key.title = title(main = "kaf"))
  }
  if (exists("PC2_PCMCQRNN") == TRUE) {
    dev.new()
    plot.new()
    akima_MCQRNN.li <- interp(PC1_PCMCQRNN,PC2_PCMCQRNN,prd_PCMCQRNN,duplicate = "mean")
    filled.contour(akima_MCQRNN.li, color.palette = topo.colors, xlim=xrange, ylim=yrange,levels = range, plot.title = title(main = "PCMCQRNN", xlab = "PC1", ylab = "PC2"), key.title = title(main = "kaf"))
  }
  if (exists("PC2_PCSVM") == TRUE) {
    dev.new()
    plot.new()
    akima_SVM.li <- interp(PC1_PCSVM,PC2_PCSVM,prd_PCSVM,duplicate = "mean")
    filled.contour(akima_SVM.li, color.palette = topo.colors, xlim=xrange, ylim=yrange,levels = range, plot.title = title(main = "PCSVM", xlab = "PC1", ylab = "PC2"), key.title = title(main = "kaf"))
  }
  
}


###############################################################################################################################################################

# CLOSE OUT INVERSE RUN

}



###############################################################################################################################################################
####  PERFORM FORWARD RUN  ####################################################################################################################################
###############################################################################################################################################################


if (RunTypeFlag == "FORECAST") {

  
  #############################################################################################################################################################  
  
  # LOAD R LIBRARIES & ANNOUNCE CUSTOM FUNCTIONS IN EXTERNAL FILES
  
  library(forecast)             # contains functions for Box-Cox transform
  library(qrnn)                 # R package for neural network-based nonlinear quantile regression modeling with optional monotonicity and non-negativity constraints
  library(e1071)                # R package to perform SVM modeling
  library(randomForest)         # R package for random forests
  library(monmlp)               # R package for MLP modeling with optional monotonicity constraint
  source("AppendEnsemble-Module_v1.R")
  source("InitializeEnsemble-Module_v1.R")
  source("FinalizeEnsemble-Module_v1.R")
  
  
  #############################################################################################################################################################
  
  # READ IN EXISTING MODELS AND NEW DATA FROM FILES  
  
  # Read in new sample of predictor data from external input file, standardize using calibration-period mean and variance of each input variate
  
  datmat <- read.table("MMPEInputData_ForecastingMode.txt",header=TRUE)
  datmat <- t(datmat)
  
  datmat_PCR <- datmat[VariableSelection_Frwrd_LR == 1]
  MeanOfEachVariate <- read.table("PCR_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCR_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCR <- (datmat_PCR-MeanOfEachVariate)/StdevOfEachVariate
  
  datmat_PCQR <- datmat[VariableSelection_Frwrd_QR == 1]
  MeanOfEachVariate <- read.table("PCQR_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCQR_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCQR <- (datmat_PCQR-MeanOfEachVariate)/StdevOfEachVariate
  
  datmat_PCANN <- datmat[VariableSelection_Frwrd_mANN == 1]
  MeanOfEachVariate <- read.table("PCANN_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCANN_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCANN <- (datmat_PCANN-MeanOfEachVariate)/StdevOfEachVariate
  
  datmat_PCMCQRNN <- datmat[VariableSelection_Frwrd_MCQRNN == 1]
  MeanOfEachVariate <- read.table("PCMCQRNN_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCMCQRNN_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCMCQRNN <- (datmat_PCMCQRNN-MeanOfEachVariate)/StdevOfEachVariate
  
  datmat_PCRF <- datmat[VariableSelection_Frwrd_RF == 1]
  MeanOfEachVariate <- read.table("PCRF_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCRF_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCRF <- (datmat_PCRF-MeanOfEachVariate)/StdevOfEachVariate
  
  datmat_PCSVM <- datmat[VariableSelection_Frwrd_SVM == 1]
  MeanOfEachVariate <- read.table("PCSVM_MeanOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  MeanOfEachVariate <- MeanOfEachVariate[,2]
  StdevOfEachVariate <- read.table("PCSVM_StdevOfEachRetainedInputVariate.csv", header = TRUE, sep=",")
  StdevOfEachVariate <- StdevOfEachVariate[,2]
  datmat_PCSVM <- (datmat_PCSVM-MeanOfEachVariate)/StdevOfEachVariate
  
  # Read in existing models
  
  load("PCRmodel.Rdata")
  load("PCRFmodel.Rdata")
  load("PCSVMmodel.Rdata")
  load("PCANNmodel.Rdata")
  load("PCMCQRNNmodel.Rdata")
  QRcoeffs <- read.table("PCQR_model_summary.txt")
  
  # Read in corresponding eigenvectors
  
  E_PCR <- read.table("PCR_eigenvector.csv", header = TRUE, sep=",")
  E_PCR <- E_PCR[,-1]
  E_PCRF <- read.table("PCRF_eigenvector.csv", header = TRUE, sep=",")
  E_PCRF <- E_PCRF[,-1]
  E_PCANN <- read.table("PCANN_eigenvector.csv", header = TRUE, sep=",")
  E_PCANN <- E_PCANN[,-1]
  E_PCSVM <- read.table("PCSVM_eigenvector.csv", header = TRUE, sep=",")
  E_PCSVM <- E_PCSVM[,-1]
  E_PCMCQRNN <- read.table("PCMCQRNN_eigenvector.csv", header = TRUE, sep=",")
  E_PCMCQRNN <- E_PCMCQRNN[,-1]
  E_PCQR <- read.table("PCQR_eigenvector.csv", header = TRUE, sep=",")
  E_PCQR <- E_PCQR[,-1]
  
  # Read in ensemble member exclusions (in this version assume that AutoEnsembleFlag was Y when models were built)
  
  AutomatedEnsembleMemberExclusions <- read.table("AutomatedEnsembleMemberExclusions.csv", header = TRUE, sep = ",", colClasses = "character")
  LR_BC_exclusion_flag <- AutomatedEnsembleMemberExclusions[1,3]
  QR_exclusion_flag <- AutomatedEnsembleMemberExclusions[2,3]
  ANN_BC_exclusion_flag <- AutomatedEnsembleMemberExclusions[3,3]
  RF_BC_exclusion_flag <- AutomatedEnsembleMemberExclusions[4,3]
  MCQRNN_exclusion_flag <- AutomatedEnsembleMemberExclusions[5,3]
  SVM_BC_exclusion_flag <- AutomatedEnsembleMemberExclusions[6,3]
  
  # Read in Box-Cox transform space information
  
  BoxCoxInfo <- read.table("BoxCoxInfo.csv", header = TRUE, sep = ",")
  lambda_prd_PCR_LOOCV <- BoxCoxInfo[1,3]
  LOOCV_RMSE_PCRmodel_BC <- BoxCoxInfo[2,3]
  lambda_prd_PCANN_LOOCV <- BoxCoxInfo[3,3]
  LOOCV_RMSE_PCANNmodel_BC <- BoxCoxInfo[4,3]
  lambda_prd_PCSVM_LOOCV <- BoxCoxInfo[5,3]
  LOOCV_RMSE_PCSVMmodel_BC <- BoxCoxInfo[6,3]
  lambda_prd_PCRF_LOOCV <- BoxCoxInfo[7,3]
  LOOCV_RMSE_PCRFmodel_BC <- BoxCoxInfo[8,3]
  
  # Read in whether input PCs were multiplied by -1 to ensure positive feature-target correlations in the case where monotonicity contraints were enforced for ANNs
  
  if (ANN_monotone_flag_Frwrd == "Y") {
    MultiplyMinusOne <- read.table("MonotonicityConstraintRequiredAdjustments.csv", header = TRUE, sep = ",", colClasses = "character")
    mANN_PC1_multiply <- MultiplyMinusOne[1,3]
    mANN_PC2_multiply <- MultiplyMinusOne[2,3]
    mANN_PC3_multiply <- MultiplyMinusOne[3,3]
    mANN_PC4_multiply <- MultiplyMinusOne[4,3]
    MCQRNN_PC1_multiply <- MultiplyMinusOne[5,3]
    MCQRNN_PC2_multiply <- MultiplyMinusOne[6,3]
    MCQRNN_PC3_multiply <- MultiplyMinusOne[7,3]
    MCQRNN_PC4_multiply <- MultiplyMinusOne[8,3]
  }
  
  
  ############################################################################################################################################################# 
  
  # RUN LINEAR PRINCIPAL COMPONENTS REGRESSION FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on LR-specifc retained input variables and calibration-period eigenvectors
  
  A_PCR <- t(E_PCR)%*%datmat_PCR
  
  # Select LR-specific retained PCA modes and create corresponding data frame
  
  if (identical(PCSelection_Frwrd_LR, c(1))) {
    PC1 <- A_PCR[1,]
    dat_model_PCR <- data.frame(PC1)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,2))) {
    PC1 <- A_PCR[1,]
    PC2 <- A_PCR[2,]
    dat_model_PCR <- data.frame(PC1,PC2)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,2,3))) {
    PC1 <- A_PCR[1,]
    PC2 <- A_PCR[2,]
    PC3 <- A_PCR[3,]
    dat_model_PCR <- data.frame(PC1,PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,2,3,4))) {
    PC1 <- A_PCR[1,]
    PC2 <- A_PCR[2,]
    PC3 <- A_PCR[3,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC1,PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,3,4))) {
    PC1 <- A_PCR[1,]
    PC3 <- A_PCR[3,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC1,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,2,4))) {
    PC1 <- A_PCR[1,]
    PC2 <- A_PCR[2,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC1,PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,3))) {
    PC1 <- A_PCR[1,]
    PC3 <- A_PCR[3,]
    dat_model_PCR <- data.frame(PC1,PC3)
  }
  if (identical(PCSelection_Frwrd_LR, c(1,4))) {
    PC1 <- A_PCR[1,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC1,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(2))) {
    PC2 <- A_PCR[2,]
    dat_model_PCR <- data.frame(PC2)
  }
  if (identical(PCSelection_Frwrd_LR, c(2,3))) {
    PC2 <- A_PCR[2,]
    PC3 <- A_PCR[3,]
    dat_model_PCR <- data.frame(PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_LR, c(2,4))) {
    PC2 <- A_PCR[2,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(2,3,4))) {
    PC2 <- A_PCR[2,]
    PC3 <- A_PCR[3,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(3))) {
    PC3 <- A_PCR[3,]
    dat_model_PCR <- data.frame(PC34)
  }
  if (identical(PCSelection_Frwrd_LR, c(3,4))) {
    PC3 <- A_PCR[3,]
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_LR, c(4))) {
    PC4 <- A_PCR[4,]
    dat_model_PCR <- data.frame(PC4)
  }
  
  # Run PCR and obtain prediction intervals
  
  frwrd_PCR <- predict(PCRmodel,dat_model_PCR)
  
  frwrd_PCR_BC <- BoxCox(frwrd_PCR,lambda_prd_PCR_LOOCV)
  y90_PCR_BC <- frwrd_PCR_BC + (-1.282 * LOOCV_RMSE_PCRmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCR_BC <- frwrd_PCR_BC + (-0.524 * LOOCV_RMSE_PCRmodel_BC)
  y30_PCR_BC <- frwrd_PCR_BC + (0.524 * LOOCV_RMSE_PCRmodel_BC)
  y10_PCR_BC <- frwrd_PCR_BC + (1.282 * LOOCV_RMSE_PCRmodel_BC)
  
  y90_PCR_BCbased <- InvBoxCox(y90_PCR_BC,lambda_prd_PCR_LOOCV)   # inverse-transform results and return them to main program
  y70_PCR_BCbased <- InvBoxCox(y70_PCR_BC,lambda_prd_PCR_LOOCV)
  y30_PCR_BCbased <- InvBoxCox(y30_PCR_BC,lambda_prd_PCR_LOOCV)
  y10_PCR_BCbased <- InvBoxCox(y10_PCR_BC,lambda_prd_PCR_LOOCV)
  
  rm(frwrd_PCR_BC,y90_PCR_BC,y70_PCR_BC,y30_PCR_BC,y10_PCR_BC)
  
  
  ############################################################################################################################################################# 
  
  # RUN RANDOM FORESTS FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on RF-specifc retained input variables and calibration-period eigenvectors
  
  A_PCRF <- t(E_PCRF)%*%datmat_PCRF
  
  # Select RF-specific retained PCA modes and create corresponding data frame
  
  if (identical(PCSelection_Frwrd_RF, c(1))) {
    PC1 <- A_PCRF[1,]
    dat_model_PCRF <- data.frame(PC1)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,2))) {
    PC1 <- A_PCRF[1,]
    PC2 <- A_PCRF[2,]
    dat_model_PCRF <- data.frame(PC1,PC2)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,2,3))) {
    PC1 <- A_PCRF[1,]
    PC2 <- A_PCRF[2,]
    PC3 <- A_PCRF[3,]
    dat_model_PCRF <- data.frame(PC1,PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,2,3,4))) {
    PC1 <- A_PCRF[1,]
    PC2 <- A_PCRF[2,]
    PC3 <- A_PCRF[3,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC1,PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,3,4))) {
    PC1 <- A_PCRF[1,]
    PC3 <- A_PCRF[3,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC1,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,2,4))) {
    PC1 <- A_PCRF[1,]
    PC2 <- A_PCRF[2,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC1,PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,3))) {
    PC1 <- A_PCRF[1,]
    PC3 <- A_PCRF[3,]
    dat_model_PCRF <- data.frame(PC1,PC3)
  }
  if (identical(PCSelection_Frwrd_RF, c(1,4))) {
    PC1 <- A_PCRF[1,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC1,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(2))) {
    PC2 <- A_PCRF[2,]
    dat_model_PCRF <- data.frame(PC2)
  }
  if (identical(PCSelection_Frwrd_RF, c(2,3))) {
    PC2 <- A_PCRF[2,]
    PC3 <- A_PCRF[3,]
    dat_model_PCRF <- data.frame(PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_RF, c(2,4))) {
    PC2 <- A_PCRF[2,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(2,3,4))) {
    PC2 <- A_PCRF[2,]
    PC3 <- A_PCRF[3,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(3))) {
    PC3 <- A_PCRF[3,]
    dat_model_PCRF <- data.frame(PC34)
  }
  if (identical(PCSelection_Frwrd_RF, c(3,4))) {
    PC3 <- A_PCRF[3,]
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_RF, c(4))) {
    PC4 <- A_PCRF[4,]
    dat_model_PCRF <- data.frame(PC4)
  }
  
  # Run PCRF and obtain prediction intervals
  
  frwrd_PCRF <- predict(PCRFmodel,dat_model_PCRF)
  
  frwrd_PCRF_BC <- BoxCox(frwrd_PCRF,lambda_prd_PCRF_LOOCV)
  y90_PCRF_BC <- frwrd_PCRF_BC + (-1.282 * LOOCV_RMSE_PCRFmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCRF_BC <- frwrd_PCRF_BC + (-0.524 * LOOCV_RMSE_PCRFmodel_BC)
  y30_PCRF_BC <- frwrd_PCRF_BC + (0.524 * LOOCV_RMSE_PCRFmodel_BC)
  y10_PCRF_BC <- frwrd_PCRF_BC + (1.282 * LOOCV_RMSE_PCRFmodel_BC)
  
  y90_PCRF_BCbased <- InvBoxCox(y90_PCRF_BC,lambda_prd_PCRF_LOOCV)   # inverse-transform results and return them to main program
  y70_PCRF_BCbased <- InvBoxCox(y70_PCRF_BC,lambda_prd_PCRF_LOOCV)
  y30_PCRF_BCbased <- InvBoxCox(y30_PCRF_BC,lambda_prd_PCRF_LOOCV)
  y10_PCRF_BCbased <- InvBoxCox(y10_PCRF_BC,lambda_prd_PCRF_LOOCV)
  
  rm(frwrd_PCRF_BC,y90_PCRF_BC,y70_PCRF_BC,y30_PCRF_BC,y10_PCRF_BC)
  
  
  ############################################################################################################################################################# 
  
  # RUN SUPPORT VECTOR MACHINE FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on SVM-specifc retained input variables and calibration-period eigenvectors
  
  A_PCSVM <- t(E_PCSVM)%*%datmat_PCSVM
  
  # Select SVM-specific retained PCA modes and create corresponding data frame
  
  if (identical(PCSelection_Frwrd_SVM, c(1))) {
    PC1 <- A_PCSVM[1,]
    dat_model_PCSVM <- data.frame(PC1)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,2))) {
    PC1 <- A_PCSVM[1,]
    PC2 <- A_PCSVM[2,]
    dat_model_PCSVM <- data.frame(PC1,PC2)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,2,3))) {
    PC1 <- A_PCSVM[1,]
    PC2 <- A_PCSVM[2,]
    PC3 <- A_PCSVM[3,]
    dat_model_PCSVM <- data.frame(PC1,PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,2,3,4))) {
    PC1 <- A_PCSVM[1,]
    PC2 <- A_PCSVM[2,]
    PC3 <- A_PCSVM[3,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC1,PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,3,4))) {
    PC1 <- A_PCSVM[1,]
    PC3 <- A_PCSVM[3,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC1,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,2,4))) {
    PC1 <- A_PCSVM[1,]
    PC2 <- A_PCSVM[2,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC1,PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,3))) {
    PC1 <- A_PCSVM[1,]
    PC3 <- A_PCSVM[3,]
    dat_model_PCSVM <- data.frame(PC1,PC3)
  }
  if (identical(PCSelection_Frwrd_SVM, c(1,4))) {
    PC1 <- A_PCSVM[1,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC1,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(2))) {
    PC2 <- A_PCSVM[2,]
    dat_model_PCSVM <- data.frame(PC2)
  }
  if (identical(PCSelection_Frwrd_SVM, c(2,3))) {
    PC2 <- A_PCSVM[2,]
    PC3 <- A_PCSVM[3,]
    dat_model_PCSVM <- data.frame(PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_SVM, c(2,4))) {
    PC2 <- A_PCSVM[2,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(2,3,4))) {
    PC2 <- A_PCSVM[2,]
    PC3 <- A_PCSVM[3,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(3))) {
    PC3 <- A_PCSVM[3,]
    dat_model_PCSVM <- data.frame(PC34)
  }
  if (identical(PCSelection_Frwrd_SVM, c(3,4))) {
    PC3 <- A_PCSVM[3,]
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_SVM, c(4))) {
    PC4 <- A_PCSVM[4,]
    dat_model_PCSVM <- data.frame(PC4)
  }
  
  # Run PCSVM and obtain prediction intervals
  
  frwrd_PCSVM <- predict(PCSVMmodel,dat_model_PCSVM)
  
  frwrd_PCSVM_BC <- BoxCox(frwrd_PCSVM,lambda_prd_PCSVM_LOOCV)
  y90_PCSVM_BC <- frwrd_PCSVM_BC + (-1.282 * LOOCV_RMSE_PCSVMmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCSVM_BC <- frwrd_PCSVM_BC + (-0.524 * LOOCV_RMSE_PCSVMmodel_BC)
  y30_PCSVM_BC <- frwrd_PCSVM_BC + (0.524 * LOOCV_RMSE_PCSVMmodel_BC)
  y10_PCSVM_BC <- frwrd_PCSVM_BC + (1.282 * LOOCV_RMSE_PCSVMmodel_BC)
  
  y90_PCSVM_BCbased <- InvBoxCox(y90_PCSVM_BC,lambda_prd_PCSVM_LOOCV)   # inverse-transform results and return them to main program
  y70_PCSVM_BCbased <- InvBoxCox(y70_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  y30_PCSVM_BCbased <- InvBoxCox(y30_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  y10_PCSVM_BCbased <- InvBoxCox(y10_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  
  rm(frwrd_PCSVM_BC,y90_PCSVM_BC,y70_PCSVM_BC,y30_PCSVM_BC,y10_PCSVM_BC)
  
  
  ############################################################################################################################################################# 
  
  # RUN MONOTONE ARTIFICIAL NEURAL NETWORK FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on ANN-specifc retained input variables and calibration-period eigenvectors
  
  A_PCANN <- t(E_PCANN)%*%datmat_PCANN
  
  # Select ANN-specific retained PCA modes and create corresponding data frame
  
  if (identical(PCSelection_Frwrd_mANN, c(1))) {
    PC1 <- A_PCANN[1,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
    }
    x <- as.matrix(PC1)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,2))) {
    PC1 <- A_PCANN[1,]
    PC2 <- A_PCANN[2,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
    }
    x <- cbind(PC1,PC2)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,2,3))) {
    PC1 <- A_PCANN[1,]
    PC2 <- A_PCANN[2,]
    PC3 <- A_PCANN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC1,PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,2,3,4))) {
    PC1 <- A_PCANN[1,]
    PC2 <- A_PCANN[2,]
    PC3 <- A_PCANN[3,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,3,4))) {
    PC1 <- A_PCANN[1,]
    PC3 <- A_PCANN[3,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,2,4))) {
    PC1 <- A_PCANN[1,]
    PC2 <- A_PCANN[2,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,3))) {
    PC1 <- A_PCANN[1,]
    PC3 <- A_PCANN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC1,PC3)
  }
  if (identical(PCSelection_Frwrd_mANN, c(1,4))) {
    PC1 <- A_PCANN[1,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(2))) {
    PC2 <- A_PCANN[2,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
    }
    x <- as.matrix(PC2)
  }
  if (identical(PCSelection_Frwrd_mANN, c(2,3))) {
    PC2 <- A_PCANN[2,]
    PC3 <- A_PCANN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_mANN, c(2,4))) {
    PC2 <- A_PCANN[2,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(2,3,4))) {
    PC2 <- A_PCANN[2,]
    PC3 <- A_PCANN[3,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(3))) {
    PC3 <- A_PCANN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- as.matrix(PC3)
  }
  if (identical(PCSelection_Frwrd_mANN, c(3,4))) {
    PC3 <- A_PCANN[3,]
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_mANN, c(4))) {
    PC4 <- A_PCANN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (mANN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- as.matrix(PC4)
  }
  
  # Run PCANN and obtain prediction intervals
  
  frwrd_PCANN <- monmlp.predict(x,PCANNmodel)
  
  frwrd_PCANN_BC <- BoxCox(frwrd_PCANN,lambda_prd_PCANN_LOOCV)
  y90_PCANN_BC <- frwrd_PCANN_BC + (-1.282 * LOOCV_RMSE_PCANNmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCANN_BC <- frwrd_PCANN_BC + (-0.524 * LOOCV_RMSE_PCANNmodel_BC)
  y30_PCANN_BC <- frwrd_PCANN_BC + (0.524 * LOOCV_RMSE_PCANNmodel_BC)
  y10_PCANN_BC <- frwrd_PCANN_BC + (1.282 * LOOCV_RMSE_PCANNmodel_BC)
  
  y90_PCANN_BCbased <- InvBoxCox(y90_PCANN_BC,lambda_prd_PCANN_LOOCV)   # inverse-transform results and return them to main program
  y70_PCANN_BCbased <- InvBoxCox(y70_PCANN_BC,lambda_prd_PCANN_LOOCV)
  y30_PCANN_BCbased <- InvBoxCox(y30_PCANN_BC,lambda_prd_PCANN_LOOCV)
  y10_PCANN_BCbased <- InvBoxCox(y10_PCANN_BC,lambda_prd_PCANN_LOOCV)
  
  rm(frwrd_PCANN_BC,y90_PCANN_BC,y70_PCANN_BC,y30_PCANN_BC,y10_PCANN_BC)
  
  
  ############################################################################################################################################################# 
  
  # RUN MONOTONE COMPOSITE QUANTILE REGRESSION NEURAL NETWORK FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on MCQRNN-specifc retained input variables and calibration-period eigenvectors
  
  A_PCMCQRNN <- t(E_PCMCQRNN)%*%datmat_PCMCQRNN
  
  # Select MCQRNN-specific retained PCA modes and create corresponding data frame
  
  if (identical(PCSelection_Frwrd_MCQRNN, c(1))) {
    PC1 <- A_PCMCQRNN[1,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
    }
    x <- as.matrix(PC1)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,2))) {
    PC1 <- A_PCMCQRNN[1,]
    PC2 <- A_PCMCQRNN[2,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
    }
    x <- cbind(PC1,PC2)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,2,3))) {
    PC1 <- A_PCMCQRNN[1,]
    PC2 <- A_PCMCQRNN[2,]
    PC3 <- A_PCMCQRNN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC1,PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,2,3,4))) {
    PC1 <- A_PCMCQRNN[1,]
    PC2 <- A_PCMCQRNN[2,]
    PC3 <- A_PCMCQRNN[3,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,3,4))) {
    PC1 <- A_PCMCQRNN[1,]
    PC3 <- A_PCMCQRNN[3,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,2,4))) {
    PC1 <- A_PCMCQRNN[1,]
    PC2 <- A_PCMCQRNN[2,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,3))) {
    PC1 <- A_PCMCQRNN[1,]
    PC3 <- A_PCMCQRNN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC1,PC3)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(1,4))) {
    PC1 <- A_PCMCQRNN[1,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC1_multiply == "Y") {PC1 <- -1 * PC1}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC1,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(2))) {
    PC2 <- A_PCMCQRNN[2,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
    }
    x <- as.matrix(PC2)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(2,3))) {
    PC2 <- A_PCMCQRNN[2,]
    PC3 <- A_PCMCQRNN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- cbind(PC2,PC3)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(2,4))) {
    PC2 <- A_PCMCQRNN[2,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC2,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(2,3,4))) {
    PC2 <- A_PCMCQRNN[2,]
    PC3 <- A_PCMCQRNN[3,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC2_multiply == "Y") {PC2 <- -1 * PC2}
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC2,PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(3))) {
    PC3 <- A_PCMCQRNN[3,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
    }
    x <- as.matrix(PC3)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(3,4))) {
    PC3 <- A_PCMCQRNN[3,]
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC3_multiply == "Y") {PC3 <- -1 * PC3}
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- cbind(PC3,PC4)
  }
  if (identical(PCSelection_Frwrd_MCQRNN, c(4))) {
    PC4 <- A_PCMCQRNN[4,]
    if (ANN_monotone_flag_Frwrd == "Y") {   
      if (MCQRNN_PC4_multiply == "Y") {PC4 <- -1 * PC4}
    }
    x <- as.matrix(PC4)
  }
  
  # Run MCQRNN and obtain prediction intervals
  
  prd_matrix_PCMCQRNN <- mcqrnn.predict(x, PCMCQRNNmodel)
  
  y90_PCMCQRNN <<- prd_matrix_PCMCQRNN[,1]
  y70_PCMCQRNN <<- prd_matrix_PCMCQRNN[,2]
  frwrd_PCMCQRNN <<- prd_matrix_PCMCQRNN[,3]
  y30_PCMCQRNN <<- prd_matrix_PCMCQRNN[,4]
  y10_PCMCQRNN <<- prd_matrix_PCMCQRNN[,5]
  
  
  ############################################################################################################################################################# 
  
  # RUN LINEAR QUANTILE REGRESSION FOR NEW SAMPLE 
  
  # Create PC scores for new sample based on QR-specifc retained input variables and calibration-period eigenvectors
  
  A_PCQR <- t(E_PCQR)%*%datmat_PCQR
  input_var_num <- length(PCSelection_Frwrd_QR)
  
  # Select QR-specific retained PCA modes, run QR, and obtain prediction intervals for one-variable case
  
  if (input_var_num == 1) {
    if (identical(PCSelection_Frwrd_QR, c(1))) {X1 <- A_PCQR[1,]}
    if (identical(PCSelection_Frwrd_QR, c(2))) {X1 <- A_PCQR[2,]}
    if (identical(PCSelection_Frwrd_QR, c(3))) {X1 <- A_PCQR[3,]}
    if (identical(PCSelection_Frwrd_QR, c(4))) {X1 <- A_PCQR[4,]}
    y90_PCQR <<- QRcoeffs[1,1]+QRcoeffs[2,1]*X1      
    y70_PCQR <<- QRcoeffs[1,2]+QRcoeffs[2,2]*X1
    frwrd_PCQR <<- QRcoeffs[1,3]+QRcoeffs[2,3]*X1
    y30_PCQR <<- QRcoeffs[1,4]+QRcoeffs[2,4]*X1
    y10_PCQR <<- QRcoeffs[1,5]+QRcoeffs[2,5]*X1
  }
  
  # Select QR-specific retained PCA modes, run QR, and obtain prediction intervals for two-variable case
  
  if (input_var_num == 2) {
    if (identical(PCSelection_Frwrd_QR, c(1,2))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[2,]
    }
    if (identical(PCSelection_Frwrd_QR, c(1,3))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[3,]
    }
    if (identical(PCSelection_Frwrd_QR, c(1,4))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[4,]
    }
    if (identical(PCSelection_Frwrd_QR, c(2,3))) {
      X1 <- A_PCQR[2,]
      X2 <- A_PCQR[3,]
    }
    if (identical(PCSelection_Frwrd_QR, c(2,4))) {
      X1 <- A_PCQR[2,]
      X2 <- A_PCQR[4,]
    }
    if (identical(PCSelection_Frwrd_QR, c(3,4))) {
      X1 <- A_PCQR[3,]
      X2 <- A_PCQR[4,]
    }
    y90_PCQR <<- QRcoeffs[1,1]+QRcoeffs[2,1]*X1+QRcoeffs[3,1]*X2  
    y70_PCQR <<- QRcoeffs[1,2]+QRcoeffs[2,2]*X1+QRcoeffs[3,2]*X2
    frwrd_PCQR <<- QRcoeffs[1,3]+QRcoeffs[2,3]*X1+QRcoeffs[3,3]*X2
    y30_PCQR <<- QRcoeffs[1,4]+QRcoeffs[2,4]*X1+QRcoeffs[3,4]*X2
    y10_PCQR <<- QRcoeffs[1,5]+QRcoeffs[2,5]*X1+QRcoeffs[3,5]*X2
  }
  
  # Select QR-specific retained PCA modes, run QR, and obtain prediction intervals for three-variable case
  
  if (input_var_num == 3) {
    if (identical(PCSelection_Frwrd_QR, c(1,2,3))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[2,]
      X3 <- A_PCQR[3,]
    }
    if (identical(PCSelection_Frwrd_QR, c(1,2,4))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[2,]
      X3 <- A_PCQR[4,]
    }
    if (identical(PCSelection_Frwrd_QR, c(1,3,4))) {
      X1 <- A_PCQR[1,]
      X2 <- A_PCQR[3,]
      X3 <- A_PCQR[4,]
    }
    if (identical(PCSelection_Frwrd_QR, c(2,3,4))) {
      X1 <- A_PCQR[2,]
      X2 <- A_PCQR[3,]
      X3 <- A_PCQR[4,]
    }
    y90_PCQR <<- QRcoeffs[1,1]+QRcoeffs[2,1]*X1+QRcoeffs[3,1]*X2+QRcoeffs[4,1]*X3
    y70_PCQR <<- QRcoeffs[1,2]+QRcoeffs[2,2]*X1+QRcoeffs[3,2]*X2+QRcoeffs[4,2]*X3
    frwrd_PCQR <<- QRcoeffs[1,3]+QRcoeffs[2,3]*X1+QRcoeffs[3,3]*X2+QRcoeffs[4,3]*X3
    y30_PCQR <<- QRcoeffs[1,4]+QRcoeffs[2,4]*X1+QRcoeffs[3,4]*X2+QRcoeffs[4,4]*X3
    y10_PCQR <<- QRcoeffs[1,5]+QRcoeffs[2,5]*X1+QRcoeffs[3,5]*X2+QRcoeffs[4,5]*X3
  }
  
  # Select QR-specific retained PCA modes, run QR, and obtain prediction intervals for four-variable case
  
  if (input_var_num == 4) {
    X1 <- A_PCQR[1,]
    X2 <- A_PCQR[2,]
    X3 <- A_PCQR[3,]
    X4 <- A_PCQR[4,]
    y90_PCQR <<- QRcoeffs[1,1]+QRcoeffs[2,1]*X1+QRcoeffs[3,1]*X2+QRcoeffs[4,1]*X3+QRcoeffs[5,1]*X4
    y70_PCQR <<- QRcoeffs[1,2]+QRcoeffs[2,2]*X1+QRcoeffs[3,2]*X2+QRcoeffs[4,2]*X3+QRcoeffs[5,2]*X4
    frwrd_PCQR <<- QRcoeffs[1,3]+QRcoeffs[2,3]*X1+QRcoeffs[3,3]*X2+QRcoeffs[4,3]*X3+QRcoeffs[5,3]*X4
    y30_PCQR <<- QRcoeffs[1,4]+QRcoeffs[2,4]*X1+QRcoeffs[3,4]*X2+QRcoeffs[4,4]*X3+QRcoeffs[5,4]*X4
    y10_PCQR <<- QRcoeffs[1,5]+QRcoeffs[2,5]*X1+QRcoeffs[3,5]*X2+QRcoeffs[4,5]*X3+QRcoeffs[5,5]*X4
  }
  
  
  ############################################################################################################################################################# 
  
  # CREATE ENSEMBLE MEAN FORECAST DISTRIBUTION, IF ASKED
  
  if (Ensemble_flag_frwrd == "Y") {
    
    if (Ensemble_type_frwrd == "ALL") {
      y10_ensemble <- rowMeans(cbind(y10_PCR_BCbased,y10_PCQR,y10_PCANN_BCbased,y10_PCMCQRNN,y10_PCRF_BCbased,y10_PCSVM_BCbased))
      y30_ensemble <- rowMeans(cbind(y30_PCR_BCbased,y30_PCQR,y30_PCANN_BCbased,y30_PCMCQRNN,y30_PCRF_BCbased,y30_PCSVM_BCbased))
      frwrd_ensemble <- rowMeans(cbind(frwrd_PCR,frwrd_PCQR,frwrd_PCANN,frwrd_PCMCQRNN,frwrd_PCRF,frwrd_PCSVM))
      y70_ensemble <- rowMeans(cbind(y70_PCR_BCbased,y70_PCQR,y70_PCANN_BCbased,y70_PCMCQRNN,y70_PCRF_BCbased,y70_PCSVM_BCbased))
      y90_ensemble <- rowMeans(cbind(y90_PCR_BCbased,y90_PCQR,y90_PCANN_BCbased,y90_PCMCQRNN,y90_PCRF_BCbased,y90_PCSVM_BCbased))
    }
    
    if (Ensemble_type_frwrd == "AUTO") {
      dummyvar <- 0  # this allows custom external functions used for flexibly constructing ensembles in model-building phase to be recycled here
      prd_ensemble_LOOCV <- dummyvar
      Ymod_ensemble <- dummyvar
      obs <- dummyvar
      INITIALIZE_ENSEMBLE()
      if (LR_BC_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCR_BCbased,y70_PCR_BCbased,frwrd_PCR,dummyvar,y30_PCR_BCbased,y10_PCR_BCbased,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      if (QR_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCQR,y70_PCQR,frwrd_PCQR,dummyvar,y30_PCQR,y10_PCQR,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      if (MCQRNN_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCMCQRNN,y70_PCMCQRNN,frwrd_PCMCQRNN,dummyvar,y30_PCMCQRNN,y10_PCMCQRNN,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      if (RF_BC_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCRF_BCbased,y70_PCRF_BCbased,frwrd_PCRF,dummyvar,y30_PCRF_BCbased,y10_PCRF_BCbased,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      if (SVM_BC_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCSVM_BCbased,y70_PCSVM_BCbased,frwrd_PCSVM,dummyvar,y30_PCSVM_BCbased,y10_PCSVM_BCbased,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      if (ANN_BC_exclusion_flag == "No") {
        APPEND_ENSEMBLE(y90_PCANN_BCbased,y70_PCANN_BCbased,frwrd_PCANN,dummyvar,y30_PCANN_BCbased,y10_PCANN_BCbased,dummyvar,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles)
      }
      FINALIZE_ENSEMBLE()
      frwrd_ensemble <- prd_ensemble  # align notation with other variables in forward-run section of code, clean up workspace
      rm(prd_ensemble,prd_ensemble_LOOCV,dummyvar,Ymod_ensemble,obs,res_ensemble_LOOCV,res_ensemble)
    }
    
  }
  
  ############################################################################################################################################################# 
  
  # WRITE RESULTS
  
  if (Ensemble_flag_frwrd == "Y") {
    model_names <- c('PCR-BC','PCQR','PCANN-BC','PCMCQRNN','PCSVM-BC','PCRF-BC','ensemble')
    y90 <- c(y90_PCR_BCbased,y90_PCQR,y90_PCANN_BCbased,y90_PCMCQRNN,y90_PCSVM_BCbased,y90_PCRF_BCbased,y90_ensemble)
    y70 <- c(y70_PCR_BCbased,y70_PCQR,y70_PCANN_BCbased,y70_PCMCQRNN,y70_PCSVM_BCbased,y70_PCRF_BCbased,y70_ensemble)
    frwrd_prd <- c(frwrd_PCR,frwrd_PCQR,frwrd_PCANN,frwrd_PCMCQRNN,frwrd_PCSVM,frwrd_PCRF,frwrd_ensemble)
    y30 <- c(y30_PCR_BCbased,y30_PCQR,y30_PCANN_BCbased,y30_PCMCQRNN,y30_PCSVM_BCbased,y30_PCRF_BCbased,y30_ensemble)
    y10 <- c(y10_PCR_BCbased,y10_PCQR,y10_PCANN_BCbased,y10_PCMCQRNN,y10_PCSVM_BCbased,y10_PCRF_BCbased,y10_ensemble)
  }
  if (Ensemble_flag_frwrd == "N") {
    model_names <- c('PCR-BC','PCQR','PCANN-BC','PCMCQRNN','PCSVM-BC','PCRF-BC')
    y90 <- c(y90_PCR_BCbased,y90_PCQR,y90_PCANN_BCbased,y90_PCMCQRNN,y90_PCSVM_BCbased,y90_PCRF_BCbased)
    y70 <- c(y70_PCR_BCbased,y70_PCQR,y70_PCANN_BCbased,y70_PCMCQRNN,y70_PCSVM_BCbased,y70_PCRF_BCbased)
    frwrd_prd <- c(frwrd_PCR,frwrd_PCQR,frwrd_PCANN,frwrd_PCMCQRNN,frwrd_PCSVM,frwrd_PCRF)
    y30 <- c(y30_PCR_BCbased,y30_PCQR,y30_PCANN_BCbased,y30_PCMCQRNN,y30_PCSVM_BCbased,y30_PCRF_BCbased)
    y10 <- c(y10_PCR_BCbased,y10_PCQR,y10_PCANN_BCbased,y10_PCMCQRNN,y10_PCSVM_BCbased,y10_PCRF_BCbased)
  }
  all_frwrd_models.output <- data.frame(model_names,y90,y70,frwrd_prd,y30,y10)
  write.csv(all_frwrd_models.output, file = "ForwardRunPredictions.csv")
  
  
###############################################################################################################################################################
  
# CLOSE OUT FORWARD RUN  
  
}


  
###############################################################################################################################################################
####   WRAP UP   ##############################################################################################################################################
###############################################################################################################################################################


if (errorlog_flag == "Y") {
  sink(type="message")
  close(logfile)           
}

end_time <- Sys.time()
end_time - start_time


###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################

