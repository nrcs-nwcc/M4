# R FUNCTION TO PERFORM GENETIC ALGORITHM-BASED OPTIMAL SELECTION OF INPUT VARIABLES AND PCS TO RETAIN


PredictorOptimization <- function(model_flag) {
 
  
  # DECLARE LIBRARIES, CUSTOM FUNCTIONS IN EXTERNAL FILES
  
  library(genalg)
  library(stringr)
  source("PCA-Module_MkII.R")
  source("PCR-Module_MkII.R")
  source("PCQR-Module_MkII.R")
  source("PCANN-Module_MkII.R")
  source("PCRF-Module_MkII.R")
  source("PCMCQRNN-Module_MkII.R")
  source("PCSVM-Module_MkII.R")
  
  
  # DEFINE FUNCTION TO BE MINIMIZED
  
  # genalg will feed trial values of S, the binary predictor selection vector (chromosome), to this optimization function
  # The optimization function consists of min(model RMSE) but could be modified
  # The first Z genes in S turn on/off individual input variables in the candidate pool, the next MaxModes-1 genes turn on/off higher-than-first mode PCs 
  
  ObjFunc <- function(S) {
    
    # if ((sum(S[1:Z]) < 2)) {
    
    
    
    if ((sum(S[1:Z]) < MinNumVars)) {
      
      
    
      
      # Want at least two input variables, so penalize objective function if fewer
      return(2*max(obs)) 
      
    } else {
      
      # Generate a reduced candidate data matrix, datmatR, for this trial solution
      # datmatR includes only those rows of datmat corresponding to those elements of S[1:Z] = 1:
      Sdat <- S[1:Z]
      datmatR <- datmat[Sdat == 1, ]
      
      # Generate set of PCA modes to retain in this trial solution
      # Leading mode is always used
      # Up to 3 additional modes are switched on if the corresponding elements of S[(Z+1):(Z+MaxModes-1)] = 1
      # Need to then translate on/off switch positions (binary genes) for individual modes in trial binary chromsome to the numbers of the retained modes to use in modeling function
      PCswitchpositions <- S[(Z+1):(Z+MaxModes-1)]
      PCSelection <- integer(MaxModes)
      PCSelection[1] <- 1  
      if (MaxModes > 1) {if (PCswitchpositions[1] == 1) {PCSelection[2] <- 2}}
      if (MaxModes > 2) {if (PCswitchpositions[2] == 1) {PCSelection[3] <- 3}}
      if (MaxModes > 3) {if (PCswitchpositions[3] == 1) {PCSelection[4] <- 4}}
      PCSelection <- PCSelection[PCSelection != 0]    
      
      # Need at least as many input variables as retained modes in this trial solution, so penalize objective function if fewer
      if (sum(S[1:Z]) < max(PCSelection)) { return(2*max(obs)) } 
      
      # Feed trial reduced candidate data matrix to external function performing PCA:
      PCA(datmatR,PCSelection)
      
      # For this trial solution, feed selected PCs based on selected input variables to external modeling function:
      if (model_flag == "PCR") {
        PCR(PCSelection)
        RMSE_GA <- LOOCV_RMSE_PCRmodel
      }
      if (model_flag == "PCQR") {
        PCQR(PCSelection)
        RMSE_GA <- LOOCV_RMSE_PCQRmodel
      }
      if (model_flag == "PCRF") {
        PCRF(PCSelection)
        RMSE_GA <- LOOCV_RMSE_PCRFmodel
      }
      if (model_flag == "PCSVM") {
        PCSVM(PCSelection,SVM_config_selection,fixedgamma)
        RMSE_GA <- LOOCV_RMSE_PCSVMmodel
      }
      if (model_flag == "PCANN") {
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
        RMSE_GA <- LOOCV_RMSE_PCANNmodel
      }
      if (model_flag == "PCMCQRNN") {
        # if fitting MCQRNN using user-specified neural network configuration...
        if (AutoANNConfigFlag == "N") {PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)} 
        # if fitting MCQRNN using automated configuration selection based primarily on LOOCV RMSE and R^2 and secondarily on AIC...
        if (AutoANNConfigFlag == "Y") {
          MCQRNN_config_selection <- 1
          PCMCQRNN(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores)
          RMSE_benchmark <- mean(c(LOOCV_RMSE_PCRmodel,LOOCV_RMSE_PCQRmodel,LOOCV_RMSE_PCRFmodel,LOOCV_RMSE_PCSVMmodel))
          Rsqrd_benchmark <- mean(c(LOOCV_Rsqrd_PCRmodel,LOOCV_Rsqrd_PCQRmodel,LOOCV_Rsqrd_PCRFmodel,LOOCV_Rsqrd_PCSVMmodel))
          RMSE_deficit <- 100*(LOOCV_RMSE_PCMCQRNNmodel - RMSE_benchmark) / RMSE_benchmark
          Rsqrd_deficit <- abs(100*(LOOCV_Rsqrd_PCMCQRNNmodel - Rsqrd_benchmark) / Rsqrd_benchmark)
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
        RMSE_GA <- LOOCV_RMSE_PCMCQRNNmodel
      }
      
      # Track progress, printing to screen and file:
      tempchar1 <- paste(S, collapse=" ")
      if (model_flag == "PCR") {
        tempchar2 <- paste("principal components regression:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCR.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      } 
      if (model_flag == "PCQR") {
        tempchar2 <- paste("linear quantile regression:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCQR.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      } 
      if (model_flag == "PCRF") {
        tempchar2 <- paste("random forests:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCRF.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      }
      if (model_flag == "PCSVM") {
        tempchar2 <- paste("support vector machine:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCSVM.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      }
      if (model_flag == "PCANN") {
        tempchar2 <- paste("monotone artificial neural network:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCANN.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      }
      if (model_flag == "PCMCQRNN") {
        tempchar2 <- paste("monotone composite quantile regression neural network:    binary chromosome = ",tempchar1,",    trial cross-validated RMSE = ",round(RMSE_GA,digits = 3))
        tempchar3 <- capture.output(tempchar2, file="GA_RunLog_PCMCQRNN.txt", append=TRUE)
        if (DisplayFlag == "Y") { print(tempchar2) }
      }
      
      # Return corresponding SE value for this GA iteration:
      return(RMSE_GA)
      
    }
    
  }
  
  
  # CALL GENALG AND INSPECT RESULTS:
  
  print("Running genetic algorithm for optimal selection of input variables and PCA modes for current model...")
  GAmodel <<- rbga.bin(size = Z+MaxModes-1, popSize = GAPopSize, iters = GANumGens, mutationChance = 0.05, zeroToOneRatio=1,
                      elitism = T, evalFunc = ObjFunc)
  RunSummary <<- summary(GAmodel, echo = TRUE)
  plot.new()
  dev.new()
  plot(GAmodel)
   

  # CLOSE OUT FUNCTION
  
}