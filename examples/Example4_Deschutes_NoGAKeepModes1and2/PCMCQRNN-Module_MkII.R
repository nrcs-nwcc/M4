# R FUNCTION TO PERFORM MONOTONE COMPOSITE QUANTILE REGRESSION NEURAL NETWORK MODELING ON PC(S) AND OBS  


PCMCQRNN <- function(PCSelection,MCQRNN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores) {
  
  
  ###############################################################################################################
  
  # CALL LIBRARIES
  
  library(qrnn)                 # R package for neural network-based nonlinear quantile regression modeling with optional monotonicity and non-negativity constraints
  library(foreach)              # parallelize "for" loop used for LOOCV
  library(doParallel)           # required to set up de facto cluster/use foreach
  
  
  ###############################################################################################################
  
  # FIT MAIN MODEL
  
  # convert vectors to matrices as required by mcqrnn: target variable
  y <- as.matrix(obs)
  
  # set vector of forecast distribution quantiles to be modeled
  qs <- seq(0.1,0.9,0.2)
  
  # look-up table to guide creation of data matrix containing selected PCA modes
  input_var_num <- length(PCSelection)
  if (identical(PCSelection, c(1))) {x <- as.matrix(PC1)}
  if (identical(PCSelection, c(1,2))) {x <- cbind(PC1,PC2)}
  if (identical(PCSelection, c(1,2,3))) {x <- cbind(PC1,PC2,PC3)}
  if (identical(PCSelection, c(1,2,3,4))) {x <- cbind(PC1,PC2,PC3,PC4)}
  if (identical(PCSelection, c(1,3,4))) {x <- cbind(PC1,PC3,PC4)}
  if (identical(PCSelection, c(1,2,4))) {x <- cbind(PC1,PC2,PC4)}
  if (identical(PCSelection, c(1,3))) {x <- cbind(PC1,PC3)}
  if (identical(PCSelection, c(1,4))) {x <- cbind(PC1,PC4)}
  if (identical(PCSelection, c(2))) {x <- as.matrix(PC2)}
  if (identical(PCSelection, c(2,3))) {x <- cbind(PC2,PC3)}
  if (identical(PCSelection, c(2,4))) {x <- cbind(PC2,PC4)}
  if (identical(PCSelection, c(2,3,4))) {x <- cbind(PC2,PC3,PC4)}
  if (identical(PCSelection, c(3))) {x <- as.matrix(PC3)}
  if (identical(PCSelection, c(3,4))) {x <- cbind(PC3,PC4)}
  if (identical(PCSelection, c(4))) {x <- as.matrix(PC4)}
  
  # if flag switched on, set problem up to ensure montonic relationships between each predictor and the predictand; MCQRNN monotonicity constraint only handles positive relationships - if negatively correlated, reverse sign
  if (ANN_monotone_flag == "Y") {
    MCQRNN_PC1_multiply <<- "N"
    MCQRNN_PC2_multiply <<- "N"
    MCQRNN_PC3_multiply <<- "N"
    MCQRNN_PC4_multiply <<- "N"
    if (input_var_num == 1) {
      monotone_select <- 1
      if (cor(x,y) < 0) {
        x <- -1*x
        MCQRNN_PC1_multiply <<- "Y"     # Need to track and save this for forward runs
      }         
    }
    if (input_var_num == 2) {
      monotone_select <- cbind(1,2)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        MCQRNN_PC1_multiply <<- "Y"
      }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]
        MCQRNN_PC2_multiply <<- "Y"
      }
    }
    if (input_var_num == 3) {
      monotone_select <- cbind(1,2,3)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        MCQRNN_PC1_multiply <<- "Y"
      }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]
        MCQRNN_PC2_multiply <<- "Y"
      }
      if (cor(x[,3],y) < 0) {
        x[,3] <- -1*x[,3]
        MCQRNN_PC3_multiply <<- "Y"
      }
    }
    if (input_var_num == 4) {
      monotone_select <- cbind(1,2,3,4)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        MCQRNN_PC1_multiply <<- "Y"
      }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]
        MCQRNN_PC2_multiply <<- "Y"
      }
      if (cor(x[,3],y) < 0) {
        x[,3] <- -1*x[,3]
        MCQRNN_PC3_multiply <<- "Y"
      }
      if (cor(x[,4],y) < 0) {
        x[,4] <- -1*x[,4]
        MCQRNN_PC4_multiply <<- "Y"
      }
    }
  } else {
    monotone_select <- NULL
  }
  
    # fit MCQRNN in accordance with configuration flag, monotonicity flag, non-negativity flag, and number of PCs
  if (MCQRNN_config_selection == 1) {
    hidden_neuron_num <<- 1
    PCMCQRNNmodel <<- mcqrnn.fit(x, y, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, n.trials=1, penalty=0.09, iter.max=50) # default: one hidden-layer neuron
  }
  if (MCQRNN_config_selection == 2) {
    hidden_neuron_num <<- 2
    PCMCQRNNmodel <<- mcqrnn.fit(x, y, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, n.trials=1, penalty=0.09, iter.max=50) # alternate: two hidden-layer neurons, 10 weight initialization restarts to avoid local minima
  }
  if (MCQRNN_config_selection == 3) {
    hidden_neuron_num <<- 2
    PCMCQRNNmodel <<- mcqrnn.fit(x, y, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, n.trials=1, penalty=0.09, iter.max=50) # user-specified custom configuration hard-wired here and invoked only if MCQRNN_config_selection is 3
  }
  
  # generate predicted values, corresponding in-sample residuals, and prediction bounds; save model object; clean up workspace  
  prd_matrix_PCMCQRNN <<- mcqrnn.predict(x, PCMCQRNNmodel)
  y90_PCMCQRNN <<- prd_matrix_PCMCQRNN[,1]
  y70_PCMCQRNN <<- prd_matrix_PCMCQRNN[,2]
  prd_PCMCQRNN <<- prd_matrix_PCMCQRNN[,3]
  y30_PCMCQRNN <<- prd_matrix_PCMCQRNN[,4]
  y10_PCMCQRNN <<- prd_matrix_PCMCQRNN[,5]
  res_PCMCQRNN <<- prd_PCMCQRNN - obs
  save(PCMCQRNNmodel,file = "PCMCQRNNmodel.Rdata")
  
  # store original predictor and predictand matrices for subsequent use in some probabilistic calculations after cross-validation is completed:
  x_full <- x
  y_full <- y
  
  
  ##############################################################################################################
  
  # PERFORM CROSS-VALIDATION (WITHOUT PARALLELIZATION)

  if (ANN_parallel_flag == "N") {
    
    sq_err <<- 0
    prd_PCMCQRNN_LOOCV <<- numeric(N)
   
    for (t in 1:N) {
      
      #create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
      y_subset <- as.matrix(y[-t])
      if (input_var_num == 1) {
        x_subset <- as.matrix(x[-t])
      } else {
         x_subset <- x[-t,]
      }
      
      # re-fit model to subsetted data
      if (MCQRNN_config_selection == 1) {
        hidden_neuron_num <- 1
        PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1, iter.max=50) # default: one hidden-layer neuron
        }
      if (MCQRNN_config_selection == 2) {
        hidden_neuron_num <- 2
        PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # alternate: two hidden-layer neurons, 10 weight initialization restarts to avoid local minima
      }
      if (MCQRNN_config_selection == 3) {
        hidden_neuron_num <- 2
        PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # user-specified custom configuration hard-wired here
      }
      
      # find submodel-predicted value for the data pair that was left out during the submodel's construction
      if (input_var_num == 1) {
        x_subset_test <- as.matrix(x[t])
      } else {
        x_subset_test <- t(x[t,])    # quirk - in this particular case, need to take transpose of x[t,] to get dimensions to line up right for mcqrnn
      }
      prd_matrix_PCMCQRNN_LOOCV <- mcqrnn.predict(x_subset_test, PCMCQRNNmodel_subset)  # find model-predicted value for left-out data pair
      prd_PCMCQRNN_LOOCV[t] <<- prd_matrix_PCMCQRNN_LOOCV[,3]
      
      # find sub-model square error and tidy up workspace
      sq_err[t] <<- (prd_PCMCQRNN_LOOCV[t] - obs[t])^2
      rm(x_subset,x_subset_test,y_subset)
      
    }
    
  }
      
    
  ##############################################################################################################
    
  # PERFORM CROSS-VALIDATION (WITH PARALLELIZATION)
    
  if (ANN_parallel_flag == "Y") {   
    
    #set up parallel backend to use num_core processors
    cl<-makeCluster(num_cores)
    registerDoParallel(cl)
      
    # start foreach loop
    NN_ParaCrossVal <- foreach(t=1:N, .combine=cbind, .inorder = TRUE, .packages = "qrnn") %dopar% {
        
      # create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
      y_subset <- as.matrix(y[-t])
      if (input_var_num == 1) {
        x_subset <- as.matrix(x[-t])
      } else {
        x_subset <- x[-t,]
      }
        
      # re-fit model to subsetted data
      if (MCQRNN_config_selection == 1) {
        hidden_neuron_num <- 1
        PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1, iter.max=50) # default: one hidden-layer neuron
      }
      if (MCQRNN_config_selection == 2) {
         hidden_neuron_num <- 2
         PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # alternate: two hidden-layer neurons, 10 weight initialization restarts to avoid local minima
      }
      if (MCQRNN_config_selection == 3) {
        hidden_neuron_num <- 2
        PCMCQRNNmodel_subset <- mcqrnn.fit(x_subset, y_subset, tau=qs, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # user-specified custom configuration hard-wired here
      }
        
      # find submodel-predicted value *AND ERROR* for the data pair that was left out during the submodel's construction
      if (input_var_num == 1) {
        x_subset_test <- as.matrix(x[t])
      } else {
        x_subset_test <- t(x[t,])    # quirk - in this particular case, need to take transpose of x[t,] to get dimensions to line up right for mcqrnn
      }
      prd_matrix_PCMCQRNN_LOOCV <- mcqrnn.predict(x_subset_test, PCMCQRNNmodel_subset)  # find model-predicted value for left-out data pair
      prd_PCMCQRNN_LOOCV_parallel <- prd_matrix_PCMCQRNN_LOOCV[,3]
      
      # find sub-model square error and tidy up workspace
      sq_err_parallel <- (prd_PCMCQRNN_LOOCV_parallel - y[t])^2
      rm(x_subset,x_subset_test,y_subset)
      
      # result that will appear in matrix NN_ParaCrossVal:
      output <- c(prd_PCMCQRNN_LOOCV_parallel,sq_err_parallel)
        
    }
      
    stopCluster(cl)
    
    prd_PCMCQRNN_LOOCV <<- NN_ParaCrossVal[1,]
    sq_err <<- NN_ParaCrossVal[2,]
      
  }
  
  
  ##############################################################################################################
  
  # FIND LOOCV RESIDUALS, PERFORMANCE METRICS FOR MODEL SELECTION
  
  # find RMSE, residuals, R^2, and (for an ANN, approximate) AIC
  sum_sq_err <- sum(sq_err)
  LOOCV_RMSE_PCMCQRNNmodel <<- sqrt(sum_sq_err/N)
  LOOCV_Rsqrd_PCMCQRNNmodel <<- (cor(obs,prd_PCMCQRNN_LOOCV))^2
  param_num_PCMCQRNN <<- input_var_num * hidden_neuron_num + hidden_neuron_num + hidden_neuron_num * 1 + 1  # number of parameters in an ANN with one hidden layer
  AIC_MCQRNN <<- N*log(sum_sq_err/N) + 2*(param_num_PCMCQRNN+1) + (2*(param_num_PCMCQRNN+1)*(param_num_PCMCQRNN+1+1))/(N-(param_num_PCMCQRNN+1)-1)
  res_PCMCQRNN_LOOCV <<- prd_PCMCQRNN_LOOCV - obs 
  
  x <- x_full
  y <- y_full
  qs_forRPSS <- seq(0.0,1.0,0.05)  # build set of 21 QR models solely to approximate CDF for RPSS calculation
  
  if (MCQRNN_config_selection == 1) {
    hidden_neuron_num <- 1
    PCMCQRNNmodel_forRPSS <- mcqrnn.fit(x, y, tau=qs_forRPSS, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # default: one hidden-layer neuron
  }
  if (MCQRNN_config_selection == 2) {
    hidden_neuron_num <- 2
    PCMCQRNNmodel_forRPSS <- mcqrnn.fit(x, y, tau=qs_forRPSS, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # alternate: two hidden-layer neurons, 10 weight initialization restarts to avoid local minima
  }
  if (MCQRNN_config_selection == 3) {
    hidden_neuron_num <- 2
    PCMCQRNNmodel_forRPSS <- mcqrnn.fit(x, y, tau=qs_forRPSS, n.hidden=hidden_neuron_num, n.hidden2=NULL, monotone=monotone_select, lower=0, penalty=0.1, n.trials=1,iter.max=50) # user-specified custom configuration hard-wired here and invoked only if MCQRNN_config_selection is 3
  }
  FlowQuantiles <- mcqrnn.predict(x, PCMCQRNNmodel_forRPSS)
  
  Ymod_PCMCQRNN <<- matrix(0,N,3)
  for (t in 1:N) {
    PCMCQRNN_CDF <- ecdf(FlowQuantiles[t,1:length(qs_forRPSS)])  # create empirical CDF of flow for time, t
    for (m in 1:3) {
      Ymod_PCMCQRNN[t,m] <<- PCMCQRNN_CDF(Qcrit[m])  # sample the empirical CDF to obtain P(flow < Qcrit[m])
    }
  }
  
  
# CLOSE OUT FUNCTION

}
  
