# R FUNCTION TO PERFORM ARTIFICIAL NEURAL NETWORK MODELING ON PC(S) AND OBS  


PCANN <- function(PCSelection,mANN_config_selection,ANN_monotone_flag,ANN_parallel_flag,num_cores) {
  
  
  ###############################################################################################################
  
  # CALL LIBRARIES
  
  library(monmlp)               # R package for MLP modeling with optional monotonicity constraint
  library(forecast)             # contains functions for Box-Cox transform
  library(foreach)              # parallelize "for" loop used for LOOCV
  library(doParallel)           # required to set up de facto cluster/use foreach
  
  
  ###############################################################################################################
  
  # FIT MAIN MODEL
  
  # convert vectors to matrices as required by monmlp: target variable
  y <- as.matrix(obs)
  
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
  
  # if flag switched on, set problem up to ensure monotonic relationships between each predictor and the predictand; monmlp monotonicity constraint only handles positive relationships - if negatively correlated, reverse sign
  if (ANN_monotone_flag == "Y") {
    mANN_PC1_multiply <<- "N"
    mANN_PC2_multiply <<- "N"
    mANN_PC3_multiply <<- "N"
    mANN_PC4_multiply <<- "N"
    if (input_var_num == 1) {
      monotone_select <- 1
      if (cor(x,y) < 0) {
        x <- -1*x
        mANN_PC1_multiply <<- "Y"  # Need to track and save this for forward runs
      }         
    }
    if (input_var_num == 2) {
      monotone_select <- cbind(1,2)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        mANN_PC1_multiply <<- "Y"
      }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]
        mANN_PC2_multiply <<- "Y"
      }
    }
    if (input_var_num == 3) {
      monotone_select <- cbind(1,2,3)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        mANN_PC1_multiply <<- "Y"
        }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]        
        mANN_PC2_multiply <<- "Y"
      }
      if (cor(x[,3],y) < 0) {
        x[,3] <- -1*x[,3]
        mANN_PC3_multiply <<- "Y"
      }
    }
    if (input_var_num == 4) {
      monotone_select <- cbind(1,2,3,4)
      if (cor(x[,1],y) < 0) {
        x[,1] <- -1*x[,1]
        mANN_PC1_multiply <<- "Y"
      }
      if (cor(x[,2],y) < 0) {
        x[,2] <- -1*x[,2]
        mANN_PC2_multiply <<- "Y"
      }
      if (cor(x[,3],y) < 0) {
        x[,3] <- -1*x[,3]
        mANN_PC3_multiply <<- "Y"
      }
      if (cor(x[,4],y) < 0) {
        x[,4] <- -1*x[,4]
        mANN_PC4_multiply <<- "Y"
      }
    }
  } else {
    monotone_select <- NULL
  }
  
  # fit ANN in accordance with configuration flag, monotonicity flag, and number of PCs
  if (mANN_config_selection == 1) {
    hidden_neuron_num <- 1
    PCANNmodel <<- monmlp.fit(x, y, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=200, iter.stopped = 10) # default: one hidden-layer neuron, no bagging
  }
  if (mANN_config_selection == 2) {
    hidden_neuron_num <- 2
    PCANNmodel <<- monmlp.fit(x, y, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, n.ensemble=10, bag=TRUE, iter.stopped=10) # alternate: two hidden-layer neurons, bagging switched on with 10 bootstraps
  }
  if (mANN_config_selection == 3) {
    hidden_neuron_num <- 10
    PCANNmodel <<- monmlp.fit(x, y, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, iter.stopped=10) # user-specified custom configuration hard-wired here and invoked only if ANN_config_selection is 3
  }
  
  # generate predicted values and corresponding in-sample residuals, save model object, clean up workspace  
  prd_PCANN <<- monmlp.predict(x, PCANNmodel)
  res_PCANN <<- prd_PCANN - obs
  save(PCANNmodel,file = "PCANNmodel.Rdata")
  
  
  ##############################################################################################################
  
  # PERFORM CROSS-VALIDATION (WITHOUT PARALLELIZATION)

  if (ANN_parallel_flag == "N") {
    
    sq_err <<- 0
    prd_PCANN_LOOCV <<- numeric(N)
   
    for (t in 1:N) {
      
      #create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
      y_subset <- as.matrix(y[-t])
      if (input_var_num == 1) {
        x_subset <- as.matrix(x[-t])
      } else {
         x_subset <- x[-t,]
      }
      
      # re-fit model to subsetted data
      if (mANN_config_selection == 1) {
        hidden_neuron_num <- 1
        PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=200, iter.stopped = 10) # default: one hidden-layer neuron, no bagging
      }
      if (mANN_config_selection == 2) {
        hidden_neuron_num <- 2
        PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, n.ensemble=10, bag=TRUE, iter.stopped=10) # alternate: two hidden-layer neurons, bagging switched on with 20 bootstraps
      }
      if (mANN_config_selection == 3) {
        hidden_neuron_num <- 10
        PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, iter.stopped=10) # user-specified custom configuration hard-wired here
      }
      
      # find submodel-predicted value for the data pair that was left out during the submodel's construction
      if (input_var_num == 1) {
        x_subset_test <- as.matrix(x[t])
      } else {
        x_subset_test <- t(x[t,])    # quirk - in this particular case, need to take transpose of x[t,] to get dimensions to line up right for monmlp
      }
      prd_PCANN_LOOCV[t] <<- monmlp.predict(x_subset_test, PCANNmodel_subset)
      
      # find sub-model square error and tidy up workspace
      sq_err[t] <<- (prd_PCANN_LOOCV[t] - obs[t])^2
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
    NN_ParaCrossVal <- foreach(t=1:N, .combine=cbind, .inorder = TRUE, .packages = "monmlp") %dopar% {
        
      # create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
      y_subset <- as.matrix(y[-t])
      if (input_var_num == 1) {
        x_subset <- as.matrix(x[-t])
      } else {
        x_subset <- x[-t,]
      }
        
      # re-fit model to subsetted data
      if (mANN_config_selection == 1) {
        hidden_neuron_num <- 1
        PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=200, iter.stopped = 10) # default: one hidden-layer neuron, no bagging
      }
      if (mANN_config_selection == 2) {
         hidden_neuron_num <- 2
         PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, n.ensemble=10, bag=TRUE, iter.stopped=10) # alternate: two hidden-layer neurons, bagging switched on with 20 bootstraps
      }
      if (mANN_config_selection == 3) {
        hidden_neuron_num <- 10
        PCANNmodel_subset <- monmlp.fit(x_subset, y_subset, hidden1=hidden_neuron_num, monotone=monotone_select, iter.max=20, iter.stopped=10) # user-specified custom configuration hard-wired here
      }
        
      # find submodel-predicted value *AND ERROR* for the data pair that was left out during the submodel's construction
      if (input_var_num == 1) {
        x_subset_test <- as.matrix(x[t])
      } else {
        x_subset_test <- t(x[t,])    # quirk - in this particular case, need to take transpose of x[t,] to get dimensions to line up right for monmlp
      }
      prd_PCANN_LOOCV_parallel <- monmlp.predict(x_subset_test, PCANNmodel_subset)
        
      # find sub-model square error and tidy up workspace
      sq_err_parallel <- (prd_PCANN_LOOCV_parallel - y[t])^2
      rm(x_subset,x_subset_test,y_subset)
      
      # result that will appear in matrix NN_ParaCrossVal:
      output <- c(prd_PCANN_LOOCV_parallel,sq_err_parallel)
        
    }
      
    stopCluster(cl)
    
    prd_PCANN_LOOCV <<- NN_ParaCrossVal[1,]
    sq_err <<- NN_ParaCrossVal[2,]
      
  }
  
  
  ##############################################################################################################
  
  # FIND LOOCV RESIDUALS, PERFORMANCE METRICS FOR MODEL SELECTION, AND PREDICTION BOUNDS
  
  # find RMSE, residuals, R^2, and (for an ANN, approximate) AIC
  sum_sq_err <- sum(sq_err)
  LOOCV_RMSE_PCANNmodel <<- sqrt(sum_sq_err/N)
  LOOCV_Rsqrd_PCANNmodel <<- (cor(obs,prd_PCANN_LOOCV))^2
  param_num_PCANN <<- input_var_num * hidden_neuron_num + hidden_neuron_num + hidden_neuron_num * 1 + 1  # number of parameters in an ANN with one hidden layer
  AIC_ANN <<- N*log(sum_sq_err/N) + 2*(param_num_PCANN+1) + (2*(param_num_PCANN+1)*(param_num_PCANN+1+1))/(N-(param_num_PCANN+1)-1)
  res_PCANN_LOOCV <<- prd_PCANN_LOOCV - obs 
  
  # Estimate prediction bounds (10, 30, 70, and 90% exceedance probablity flows) assuming normal dist centered at predicted value and sd = LOOCV RMSE as per ~NRCS practice
  y90_PCANN <<- prd_PCANN + (-1.282 * LOOCV_RMSE_PCANNmodel)
  y70_PCANN <<- prd_PCANN + (-0.524 * LOOCV_RMSE_PCANNmodel)
  y30_PCANN <<- prd_PCANN + (0.524 * LOOCV_RMSE_PCANNmodel)
  y10_PCANN <<- prd_PCANN + (1.282 * LOOCV_RMSE_PCANNmodel)
  
  # Similarly estimate prediction bounds, but in Box-Cox transform space to accomodate heteroscedastic and non-Gaussian residuals
  lambda_prd_PCANN_LOOCV <<- BoxCox.lambda(prd_PCANN_LOOCV,lower=0)   # find optimal lambda value; initial experimentation suggested that for the LOOCV predictions provides best results - needed here, also needs to be saved for forward runs
  # lambda_prd_PCANN_LOOCV <<- BoxCox.lambda(prd_PCANN_LOOCV)
  obs_BC <- BoxCox(obs,lambda_prd_PCANN_LOOCV)                       # perform forward Box-Cox transforms
  prd_PCANN_LOOCV_BC <- BoxCox(prd_PCANN_LOOCV,lambda_prd_PCANN_LOOCV)
  prd_PCANN_BC <- BoxCox(prd_PCANN,lambda_prd_PCANN_LOOCV)
  res_PCANN_LOOCV_BC <- prd_PCANN_LOOCV_BC - obs_BC                  # find residuals in Box-Cox transform space
  sum_sq_err_BC <- 0    # find Box-Cox transform-space RMSE - needed here, also needs to be saved for forward runs
  for (t in 1:N) {
    sum_sq_err_BC <- sum_sq_err_BC + (res_PCANN_LOOCV_BC[t])^2
  }
  LOOCV_RMSE_PCANNmodel_BC <<- sqrt(sum_sq_err_BC/N)
  
  y90_PCANN_BC <- prd_PCANN_BC + (-1.282 * LOOCV_RMSE_PCANNmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCANN_BC <- prd_PCANN_BC + (-0.524 * LOOCV_RMSE_PCANNmodel_BC)
  y30_PCANN_BC <- prd_PCANN_BC + (0.524 * LOOCV_RMSE_PCANNmodel_BC)
  y10_PCANN_BC <- prd_PCANN_BC + (1.282 * LOOCV_RMSE_PCANNmodel_BC)
  
  y90_PCANN_BCbased <<- InvBoxCox(y90_PCANN_BC,lambda_prd_PCANN_LOOCV)   # inverse-transform results and return them to main program
  y70_PCANN_BCbased <<- InvBoxCox(y70_PCANN_BC,lambda_prd_PCANN_LOOCV)
  y30_PCANN_BCbased <<- InvBoxCox(y30_PCANN_BC,lambda_prd_PCANN_LOOCV)
  y10_PCANN_BCbased <<- InvBoxCox(y10_PCANN_BC,lambda_prd_PCANN_LOOCV)
  
  # Find model-derived LOOCV probability that flow lies within category m or within a lower category:
  Qcrit_BC <- numeric(3)
  Ymod_PCANN <<- matrix(0,N,3)
  Ymod_PCANN_BC <<- matrix(0,N,3)
  for (t in 1:N) {
    for (m in 1:3) {
      Ymod_PCANN[t,m] <<- pnorm(Qcrit[m], mean = prd_PCANN_LOOCV[t], sd = LOOCV_RMSE_PCANNmodel)  # (1) assuming homoscedastic normally distributed residuals
      Qcrit_BC[m] <- BoxCox(Qcrit[m],lambda_prd_PCANN_LOOCV)
      Ymod_PCANN_BC[t,m] <<- pnorm(Qcrit_BC[m], mean = prd_PCANN_LOOCV_BC[t], sd = LOOCV_RMSE_PCANNmodel_BC)  # (2) assuming that in Box-Cox transform space the residuals are homoscedastic and normally distributed
    }
  }
  
  
# CLOSE OUT FUNCTION

}
  
