# R FUNCTION TO PERFORM SUPPORT VECTOR MACHINE MODELING ON PC(S) AND OBS  


PCSVM <- function(PCSelection,SVM_config_selection,fixedgamma) {
  
  
  ###############################################################################################################
  
  # CALL LIBRARIES

  library(e1071)                # R package to perform SVM modeling
  library(forecast)             # contains functions for Box-Cox transform
  
  
  ###############################################################################################################
  
  # FIT MAIN MODEL
  
  # general preparation - look-up table to guide creation of data frame containing selected PCA modes
  
  if (identical(PCSelection, c(1))) {dat_model <- data.frame(obs,PC1)}
  if (identical(PCSelection, c(1,2))) {dat_model <- data.frame(obs,PC1,PC2)}
  if (identical(PCSelection, c(1,2,3))) {dat_model <- data.frame(obs,PC1,PC2,PC3)}
  if (identical(PCSelection, c(1,2,3,4))) {dat_model <- data.frame(obs,PC1,PC2,PC3,PC4)}
  if (identical(PCSelection, c(1,3,4))) {dat_model <- data.frame(obs,PC1,PC3,PC4)}
  if (identical(PCSelection, c(1,2,4))) {dat_model <- data.frame(obs,PC1,PC2,PC4)}
  if (identical(PCSelection, c(1,3))) {dat_model <- data.frame(obs,PC1,PC3)}
  if (identical(PCSelection, c(1,4))) {dat_model <- data.frame(obs,PC1,PC4)}
  if (identical(PCSelection, c(2))) {dat_model <- data.frame(obs,PC2)}
  if (identical(PCSelection, c(2,3))) {dat_model <- data.frame(obs,PC2,PC3)}
  if (identical(PCSelection, c(2,4))) {dat_model <- data.frame(obs,PC2,PC4)}
  if (identical(PCSelection, c(2,3,4))) {dat_model <- data.frame(obs,PC2,PC3,PC4)}
  if (identical(PCSelection, c(3))) {dat_model <- data.frame(obs,PC3)}
  if (identical(PCSelection, c(3,4))) {dat_model <- data.frame(obs,PC3,PC4)}
  if (identical(PCSelection, c(4))) {dat_model <- data.frame(obs,PC4)}
  
  # build support vector machine model of obs on all variables in data frame
  
  if (SVM_config_selection == 1) {            # perform grid search across specified ranges of gamma, epsilon, and C to find optimal SVM hyperparameter values & associated model
    
    gammalist <- c(0.0005,0.002,0.004,0.10,0.20,0.50)
    epsilonlist <- seq(0,1,0.1)
    costlist = 2^(2:6)
    tuneResult <- tune(svm, obs ~ .,  data = dat_model, kernel="radial", ranges = list(gamma = gammalist, epsilon = epsilonlist, cost = costlist))    
    print(tuneResult)
    PCSVMmodel <<- tuneResult$best.model
    optimalgamma <<- PCSVMmodel$gamma
    optimalepsilon <<- PCSVMmodel$epsilon
    optimalC <<- PCSVMmodel$cost
    
  }
  
  if (SVM_config_selection == 2) {            # keep gamma fixed at value specified in main/input file but perform grid search across specified ranges of epsilon and C to find optimal SVM hyperparameter values & associated model
    
    epsilonlist <- seq(0,1,0.1)
    costlist = 2^(2:6)
    tuneResult <- tune(svm, obs ~ .,  data = dat_model, kernel="radial", gamma=fixedgamma, ranges = list(epsilon = epsilonlist, cost = costlist))    
    print(tuneResult)
    PCSVMmodel <<- tuneResult$best.model
    optimalgamma <<- fixedgamma
    optimalepsilon <<- PCSVMmodel$epsilon
    optimalC <<- PCSVMmodel$cost
    
  }
  
  if (SVM_config_selection == 3) {            # manually hard-wire values of all SVM hyperparameters
    
    optimalgamma <<- 0.0005
    optimalC <<-  256
    optimalepsilon <<- 0.45
    PCSVMmodel <<- svm(obs ~ ., data = dat_model, kernel="radial", epsilon=optimalepsilon, cost=optimalC, gamma=optimalgamma)
    
  }
  

  # generate predicted values and corresponding in-sample residuals, save model object  

  PCSVM_model_summary <<- summary(PCSVMmodel)
  plot(PCSVMmodel)
  prd_PCSVM <<- predict(PCSVMmodel)
  res_PCSVM <<- prd_PCSVM - obs
  save(PCSVMmodel,file = "PCSVMmodel.Rdata")
  
  
  ##############################################################################################################
  
  # PERFORM CROSS-VALIDATION
  
  sum_sq_err <- 0
  prd_PCSVM_LOOCV <<- numeric(N)
  for (t in 1:N) {
    
    # create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
    dat_model_subset <- dat_model[-t,]  
    
    # re-fit model to subsetted data
    PCSVMmodel_subset <- svm(obs ~ ., data = dat_model_subset, kernel="radial", epsilon=optimalepsilon, cost=optimalC, gamma=optimalgamma)   
    
    # find submodel-predicted value for the data pair that was left out during the submodel's construction
    dat_model_subset_test <- dat_model[t,]
    prd_PCSVM_LOOCV[t] <<- predict(PCSVMmodel_subset, dat_model_subset_test) # find model-predicted value for left-out data pair
    
    # find sub-model square error and tidy up workspace
    sum_sq_err <- sum_sq_err + (prd_PCSVM_LOOCV[t] - obs[t])^2 # calculate square error for left-out data pair and add to total
    rm(dat_model_subset,PCSVMmodel_subset,dat_model_subset_test) 
  
  }
  
  
  ##############################################################################################################
  
  # FIND LOOCV RESIDUALS, PERFORMANCE METRICS FOR MODEL SELECTION, AND PREDICTION BOUNDS
    
  # Find RMSE, residuals, and R^2
  LOOCV_RMSE_PCSVMmodel <<- sqrt(sum_sq_err/N)  
  res_PCSVM_LOOCV <<- prd_PCSVM_LOOCV - obs
  LOOCV_Rsqrd_PCSVMmodel <<- (cor(obs,prd_PCSVM_LOOCV))^2
  
  # Estimate prediction bounds (10, 30, 70, and 90% exceedance probablity flows) assuming normal dist centered at predicted value and sd = LOOCV SE as per NRCS practice
  
  y90_PCSVM <<- prd_PCSVM + (-1.282 * LOOCV_RMSE_PCSVMmodel)
  y70_PCSVM <<- prd_PCSVM + (-0.524 * LOOCV_RMSE_PCSVMmodel)
  y30_PCSVM <<- prd_PCSVM + (0.524 * LOOCV_RMSE_PCSVMmodel)
  y10_PCSVM <<- prd_PCSVM + (1.282 * LOOCV_RMSE_PCSVMmodel)
  
  # Similarly estimate prediction bounds, but in Box-Cox transform space to accomodate heteroscedastic and non-Gaussian residuals
  
  lambda_prd_PCSVM_LOOCV <<- BoxCox.lambda(prd_PCSVM_LOOCV,lower=0)     # find optimal lambda value; initial experimentation suggested that for the LOOCV predictions provides best results
  # lambda_prd_PCSVM_LOOCV <<- BoxCox.lambda(prd_PCSVM_LOOCV)
  
  obs_BC <- BoxCox(obs,lambda_prd_PCSVM_LOOCV)                       # perform forward Box-Cox transforms
  prd_PCSVM_LOOCV_BC <- BoxCox(prd_PCSVM_LOOCV,lambda_prd_PCSVM_LOOCV)
  prd_PCSVM_BC <- BoxCox(prd_PCSVM,lambda_prd_PCSVM_LOOCV)
  res_PCSVM_LOOCV_BC <- prd_PCSVM_LOOCV_BC - obs_BC                    # find residuals in Box-Cox transform space
  
  sum_sq_err_BC <- 0    # find Box-Cox transform-space RMSE
  for (t in 1:N) {
    sum_sq_err_BC <- sum_sq_err_BC + (res_PCSVM_LOOCV_BC[t])^2
  }
  LOOCV_RMSE_PCSVMmodel_BC <<- sqrt(sum_sq_err_BC/N)
  
  y90_PCSVM_BC <- prd_PCSVM_BC + (-1.282 * LOOCV_RMSE_PCSVMmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCSVM_BC <- prd_PCSVM_BC + (-0.524 * LOOCV_RMSE_PCSVMmodel_BC)
  y30_PCSVM_BC <- prd_PCSVM_BC + (0.524 * LOOCV_RMSE_PCSVMmodel_BC)
  y10_PCSVM_BC <- prd_PCSVM_BC + (1.282 * LOOCV_RMSE_PCSVMmodel_BC)
  
  y90_PCSVM_BCbased <<- InvBoxCox(y90_PCSVM_BC,lambda_prd_PCSVM_LOOCV)   # inverse-transform results and return them to main program
  y70_PCSVM_BCbased <<- InvBoxCox(y70_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  y30_PCSVM_BCbased <<- InvBoxCox(y30_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  y10_PCSVM_BCbased <<- InvBoxCox(y10_PCSVM_BC,lambda_prd_PCSVM_LOOCV)
  
  # Find model-derived LOOCV probability that flow lies within category m or within a lower category:
  
  Qcrit_BC <- numeric(3)
  Ymod_PCSVM <<- matrix(0,N,3)
  Ymod_PCSVM_BC <<- matrix(0,N,3)
  for (t in 1:N) {
    for (m in 1:3) {
      Ymod_PCSVM[t,m] <<- pnorm(Qcrit[m], mean = prd_PCSVM_LOOCV[t], sd = LOOCV_RMSE_PCSVMmodel)  # assuming homoscedastic normally distributed residuals
      Qcrit_BC[m] <- BoxCox(Qcrit[m],lambda_prd_PCSVM_LOOCV)
      Ymod_PCSVM_BC[t,m] <<- pnorm(Qcrit_BC[m], mean = prd_PCSVM_LOOCV_BC[t], sd = LOOCV_RMSE_PCSVMmodel_BC)  # assuming that in Box-Cox transform space the residuals are homoscedastic and normally distributed
    }
  }
  
  
  ##############################################################################################################
  
# CLOSE OUT FUNCTION

}
  
