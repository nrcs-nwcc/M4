# R FUNCTION TO PERFORM RANDOM FORESTS MODELING ON PC(S) AND OBS  


PCRF <- function(PCSelection) {
  
  
  # CALL LIBRARIES
  
  library(randomForest)       # R package for random forests
  library(forecast)           # contains functions for Box-Cox transform
  
  
  # FIT MODEL
  
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
  
  
  
  debug1 <<- dat_model
  
  
  
  # build random forests model of obs on all variables in data frame
  
  NumTrees <- 500
  PCRFmodel <<- randomForest(obs ~ ., data = dat_model, ntree = NumTrees)
  
  # generate predicted values and corresponding in-sample residuals, save model object  
  
  PCRF_model_summary <<- summary(PCRFmodel)
  plot(PCRFmodel)
  prd_PCRF <<- predict(PCRFmodel,dat_model)
  res_PCRF <<- prd_PCRF - obs
  save(PCRFmodel,file = "PCRFmodel.Rdata")
   
  
  
  debug2 <<- prd_PCRF
  
  
  
  # PERFORM CROSS-VALIDATION: FIND LOOCV TIME SERIES AND SUM OF SQUARED ERRORS
  
  sum_sq_err <- 0
  prd_PCRF_LOOCV <<- numeric(N)
  for (t in 1:N) {
    
    # create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
    dat_model_subset <- dat_model[-t,]  
    
    # re-fit model to subsetted data
    PCRFmodel_subset <- randomForest(obs ~ ., data = dat_model_subset, ntree = NumTrees)   
    
    # find submodel-predicted value for the data pair that was left out during the submodel's construction
    dat_model_subset_test <- dat_model[t,]
    prd_PCRF_LOOCV[t] <<- predict(PCRFmodel_subset, dat_model_subset_test) # find model-predicted value for left-out data pair
    
    # find sub-model square error and tidy up workspace
    sum_sq_err <- sum_sq_err + (prd_PCRF_LOOCV[t] - obs[t])^2 # calculate square error for left-out data pair and add to total
    rm(dat_model_subset,PCRFmodel_subset,dat_model_subset_test) 
  
  }

  
  # FIND LOOCV RESIDUALS, PERFORMANCE METRICS FOR MODEL SELECTION, AND PREDICTION BOUNDS
    
  # Find RMSE, residuals, and R^2
  LOOCV_RMSE_PCRFmodel <<- sqrt(sum_sq_err/N)  
  res_PCRF_LOOCV <<- prd_PCRF_LOOCV - obs
  LOOCV_Rsqrd_PCRFmodel <<- (cor(obs,prd_PCRF_LOOCV))^2
  
  # Estimate prediction bounds (10, 30, 70, and 90% exceedance probablity flows) assuming normal dist centered at predicted value and sd = LOOCV SE as per NRCS practice
  
  y90_PCRF <<- prd_PCRF + (-1.282 * LOOCV_RMSE_PCRFmodel)
  y70_PCRF <<- prd_PCRF + (-0.524 * LOOCV_RMSE_PCRFmodel)
  y30_PCRF <<- prd_PCRF + (0.524 * LOOCV_RMSE_PCRFmodel)
  y10_PCRF <<- prd_PCRF + (1.282 * LOOCV_RMSE_PCRFmodel)
  
  # Similarly estimate prediction bounds, but in Box-Cox transform space to accomodate heteroscedastic and non-Gaussian residuals
  
  lambda_prd_PCRF_LOOCV <<- BoxCox.lambda(prd_PCRF_LOOCV,lower=0)     # find optimal lambda value; initial experimentation suggested that for the LOOCV predictions provides best results
  obs_BC <- BoxCox(obs,lambda_prd_PCRF_LOOCV)                       # perform forward Box-Cox transforms
  prd_PCRF_LOOCV_BC <- BoxCox(prd_PCRF_LOOCV,lambda_prd_PCRF_LOOCV)
  prd_PCRF_BC <- BoxCox(prd_PCRF,lambda_prd_PCRF_LOOCV)
  res_PCRF_LOOCV_BC <- prd_PCRF_LOOCV_BC - obs_BC                    # find residuals in Box-Cox transform space
  
  sum_sq_err_BC <- 0    # find Box-Cox transform-space RMSE
  for (t in 1:N) {
    sum_sq_err_BC <- sum_sq_err_BC + (res_PCRF_LOOCV_BC[t])^2
  }
  LOOCV_RMSE_PCRFmodel_BC <<- sqrt(sum_sq_err_BC/N)
  
  y90_PCRF_BC <- prd_PCRF_BC + (-1.282 * LOOCV_RMSE_PCRFmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCRF_BC <- prd_PCRF_BC + (-0.524 * LOOCV_RMSE_PCRFmodel_BC)
  y30_PCRF_BC <- prd_PCRF_BC + (0.524 * LOOCV_RMSE_PCRFmodel_BC)
  y10_PCRF_BC <- prd_PCRF_BC + (1.282 * LOOCV_RMSE_PCRFmodel_BC)
  
  y90_PCRF_BCbased <<- InvBoxCox(y90_PCRF_BC,lambda_prd_PCRF_LOOCV)   # inverse-transform results and return them to main program
  y70_PCRF_BCbased <<- InvBoxCox(y70_PCRF_BC,lambda_prd_PCRF_LOOCV)
  y30_PCRF_BCbased <<- InvBoxCox(y30_PCRF_BC,lambda_prd_PCRF_LOOCV)
  y10_PCRF_BCbased <<- InvBoxCox(y10_PCRF_BC,lambda_prd_PCRF_LOOCV)
  
  # Find model-derived LOOCV probability that flow lies within category m or within a lower category:
  
  Qcrit_BC <- numeric(3)
  Ymod_PCRF <<- matrix(0,N,3)
  Ymod_PCRF_BC <<- matrix(0,N,3)
  for (t in 1:N) {
    for (m in 1:3) {
      Ymod_PCRF[t,m] <<- pnorm(Qcrit[m], mean = prd_PCRF_LOOCV[t], sd = LOOCV_RMSE_PCRFmodel)  # assuming homoscedastic normally distributed residuals
      Qcrit_BC[m] <- BoxCox(Qcrit[m],lambda_prd_PCRF_LOOCV)
      Ymod_PCRF_BC[t,m] <<- pnorm(Qcrit_BC[m], mean = prd_PCRF_LOOCV_BC[t], sd = LOOCV_RMSE_PCRFmodel_BC)  # assuming that in Box-Cox transform space the residuals are homoscedastic and normally distributed
    }
  }
  
  
# CLOSE OUT FUNCTION

}
  
