# R FUNCTION TO PERFORM LINEAR REGRESSION MODELING ON PC(S) AND OBS  


PCR <- function(PCSelection) {
  
  
  # CALL LIBRARIES
  
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
  
  # build linear regression model of obs on all variables in data frame
  
  PCRmodel <<- lm(obs ~ ., data = dat_model)
  
  # generate predicted values and corresponding in-sample residuals, save model object  
  
  PCR_model_summary <<- summary(PCRmodel)
  prd_PCR <<- predict(PCRmodel)
  res_PCR <<- prd_PCR - obs
  save(PCRmodel,file = "PCRmodel.Rdata")
   
  
  # PERFORM CROSS-VALIDATION: FIND LOOCV TIME SERIES AND SUM OF SQUARED ERRORS
  
  sum_sq_err <- 0
  prd_PCR_LOOCV <<- numeric(N)
  for (t in 1:N) {
    
    # create dataset with data pair PC1(i), obs(i) missing (or triplet PC1(i), PC2(i), obs(i) missing, etc.)
    dat_model_subset <- dat_model[-t,]  
    
    # re-fit model to subsetted data
    PCRmodel_subset <- lm(obs ~ ., data = dat_model_subset)   
    
    # find submodel-predicted value for the data pair that was left out during the submodel's construction
    dat_model_subset_test <- dat_model[t,]
    prd_PCR_LOOCV[t] <<- predict(PCRmodel_subset, dat_model_subset_test) # find model-predicted value for left-out data pair
    
    # find sub-model square error and tidy up workspace
    sum_sq_err <- sum_sq_err + (prd_PCR_LOOCV[t] - obs[t])^2 # calculate square error for left-out data pair and add to total
    rm(dat_model_subset,PCRmodel_subset,dat_model_subset_test) 
  
  }

  
  # FIND LOOCV RESIDUALS, PERFORMANCE METRICS FOR MODEL SELECTION, AND PREDICTION BOUNDS
    
  # Find RMSE, residuals, and R^2
  LOOCV_RMSE_PCRmodel <<- sqrt(sum_sq_err/N)  
  res_PCR_LOOCV <<- prd_PCR_LOOCV - obs
  LOOCV_Rsqrd_PCRmodel <<- (cor(obs,prd_PCR_LOOCV))^2
  
  # Estimate prediction bounds (10, 30, 70, and 90% exceedance probablity flows) assuming normal dist centered at predicted value and sd = LOOCV SE as per NRCS practice
  
  y90_PCR <<- prd_PCR + (-1.282 * LOOCV_RMSE_PCRmodel)
  y70_PCR <<- prd_PCR + (-0.524 * LOOCV_RMSE_PCRmodel)
  y30_PCR <<- prd_PCR + (0.524 * LOOCV_RMSE_PCRmodel)
  y10_PCR <<- prd_PCR + (1.282 * LOOCV_RMSE_PCRmodel)
  
  # Similarly estimate prediction bounds, but in Box-Cox transform space to accomodate heteroscedastic and non-Gaussian residuals
  
  lambda_prd_PCR_LOOCV <<- BoxCox.lambda(prd_PCR_LOOCV,lower=0)     # find optimal lambda value; initial experimentation suggested that for the LOOCV predictions provides best results - needed here, also needs to be saved for forward runs
  # lambda_prd_PCR_LOOCV <<- BoxCox.lambda(prd_PCR_LOOCV)
  
  obs_BC <- BoxCox(obs,lambda_prd_PCR_LOOCV)                       # perform forward Box-Cox transforms
  prd_PCR_LOOCV_BC <- BoxCox(prd_PCR_LOOCV,lambda_prd_PCR_LOOCV)
  prd_PCR_BC <- BoxCox(prd_PCR,lambda_prd_PCR_LOOCV)
  res_PCR_LOOCV_BC <- prd_PCR_LOOCV_BC - obs_BC                    # find residuals in Box-Cox transform space
  
  sum_sq_err_BC <- 0    # find Box-Cox transform-space RMSE - needed here, also needs to be saved for forward runs
  for (t in 1:N) {
    sum_sq_err_BC <- sum_sq_err_BC + (res_PCR_LOOCV_BC[t])^2
  }
  LOOCV_RMSE_PCRmodel_BC <<- sqrt(sum_sq_err_BC/N)
  
  y90_PCR_BC <- prd_PCR_BC + (-1.282 * LOOCV_RMSE_PCRmodel_BC)   # find exceedance values in Box-Cox space
  y70_PCR_BC <- prd_PCR_BC + (-0.524 * LOOCV_RMSE_PCRmodel_BC)
  y30_PCR_BC <- prd_PCR_BC + (0.524 * LOOCV_RMSE_PCRmodel_BC)
  y10_PCR_BC <- prd_PCR_BC + (1.282 * LOOCV_RMSE_PCRmodel_BC)
  
  y90_PCR_BCbased <<- InvBoxCox(y90_PCR_BC,lambda_prd_PCR_LOOCV)   # inverse-transform results and return them to main program
  y70_PCR_BCbased <<- InvBoxCox(y70_PCR_BC,lambda_prd_PCR_LOOCV)
  y30_PCR_BCbased <<- InvBoxCox(y30_PCR_BC,lambda_prd_PCR_LOOCV)
  y10_PCR_BCbased <<- InvBoxCox(y10_PCR_BC,lambda_prd_PCR_LOOCV)
  
  # Find model-derived LOOCV probability that flow lies within category m or within a lower category:
  
  Qcrit_BC <- numeric(3)
  Ymod_PCR <<- matrix(0,N,3)
  Ymod_PCR_BC <<- matrix(0,N,3)
  for (t in 1:N) {
    for (m in 1:3) {
      Ymod_PCR[t,m] <<- pnorm(Qcrit[m], mean = prd_PCR_LOOCV[t], sd = LOOCV_RMSE_PCRmodel)  # assuming homoscedastic normally distributed residuals
      Qcrit_BC[m] <- BoxCox(Qcrit[m],lambda_prd_PCR_LOOCV)
      Ymod_PCR_BC[t,m] <<- pnorm(Qcrit_BC[m], mean = prd_PCR_LOOCV_BC[t], sd = LOOCV_RMSE_PCRmodel_BC)  # assuming that in Box-Cox transform space the residuals are homoscedastic and normally distributed
    }
  }
  
  
# CLOSE OUT FUNCTION

}
  
