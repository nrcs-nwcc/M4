# R FUNCTION TO PERFORM QUANTILE REGRESSION MODELING ON PC(S) AND OBS  


PCQR <- function(PCSelection) {
  
  # CALL LIBRARIES
  
  library(quantreg)           # quantile regression
  library(quantregGrowth)     # quantile regression with quantile no-crossing constraint imposed; uses quantreg
  
  
  # TAKE CARE OF SOME PRELIMINARIES FOR QUANITLE REGRESSION
  
  qs <- seq(0.1,0.9,0.2)  # set desired quantiles equal to NRCS standard values
  
  # coding note: while quantreg follows typical R practices, quantregGrowth package has two quirks that require more hard-wired, tedious coding approaches relative to the other modeling modules in the prediction engine
  # a. It apparently does not accept arguments of the standard R form "model <- gcqr(y~.)" but instead requires explicit identification of each predictor in the data frame, "model <- gcqr(y~x1,x2 etc)"
  # b. In this package, the standard R "predict"-type functionality apparently works for polynomial regression but not linear regression so that predictQR(model,newdata) doesn't consistently work and regression equations must be explicitly framed in terms of individual regression coefficients extracted from the fitted model
  
  input_var_num <- length(PCSelection)
  
  
  # PERFORM MODELING AND OBTAIN PREDICTION BOUNDS (IF ONE-PC MODEL)
  
  if (input_var_num == 1) {
    
    # Create R dataframe
    
    if (identical(PCSelection, c(1))) {X1 <- PC1}
    if (identical(PCSelection, c(2))) {X1 <- PC2}
    if (identical(PCSelection, c(3))) {X1 <- PC3}
    if (identical(PCSelection, c(4))) {X1 <- PC4}
    dat_model <- data.frame(obs,X1)
    
    # Fit linear quantile regression model and obtain prediction bounds using a quantile no-crossing constraint
    
    PCQRmodel <<- gcrq(obs ~ X1, tau = qs, data = dat_model, cv = FALSE) 
    PCQR_model_summary <<- PCQRmodel$coef
    save(PCQRmodel,file="PCQRmodel.Rdata")
    write.table(PCQR_model_summary,file="PCQR_model_summary.txt",append = FALSE,row.names = TRUE, col.names = TRUE)
    y90_PCQR <<- PCQRmodel$coef[1]+PCQRmodel$coef[2]*X1      
    y70_PCQR <<- PCQRmodel$coef[3]+PCQRmodel$coef[4]*X1
    prd_PCQR <<- PCQRmodel$coef[5]+PCQRmodel$coef[6]*X1
    y30_PCQR <<- PCQRmodel$coef[7]+PCQRmodel$coef[8]*X1
    y10_PCQR <<- PCQRmodel$coef[9]+PCQRmodel$coef[10]*X1
    res_PCQR <<- prd_PCQR - obs
    
    # Estimate leave-one-out cross-validated predictions and residuals for subsequent use in diagnostics, again using regular quantreg package
    
    X1_subset <- numeric(N-1)
    obs_subset <- numeric(N-1)
    prd_PCQR_LOOCV <<- numeric(N)
    sum_sq_err <- 0
    for (t in 1:N) {
      
      X1_subset <- X1[-t]
      obs_subset <- obs[-t]
      dat_model_subset <- data.frame(X1_subset,obs_subset)                    # create dataset with data pair X1(t), obs(t) missing
      PCQRmodel_subset <- rq(obs_subset ~ X1_subset, data = dat_model_subset) # re-fit model to subsetted data
      
      X1_subset_test <- X1[t]
      prd_PCQR_LOOCV[t] <<- predict(PCQRmodel_subset, data.frame(X1_subset = X1_subset_test))   # find model-predicted value for left-out data pair
      sum_sq_err <- sum_sq_err + (prd_PCQR_LOOCV[t] - obs[t])^2                                 # calculate square error for left-out data pair and add to total
      
      rm(X1_subset,obs_subset,dat_model_subset,X1_subset_test) # tidy up work space
      
    }
    res_PCQR_LOOCV <<- prd_PCQR_LOOCV - obs
    LOOCV_RMSE_PCQRmodel <<- sqrt(sum_sq_err/N) 
    LOOCV_Rsqrd_PCQRmodel <<- (cor(obs,prd_PCQR_LOOCV))^2
    
    # Find model-derived probability that flow lies within category m or within a lower category:
    
    q_forRPSS <- seq(0.0,1.0,0.05)                                    # build set of 21 QR models solely to approximate CDF for subsequent RPSS calculation in diagnostics
    PCQR_forRPSS <- rq(obs ~ X1, data = dat_model, tau = q_forRPSS)   # use reqular quantreg so can robustly invoke standard R "predict" functionality later (very cumbersome otherwise, as would have to use explicit regression coefficient references for all 21 models); possibility for slight mismatch between error models for prediction bounds and RPSS calculation, but not best estimate as quantreg and quantregGrowth appear to yield the same median (best) estimate
    FlowQuantiles <- predict(PCQR_forRPSS)                            # obtain flow estimates for all 21 quantiles at all N times
    Ymod_PCQR <<- matrix(0,N,3)
    for (t in 1:N) {
      PCQR_CDF <- ecdf(FlowQuantiles[t,1:length(q_forRPSS)])          # create empirical CDF of flow for time, t
      for (m in 1:3) {
        Ymod_PCQR[t,m] <<- PCQR_CDF(Qcrit[m])                         # sample the empirical CDF to obtain P(flow < Qcrit[m])
      }
    }
    
  }
  
  
  # PERFORM MODELING AND OBTAIN PREDICTION BOUNDS (IF TWO-PC MODEL)
  
  if (input_var_num == 2) {
    
    # Create R dataframe
    
    if (identical(PCSelection, c(1,2))) {
      X1 <- PC1
      X2 <- PC2
    }
    if (identical(PCSelection, c(1,3))) {
      X1 <- PC1
      X2 <- PC3
    }
    if (identical(PCSelection, c(1,4))) {
      X1 <- PC1
      X2 <- PC4
    }
    if (identical(PCSelection, c(2,3))) {
      X1 <- PC2
      X2 <- PC3
    }
    if (identical(PCSelection, c(2,4))) {
      X1 <- PC2
      X2 <- PC4
    }
    if (identical(PCSelection, c(3,4))) {
      X1 <- PC3
      X2 <- PC4
    }
    dat_model <- data.frame(obs,X1,X2)

    # Fit linear quantile regression model and obtain prediction bounds using a quantile no-crossing constraint
    
    PCQRmodel <<- gcrq(obs ~ X1 + X2, tau = qs, data = dat_model, cv = FALSE) 
    PCQR_model_summary <<- PCQRmodel$coef
    save(PCQRmodel,file="PCQRmodel.Rdata")
    write.table(PCQR_model_summary,file="PCQR_model_summary.txt",append = FALSE,row.names = TRUE, col.names = TRUE)
    y90_PCQR <<- PCQRmodel$coef[1,1]+PCQRmodel$coef[2,1]*X1+PCQRmodel$coef[3,1]*X2  
    y70_PCQR <<- PCQRmodel$coef[1,2]+PCQRmodel$coef[2,2]*X1+PCQRmodel$coef[3,2]*X2
    prd_PCQR <<- PCQRmodel$coef[1,3]+PCQRmodel$coef[2,3]*X1+PCQRmodel$coef[3,3]*X2
    y30_PCQR <<- PCQRmodel$coef[1,4]+PCQRmodel$coef[2,4]*X1+PCQRmodel$coef[3,4]*X2
    y10_PCQR <<- PCQRmodel$coef[1,5]+PCQRmodel$coef[2,5]*X1+PCQRmodel$coef[3,5]*X2
    res_PCQR <<- prd_PCQR - obs
    
    # Estimate leave-one-out cross-validated predictions and residuals for subsequent use in diagnostics
    
    X1_subset <- numeric(N-1)
    X2_subset <- numeric(N-1)
    obs_subset <- numeric(N-1)
    prd_PCQR_LOOCV <<- numeric(N)
    sum_sq_err <- 0
    for (t in 1:N) {
      
      X1_subset <- X1[-t]
      X2_subset <- X2[-t]
      obs_subset <- obs[-t]
      dat_model_subset <- data.frame(X1_subset,X2_subset,obs_subset)                        # create dataset of length N-1, with data triplet X1(t), X2(t), obs(t) missing
      PCQRmodel_subset <- rq(obs_subset ~ X1_subset + X2_subset, data = dat_model_subset)   # re-fit model to subsetted data
      
      X1_subset_test <- X1[t]
      X2_subset_test <- X2[t]
      prd_PCQR_LOOCV[t] <<- predict(PCQRmodel_subset, data.frame(X1_subset = X1_subset_test, X2_subset = X2_subset_test)) # find model-predicted value for left-out data pair
      sum_sq_err <- sum_sq_err + (prd_PCQR_LOOCV[t] - obs[t])^2                                                           # calculate square error for left-out data pair and add to total
      
      rm(X1_subset,X2_subset,obs_subset,dat_model_subset,X1_subset_test,X2_subset_test)   # tidy up work space
      
    }
    res_PCQR_LOOCV <<- prd_PCQR_LOOCV - obs
    LOOCV_RMSE_PCQRmodel <<- sqrt(sum_sq_err/N) 
    LOOCV_Rsqrd_PCQRmodel <<- (cor(obs,prd_PCQR_LOOCV))^2
    
    # Find model-derived probability that flow lies within category m or within a lower category:
    
    q_forRPSS <- seq(0.0,1.0,0.05)                                        # build set of 21 QR models solely to approximate CDF for RPSS calculation
    PCQR_forRPSS <- rq(obs ~ X1 + X2, data = dat_model, tau = q_forRPSS)  # use reqular quantreg so can robustly invoke "predict" later or else cumbersome; possibility for slight mismatch between error models for prediction bounds and RPSS calculation
    FlowQuantiles <- predict(PCQR_forRPSS)                                # obtain flow estimates for all 21 quantiles at all N times
    Ymod_PCQR <<- matrix(0,N,3)
    for (t in 1:N) {
      PCQR_CDF <- ecdf(FlowQuantiles[t,1:length(q_forRPSS)])              # create empirical CDF of flow for time, t
      for (m in 1:3) {
        Ymod_PCQR[t,m] <<- PCQR_CDF(Qcrit[m])                             # sample the empirical CDF to obtain P(flow < Qcrit[m])
      }
    }
    
  }
  
  
  # PERFORM MODELING AND OBTAIN PREDICTION BOUNDS (IF THREE-PC MODEL)
  
  if (input_var_num == 3) {
    
    # Create R dataframe
    
    if (identical(PCSelection, c(1,2,3))) {
      X1 <- PC1
      X2 <- PC2
      X3 <- PC3
    }
    if (identical(PCSelection, c(1,2,4))) {
      X1 <- PC1
      X2 <- PC2
      X3 <- PC4
    }
    if (identical(PCSelection, c(1,3,4))) {
      X1 <- PC1
      X2 <- PC3
      X3 <- PC4
    }
    if (identical(PCSelection, c(2,3,4))) {
      X1 <- PC2
      X2 <- PC3
      X3 <- PC4
    }
    dat_model <- data.frame(obs,X1,X2,X3)
    
    # Fit linear quantile regression model and obtain prediction bounds using a quantile no-crossing constraint
    
    PCQRmodel <<- gcrq(obs ~ X1 + X2 + X3, tau = qs, data = dat_model, cv = FALSE) 
    PCQR_model_summary <<- PCQRmodel$coef
    save(PCQRmodel,file="PCQRmodel.Rdata")
    write.table(PCQR_model_summary,file="PCQR_model_summary.txt",append = FALSE,row.names = TRUE, col.names = TRUE)
    y90_PCQR <<- PCQRmodel$coef[1,1]+PCQRmodel$coef[2,1]*X1+PCQRmodel$coef[3,1]*X2+PCQRmodel$coef[4,1]*X3
    y70_PCQR <<- PCQRmodel$coef[1,2]+PCQRmodel$coef[2,2]*X1+PCQRmodel$coef[3,2]*X2+PCQRmodel$coef[4,2]*X3
    prd_PCQR <<- PCQRmodel$coef[1,3]+PCQRmodel$coef[2,3]*X1+PCQRmodel$coef[3,3]*X2+PCQRmodel$coef[4,3]*X3
    y30_PCQR <<- PCQRmodel$coef[1,4]+PCQRmodel$coef[2,4]*X1+PCQRmodel$coef[3,4]*X2+PCQRmodel$coef[4,4]*X3
    y10_PCQR <<- PCQRmodel$coef[1,5]+PCQRmodel$coef[2,5]*X1+PCQRmodel$coef[3,5]*X2+PCQRmodel$coef[4,5]*X3
    res_PCQR <<- prd_PCQR - obs
    
    # Estimate leave-one-out cross-validated predictions and residuals for subsequent use in diagnostics
    
    X1_subset <- numeric(N-1)
    X2_subset <- numeric(N-1)
    X3_subset <- numeric(N-1)
    obs_subset <- numeric(N-1)
    prd_PCQR_LOOCV <<- numeric(N)
    sum_sq_err <- 0
    for (t in 1:N) {
      
      X1_subset <- X1[-t]
      X2_subset <- X2[-t]
      X3_subset <- X3[-t]
      obs_subset <- obs[-t]
      dat_model_subset <- data.frame(X1_subset,X2_subset,X3_subset,obs_subset)                          # create dataset of length N-1 with data point [X1(t), X2(t), X3(t), obs(t)] missing
      PCQRmodel_subset <- rq(obs_subset ~ X1_subset + X2_subset + X3_subset, data = dat_model_subset)   # re-fit model to subsetted data
      
      X1_subset_test <- X1[t]
      X2_subset_test <- X2[t]
      X3_subset_test <- X3[t]
      prd_PCQR_LOOCV[t] <<- predict(PCQRmodel_subset, data.frame(X1_subset = X1_subset_test, X2_subset = X2_subset_test, X3_subset = X3_subset_test)) # find model-predicted value for left-out data pair
      sum_sq_err <- sum_sq_err + (prd_PCQR_LOOCV[t] - obs[t])^2   # calculate square error for left-out data pair and add to total
      
      rm(X1_subset,X2_subset,X3_subset,obs_subset,dat_model_subset,X1_subset_test,X2_subset_test,X3_subset_test)   # tidy up work space
      
    }
    res_PCQR_LOOCV <<- prd_PCQR_LOOCV - obs
    LOOCV_RMSE_PCQRmodel <<- sqrt(sum_sq_err/N) 
    LOOCV_Rsqrd_PCQRmodel <<- (cor(obs,prd_PCQR_LOOCV))^2
    
    # Find model-derived probability that flow lies within category m or within a lower category:
    
    q_forRPSS <- seq(0.0,1.0,0.05)                                            # build set of 21 QR models solely to approximate CDF for RPSS calculation
    PCQR_forRPSS <- rq(obs ~ X1 + X2 + X3, data = dat_model, tau = q_forRPSS) # use reqular quantreg so can robustly invoke "predict" later or else cumbersome; possibility for slight mismatch between error models for prediction bounds and RPSS calculation
    FlowQuantiles <- predict(PCQR_forRPSS)                                    # obtain flow estimates for all 21 quantiles at all N times
    Ymod_PCQR <<- matrix(0,N,3)
    for (t in 1:N) {
      PCQR_CDF <- ecdf(FlowQuantiles[t,1:length(q_forRPSS)])                  # create empirical CDF of flow for time, t
      for (m in 1:3) {
        Ymod_PCQR[t,m] <<- PCQR_CDF(Qcrit[m])                                 # sample the empirical CDF to obtain P(flow < Qcrit[m])
      }
    }
    
  }
  
  
  # PERFORM MODELING AND OBTAIN PREDICTION BOUNDS (IF FOUR-PC MODEL)
  
  if (input_var_num == 4) {
    
    # Create R dataframe
    
    X1 <- PC1
    X2 <- PC2
    X3 <- PC3
    X4 <- PC4
    dat_model <- data.frame(obs,X1,X2,X3,X4)
    
    # Fit linear quantile regression model and obtain prediction bounds using a quantile no-crossing constraint
    
    PCQRmodel <<- gcrq(obs ~ X1 + X2 + X3 + X4, tau = qs, data = dat_model, cv = FALSE) 
    PCQR_model_summary <<- PCQRmodel$coef
    save(PCQRmodel,file="PCQRmodel.Rdata")
    write.table(PCQR_model_summary,file="PCQR_model_summary.txt",append = FALSE,row.names = TRUE, col.names = TRUE)
    y90_PCQR <<- PCQRmodel$coef[1,1]+PCQRmodel$coef[2,1]*X1+PCQRmodel$coef[3,1]*X2+PCQRmodel$coef[4,1]*X3+PCQRmodel$coef[5,1]*X4
    y70_PCQR <<- PCQRmodel$coef[1,2]+PCQRmodel$coef[2,2]*X1+PCQRmodel$coef[3,2]*X2+PCQRmodel$coef[4,2]*X3+PCQRmodel$coef[5,2]*X4
    prd_PCQR <<- PCQRmodel$coef[1,3]+PCQRmodel$coef[2,3]*X1+PCQRmodel$coef[3,3]*X2+PCQRmodel$coef[4,3]*X3+PCQRmodel$coef[5,3]*X4
    y30_PCQR <<- PCQRmodel$coef[1,4]+PCQRmodel$coef[2,4]*X1+PCQRmodel$coef[3,4]*X2+PCQRmodel$coef[4,4]*X3+PCQRmodel$coef[5,4]*X4
    y10_PCQR <<- PCQRmodel$coef[1,5]+PCQRmodel$coef[2,5]*X1+PCQRmodel$coef[3,5]*X2+PCQRmodel$coef[4,5]*X3+PCQRmodel$coef[5,5]*X4
    res_PCQR <<- prd_PCQR - obs
    
    # Estimate leave-one-out cross-validated predictions and residuals for subsequent use in diagnostics
    
    X1_subset <- numeric(N-1)
    X2_subset <- numeric(N-1)
    X3_subset <- numeric(N-1)
    X4_subset <- numeric(N-1)
    obs_subset <- numeric(N-1)
    prd_PCQR_LOOCV <<- numeric(N)
    sum_sq_err <- 0
    for (t in 1:N) {
      
      X1_subset <- X1[-t]
      X2_subset <- X2[-t]
      X3_subset <- X3[-t]
      X4_subset <- X4[-t]
      obs_subset <- obs[-t]
      dat_model_subset <- data.frame(X1_subset,X2_subset,X3_subset,X4_subset,obs_subset)                            # create dataset of length N-1 with data point [X1(t), X2(t), X3(t), X4(t), obs(t)] missing
      PCQRmodel_subset <- rq(obs_subset ~ X1_subset + X2_subset + X3_subset + X4_subset, data = dat_model_subset)   # re-fit model to subsetted data
      
      X1_subset_test <- X1[t]
      X2_subset_test <- X2[t]
      X3_subset_test <- X3[t]
      X4_subset_test <- X4[t]
      prd_PCQR_LOOCV[t] <<- predict(PCQRmodel_subset, data.frame(X1_subset = X1_subset_test, X2_subset = X2_subset_test, X3_subset = X3_subset_test, X4_subset = X4_subset_test)) # find model-predicted value for left-out data pair
      sum_sq_err <- sum_sq_err + (prd_PCQR_LOOCV[t] - obs[t])^2   # calculate square error for left-out data pair and add to total
      
      rm(X1_subset,X2_subset,X3_subset,X4_subset,obs_subset,dat_model_subset,X1_subset_test,X2_subset_test,X3_subset_test,X4_subset_test)   # tidy up work space
      
    }
    res_PCQR_LOOCV <<- prd_PCQR_LOOCV - obs
    LOOCV_RMSE_PCQRmodel <<- sqrt(sum_sq_err/N) 
    LOOCV_Rsqrd_PCQRmodel <<- (cor(obs,prd_PCQR_LOOCV))^2
    
    # Find model-derived probability that flow lies within category m or within a lower category:
    
    q_forRPSS <- seq(0.0,1.0,0.05)                                                  # build set of 21 QR models solely to approximate CDF for RPSS calculation
    PCQR_forRPSS <- rq(obs ~ X1 + X2 + X3 + X4, data = dat_model, tau = q_forRPSS)  # use reqular quantreg so can robustly invoke "predict" later or else cumbersome; possibility for slight mismatch between error models for prediction bounds and RPSS calculation
    FlowQuantiles <- predict(PCQR_forRPSS)                                          # obtain flow estimates for all 21 quantiles at all N times
    Ymod_PCQR <<- matrix(0,N,3)
    for (t in 1:N) {
      PCQR_CDF <- ecdf(FlowQuantiles[t,1:length(q_forRPSS)])                        # create empirical CDF of flow for time, t
      for (m in 1:3) {
        Ymod_PCQR[t,m] <<- PCQR_CDF(Qcrit[m])                                       # sample the empirical CDF to obtain P(flow < Qcrit[m])
      }
    }
    
  }
  
  
  # CLOSE OUT FUNCTION
  
}

