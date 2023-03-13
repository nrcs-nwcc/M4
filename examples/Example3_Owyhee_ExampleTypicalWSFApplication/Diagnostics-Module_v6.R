# R FUNCTION TO PERFORM DIAGNOSTICS ON MODELING OUTCOMES  


DIAGNOSTICS <- function(year,N,prd,res,prd_LOOCV,res_LOOCV,obs,y90,y70,y30,y10,plottitle,Qcrit,Ymod) {
  

  # EVALUATE SOME DETERMINISTIC FIT METRICS
  
  # RMSE, R^2, and NSE (in-sample)
  
  sum_sq_err <- 0
  sum_deviations <- 0
  for (t in 1:N) {
    sum_sq_err <- sum_sq_err + (res[t])^2
    sum_deviations <- sum_deviations + (obs[t]-mean(obs))^2
  }
  RMSE <- sqrt(sum_sq_err/N)
  NSE <- 1-(sum_sq_err/sum_deviations)
  Rsqrd <- (cor(obs,prd))^2
  rm(sum_sq_err,sum_deviations)
  
  # RMSE, R^2, and NSE (LOOCV)
  
  sum_sq_err <- 0
  sum_deviations <- 0
  for (t in 1:N) {
    sum_sq_err <- sum_sq_err + (res_LOOCV[t])^2
    sum_deviations <- sum_deviations + (obs[t]-mean(obs))^2
  }
  RMSE_LOOCV <- sqrt(sum_sq_err/N)
  NSE_LOOCV <- 1-(sum_sq_err/sum_deviations)
  Rsqrd_LOOCV <- (cor(obs,prd_LOOCV))^2
  rm(sum_sq_err,sum_deviations)
  
  
  # EVALUATE SOME CATEGORICAL AND PROBABILISTIC DIAGNOSTICS: RPSS & MISS/HIT RATES
  
  # find RPSS (LOOCV for PCR and PCANN; instrinsic estimate for PCQR and PCGLM)
  
  RPSmod <- numeric(N)
  RPSref <- numeric(N)
  # loop through time series
  for (t in 1:N) {        
    # loop through three flow categories
    RPSmod[t] <- 0
    RPSref[t] <- 0
    for (m in 1:3) {      
      Yref <- m/3                                            # P(flow <= Qcrit) from climatology (just follows terciles because category cutoffs were based on climatology)    
      if (obs[t] <= Qcrit[m]) {
        O <- 1                                               # observed cumulative probability that flow is less than Qcrit
      } else {
        O <- 0
      }
      RPSmod[t] <- RPSmod[t] + (Ymod[t,m] - O)^2
      RPSref[t] <- RPSref[t] + (Yref - O)^2
    }
  }
  RPSS_LOOCV <<- 1 - (mean(RPSmod)/mean(RPSref))
  rm(RPSmod,RPSref,Ymod,Yref)
  
  
  # find probabilities of hits and misses as complement to RPSS (in-sample) 
  
  N_LowTotal <- 0
  N_LowHit <- 0
  N_LowMiss <- 0
  N_NormTotal <- 0
  N_NormHit <- 0
  N_NormMiss <- 0
  N_HighTotal <- 0
  N_HighHit <- 0
  N_HighMiss <- 0
  for (t in 1:N) {
    if (obs[t] <= Qcrit[1]) {
      N_LowTotal <- N_LowTotal + 1
      if (prd[t] <= Qcrit[1]) {
        N_LowHit <- N_LowHit + 1
      } else {
        N_LowMiss <- N_LowMiss + 1
      }
    }
    if ((obs[t] > Qcrit[1]) && (obs[t] <= Qcrit[2])) {
      N_NormTotal <- N_NormTotal + 1
      if ((prd[t] > Qcrit[1]) && (prd[t] <= Qcrit[2])) {
        N_NormHit <- N_NormHit + 1
      } else {
        N_NormMiss <- N_NormMiss + 1
      }
    }
    if (obs[t] > Qcrit[2]) {
      N_HighTotal <- N_HighTotal + 1
      if (prd[t] > Qcrit[2]) {
        N_HighHit <- N_HighHit + 1
      } else {
        N_HighMiss <- N_HighMiss + 1
      }
    }
  }
  P_LowHit <- N_LowHit/N_LowTotal
  P_LowMiss <- N_LowMiss/N_LowTotal
  P_NormHit <- N_NormHit/N_NormTotal
  P_NormMiss <- N_NormMiss/N_NormTotal
  P_HighHit <- N_HighHit/N_HighTotal
  P_HighMiss <- N_HighMiss/N_HighTotal
  rm(N_LowTotal,N_LowHit,N_LowMiss,N_NormTotal,N_NormHit,N_NormMiss,N_HighTotal,N_HighHit,N_HighMiss)
  
  # find probabilities of hits and misses as complement to RPSS (LOOCV)
  
  N_LowTotal <- 0
  N_LowHit <- 0
  N_LowMiss <- 0
  N_NormTotal <- 0
  N_NormHit <- 0
  N_NormMiss <- 0
  N_HighTotal <- 0
  N_HighHit <- 0
  N_HighMiss <- 0
  for (t in 1:N) {
    if (obs[t] <= Qcrit[1]) {
      N_LowTotal <- N_LowTotal + 1
      if (prd_LOOCV[t] <= Qcrit[1]) {
        N_LowHit <- N_LowHit + 1
      } else {
        N_LowMiss <- N_LowMiss + 1
      }
    }
    if ((obs[t] > Qcrit[1]) && (obs[t] <= Qcrit[2])) {
      N_NormTotal <- N_NormTotal + 1
      if ((prd_LOOCV[t] > Qcrit[1]) && (prd_LOOCV[t] <= Qcrit[2])) {
        N_NormHit <- N_NormHit + 1
      } else {
        N_NormMiss <- N_NormMiss + 1
      }
    }
    if (obs[t] > Qcrit[2]) {
      N_HighTotal <- N_HighTotal + 1
      if (prd_LOOCV[t] > Qcrit[2]) {
        N_HighHit <- N_HighHit + 1
      } else {
        N_HighMiss <- N_HighMiss + 1
      }
    }
  }
  P_LowHit_LOOCV <- N_LowHit/N_LowTotal
  P_LowMiss_LOOCV <- N_LowMiss/N_LowTotal
  P_NormHit_LOOCV <- N_NormHit/N_NormTotal
  P_NormMiss_LOOCV <- N_NormMiss/N_NormTotal
  P_HighHit_LOOCV <- N_HighHit/N_HighTotal
  P_HighMiss_LOOCV <- N_HighMiss/N_HighTotal
  prob_for_plotting <- c(P_LowHit_LOOCV, P_NormHit_LOOCV, P_HighHit_LOOCV)  # set aside for subsequent inclusion in diagnostic plots
  rm(N_LowTotal,N_LowHit,N_LowMiss,N_NormTotal,N_NormHit,N_NormMiss,N_HighTotal,N_HighHit,N_HighMiss)  
  
  
  # PROBABILISTIC DIAGNOSTICS: FORECAST DISTRIBUTION RELIABILITY METRIC
  
  bincount <- numeric(6)  # asses frequency that the value ultimately observed falls within the 100%-90% exceedance probability band in the prediction bounds, 90%-70%, etc
  for (t in 1:N) {
    if (obs[t] < y90[t]) { bincount[1] <- bincount[1] + 1 }
    if ((obs[t] > y90[t]) && (obs[t] <= y70[t])) { bincount[2] <- bincount[2] + 1 }
    if ((obs[t] > y70[t]) && (obs[t] <= prd[t])) { bincount[3] <- bincount[3] + 1 }
    if ((obs[t] > prd[t]) && (obs[t] <= y30[t])) { bincount[4] <- bincount[4] + 1 }
    if ((obs[t] > y30[t]) && (obs[t] <= y10[t])) { bincount[5] <- bincount[5] + 1 }
    if (obs[t] > y10[t]) { bincount[6] <- bincount[6] + 1 }
  }
  bincount_expected <- N*c(0.1,0.2,0.2,0.2,0.2,0.1)
  bincount_plot <- rbind(bincount, bincount_expected)  # set aside for subsequent inclusion in diagnostic plots
  
  
  # MULTI-PANEL FIGURE WITH DIAGNOSTIC PLOTS

  # create layout for plotting

  # graphics.off()
  # par("mar")
  # par(mar=c(1,1,1,1))
  
  dev.new()
  plot.new()
  m <- rbind(c(1,1),c(2,3),c(4,5),c(6,7),c(8,9))
  layout(m)

  # plot observed time series and mean, 10th, 30th, 70th, and 90th exceedance flow predictions

  all <- c(prd,obs,y90,y70,y30,y10)
  range <- c(min(all), max(all))
  plot(year, obs, ylim=range, xlab="year",ylab="obs (kaf)", type="o", lwd=2, col="red", main=plottitle)
  lines(year,prd, col="black", lwd=2)
  lines(year,y90, col="gray", lwd=2, lty=1)
  lines(year,y70, col="gray", lwd=2, lty=2)
  lines(year,y30, col="gray", lwd=2, lty=2)
  lines(year,y10, col="gray", lwd=2, lty=1)
  abline(0,0,col="red")

  # plot basic diagnostics: observed vs. predicted, residuals

  all = c(prd,obs)
  range = c(min(all), max(all))
  plot(prd, obs, xlim=range, ylim=range, xlab="predicted",ylab="observed", type="p", main="obs vs. pred")
  abline(0,1,col="red")
  rm(all, range)
  plot(prd,res, xlab="predicted",ylab="residual", main="residuals")
  abline(0,0,col="red")

  # examine distributional characteristics of residuals: quantile-quantile plot, Shapiro-Wilk test, histogram

  res_zscore <- (res - mean(res))/sd(res)
  qqnorm(res_zscore, main="Q-Q plot of standardized residuals")
  swout <- shapiro.test(res_zscore)
  # text(-1,1,paste("Shapiro-Wilk p = ",substr(swout[2],1,4)))
  a <- as.numeric(swout[2])
  b <- round(a,2)
  text(-1,1,paste("Shapiro-Wilk p = ",b))
  abline(0,1,col="red")
  hist(res, xlab="residual", main="histogram of residuals")
  rm(a,b)

  # examine memory structure of residuals: ACF and PACF functions

  acf(res, xlim=c(1,10), main="autocorrelation of residuals") # Plots the ACF of x for lags 1 to 10
  pacf(res, xlim=c(1,10), main="partial autocorrelation of residuals") # Plots the PACF of x for lags 1 to 10

  # include LOOCV hit rates
  
  barplot(prob_for_plotting, ylim=c(0,1), main="categorical hit rates", ylab="P(hit)", names.arg = c("dry","normal","wet"))
  abline(1,0,col="blue",lty=2)
  abline(1/3,0,col="red",lty=2)
  
  # include forecast distribution assessment
  
  colors <- c("gray","black")
  barplot(bincount_plot, beside = T, col=colors, names.arg = c("100-90","90-70","70-BE","BE-30","30-10","10-0"), xlab="exeedance probability (%)", ylab="frequency", main="forecast distribution")
  
  
  # FIND MAXIMUM ERRORS AND YEARS OF OCCURRENCE:
  
  max_abs_err <- max(abs(res))
  index <- which(abs(res) == max_abs_err)
  year_max_abs_err <- year[index]
  rm(index)
  
  percent_res <- 100*res/obs
  max_rel_err <- max(abs(percent_res))
  index <- which(abs(percent_res) == max_rel_err)
  year_max_rel_err <- year[index]
  rm(index)
  
  
  # CHECK WHETHER 90% EXCEEDANCE PROBABILITY PREDICTION OR BEST ESTIMATE PREDICTION ARE NON-PHYSICAL:
  
  if (min(y90) < 0) {
    PB_LT_0_flag = "Y"
  } else {
    PB_LT_0_flag = "N" 
  }
  if (min(prd) < 0) {
    BE_LT_0_flag = "Y"
  } else {
    BE_LT_0_flag = "N"
  }
  
  
  # COMPILE PERFORMANCE METRICS INTO A DATA FRAME FOR EASY VIEWING AND SAVING:
  
  # standard reporting suite: 
  
  # Rsqrd, RMSE, and categorical hit rates (HR-L, HR-M, HR-H) are all LOOCV
  # RPSS is LOOCV for post-processed prediction bounds (e.g., PCR) or intrinsic otherwise (e.g., PCQR)
  # physicality checks for best-estimate and prediction bounds; bin counts for forecast distribution; and maximum error statistics are in-sample
  metrics <- c('Rsqrd','RMSE','RPSS','HR-L','HR-M','HR-H','BE<0?','PB<0?','FD 100-90','FD 90-70','FD 70-BE','FD BE-30','FD 30-10','FD 10-0','Max Abs Err','Year of max','Max Rel Err','Year of max')
  values <-c(Rsqrd_LOOCV,RMSE_LOOCV,RPSS_LOOCV,P_LowHit_LOOCV,P_NormHit_LOOCV,P_HighHit_LOOCV,BE_LT_0_flag,PB_LT_0_flag,bincount[1],bincount[2],bincount[3],bincount[4],bincount[5],bincount[6],max_abs_err,year_max_abs_err,max_rel_err,year_max_rel_err)
  reporting_metrics.output <<- data.frame(metrics,values)
  rm(metrics,values)
  
  #additional reporting:
  
  # include in-sample and LOOCV Rsqrd, RMSE, and hit rates in one table to compare for overtraining etc
  metrics <- c('Rsqrd','RMSE','HR-L','HR-M','HR-H')
  in_sample_values <-c(Rsqrd,RMSE,P_LowHit,P_NormHit,P_HighHit)
  LOOCV_values <- c(Rsqrd_LOOCV,RMSE_LOOCV,P_LowHit_LOOCV,P_NormHit_LOOCV,P_HighHit_LOOCV)
  other_metrics.output <<- data.frame(metrics,in_sample_values,LOOCV_values)
  rm(metrics,in_sample_values,LOOCV_values)

  
  # CLOSE OUT FUNCTION
  
}

