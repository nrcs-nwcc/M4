# R FUNCTION TO FINALIZE ENSEMBLE MEAN GENERATION PROCESS


FINALIZE_ENSEMBLE <- function() {
  
  y90_ensemble <<- y90_ensemble/N_ensembles  
  y70_ensemble <<- y70_ensemble/N_ensembles
  prd_ensemble <<- prd_ensemble/N_ensembles
  prd_ensemble_LOOCV <<- prd_ensemble_LOOCV/N_ensembles
  y30_ensemble <<- y30_ensemble/N_ensembles
  y10_ensemble <<- y10_ensemble/N_ensembles
  res_ensemble <<- prd_ensemble - obs
  res_ensemble_LOOCV <<- prd_ensemble_LOOCV - obs
  Ymod_ensemble <<- Ymod_ensemble/N_ensembles
  
}