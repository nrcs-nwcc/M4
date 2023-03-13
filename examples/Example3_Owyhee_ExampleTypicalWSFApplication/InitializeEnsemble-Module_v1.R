# R FUNCTION TO INITIALIZE ENSEMBLE MEAN GENERATION PROCESS


INITIALIZE_ENSEMBLE <- function() {
  
  y90_ensemble <<- 0
  y70_ensemble <<- 0
  prd_ensemble <<- 0
  prd_ensemble_LOOCV <<- 0
  y30_ensemble <<- 0
  y10_ensemble <<- 0
  Ymod_ensemble <<- 0
  N_ensembles <<- 0
  
}