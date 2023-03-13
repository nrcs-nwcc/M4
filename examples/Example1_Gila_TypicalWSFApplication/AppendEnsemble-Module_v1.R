# R FUNCTION TO INCREMENTALLY ADD MORE MEMBERS TO AN ENSEMBLE MEAN  


APPEND_ENSEMBLE <- function(y90_member,y70_member,prd_member,prd_member_LOOCV,y30_member,y10_member,Ymod_member,y90_ensemble,y70_ensemble,prd_ensemble,prd_ensemble_LOOCV,y30_ensemble,y10_ensemble,Ymod_ensemble,N_ensembles) {
 
  y90_ensemble <<- y90_ensemble + y90_member
  y70_ensemble <<- y70_ensemble + y70_member
  prd_ensemble <<- prd_ensemble + prd_member
  prd_ensemble_LOOCV <<- prd_ensemble_LOOCV + prd_member_LOOCV
  y30_ensemble <<- y30_ensemble + y30_member
  y10_ensemble <<- y10_ensemble + y10_member
  Ymod_ensemble <<- Ymod_ensemble + Ymod_member
  N_ensembles <<- N_ensembles + 1
   
}