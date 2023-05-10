# R FUNCTION TO PERFORM PRINCIPAL COMPONENTS ANALYSIS


PCA <- function(datmatR,PCSelection) {


  # PROCESS DATA
  
  # Standardize each time series within data matrix to zero mean and unit variance
  MeanOfEachVariate <<- rowMeans(datmatR)
  zeroed_matrix <- datmatR-MeanOfEachVariate
  library(matrixStats)
  StdevOfEachVariate <<- rowSds(zeroed_matrix)
  Y <- zeroed_matrix/StdevOfEachVariate  

  # Find correlation matrix:
  C <- (1/ncol(Y))*Y%*%t(Y)   


  # PERFORM PCA

  # Eigenvalue decomposition
  eigenanalysis <- eigen(C)
  lambda <<- eigenanalysis$values  # vector containing eigenvalues sorted from largest to smallest
  E <<- eigenanalysis$vectors  # matrix: each column contains an eigenvector, sorted with the leading mode in the left-most column; each row corresponds to a different variable, sorted the same as in the data matrix

  # PC time series (only up to top four modes considered as modeling predictors in this version):
  A <<- t(E)%*%Y  # matrix: time increases to the right, each row corresponds to a different mode, with the leading mode in the top row
  PC1 <<- A[1,]
  if (max(PCSelection) > 1) {PC2 <<- A[2,]}
  if (max(PCSelection) > 2) {PC3 <<- A[3,]}
  if (max(PCSelection) > 3) {PC4 <<- A[4,]}
  

# CLOSE OUT FUNCTION

}
