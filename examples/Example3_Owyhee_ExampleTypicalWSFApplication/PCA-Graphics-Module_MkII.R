# R FUNCTION TO PERFORM DIAGNOSTICS ON MODELING OUTCOMES  


PCAgraphics <- function(plottitle1,plottitle2,plottitle3) {
 
  
  # Note: in PCA-Graphics-Module_v3.R, comment out ordination diagrams until notation of successive retained input variables can be clarified
  # in the context of a GA-driven input variable optimization to more intuitively reflect which variables in the original input variable candidate
  # pool are retained.   
  
  
  # Find and plot eigenspectrum
  perc_var_expl <<- 100*lambda/sum(lambda)
  names <- seq(1:ncol(E))
  # dev.new()
  barplot(perc_var_expl, ylim=c(0,100), names.arg = names, main = plottitle1, xlab='PCA mode', ylab='% variance', cex.lab=1.2, cex.axis=1.1, cex.main=1.4, cex.sub=1.1)
  
  # Plot PCs
  dev.new()
  plot(year, PC1, main = plottitle2, ylab = "PCA scores", type="b", bty="l", xlab = "year", col=rgb(0.2,0.4,0.1,0.7), lwd=2, pch=17, lty=1, cex.lab=1.5, cex.axis=1.2, cex.main=1.5, cex.sub=1.1)
  if (max(PCSelection) == 1) {
    legend("bottomleft", legend = c("PC1"),col = rgb(0.2,0.4,0.1,0.7),pch = c(17),bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1))
  }
  if (max(PCSelection) == 2) {
    lines(year, PC2, col=rgb(0.8,0.4,0.1,0.7), lwd=2, pch=19, type="b", lty=1)
    legend("bottomleft", legend = c("PC1", "PC2"),col = c(rgb(0.2,0.4,0.1,0.7),rgb(0.8,0.4,0.1,0.7)),pch = c(17,19),
           bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1))
  }
  if (max(PCSelection) == 3) {
    lines(year, PC2, col=rgb(0.8,0.4,0.1,0.7), lwd=2, pch=19, type="b", lty=1)
    lines(year, PC3, col="lightsteelblue3", lwd=3 , pch=18 , type="b", lty=2)
    legend("bottomleft",legend = c("PC1", "PC2","PC3"),col = c(rgb(0.2,0.4,0.1,0.7),rgb(0.8,0.4,0.1,0.7),"lightsteelblue3"),pch = c(17,19,18),
           bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1))
  }
  if (max(PCSelection) == 4) {
    lines(year, PC2, col=rgb(0.8,0.4,0.1,0.7), lwd=2, pch=19, type="b", lty=1)
    lines(year, PC3, col="lightsteelblue3", lwd=3 , pch=18 , type="b", lty=2)
    lines(year, PC4, col="indianred4", lwd=3 , pch=15 , type="b", lty=2)
    legend("bottomleft", legend = c("PC1", "PC2","PC3","PC4"),col = c(rgb(0.2,0.4,0.1,0.7), rgb(0.8,0.4,0.1,0.7),"lightsteelblue3","indianred4"), pch = c(17,19,18,15),
           bty = "n", pt.cex = 2, cex = 1.2, text.col = "black", horiz = F , inset = c(0.1, 0.1))
  }
  
  # Plot ordination diagram based on eigenvectors (loadings) for two leading modes
  # E1 <- E[,1]
  # E2 <- E[,2]
  # dev.new()
  # pointlabels = seq(1:nrow(E))
  # plot(E1,E2,asp = 1,pch=".",cex.lab=1.2, main = plottitle3, xlab='mode 1 eigenvector', ylab='mode 2 eigenvector', cex.axis=1.1, cex.main=1.4, cex.sub=1.1)
  # grid(col = "lightgray", lty = "dotted")
  # text(E1,E2,labels = pointlabels,col="deepskyblue4",cex = 1.1,font = 2)
  
  
  # CLOSE OUT FUNCTION
   
}