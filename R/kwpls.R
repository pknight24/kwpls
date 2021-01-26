#' Kernel-weighted partial least squares
#'
#' @param X The n by p data matrix of predictors
#' @param Y The n by q matrix of outcomes
#' @param H An n by n sample-wise similarity matrix.
#' @param Q1 A p by p similarity matrix of the features in X.
#' @param Q2 A q by q similarity matrix corresponding to columns in Y.
#' @param k Integer, the number of PLS latent variables to compute.
#' @return Scores and loadings for X and Y.
#' @export
kwpls <- function(X, Y, H = diag(nrow(X)), Q1 = diag(ncol(X)), Q2 = diag(ncol(Y)), k = 2)
{
  Xj <- X
  Yj <- Y
  T <- matrix(0, nrow(X), k)
  U <- matrix(0, nrow(Y), k)
  P <- matrix(0, ncol(X), k)
  C <- matrix(0, ncol(Y), k)
  for (j in 1:k)
  {
    svd.out <- svd(Q1 %*% t(Xj) %*% H %*% Yj %*% Q2, nu = 1, nv = 1) # compute the weights, this is the slowest part
    tj <- Xj %*% svd.out$u[,1] # new latent variable for X
    if (is.vector(Y)) # latent variable for Y, checking to see if Y is a vector to avoid conformability errors
      uj <- Yj * svd.out$v[1,1]
    else
      uj <- Yj %*% svd.out$v[,1]
    pj <- t(Xj) %*% tj / (t(tj) %*% tj)[1,1]
    Xj <- Xj - tj %*% t(pj) # deflate X
    b <- t(tj) %*% uj / (t(tj) %*% tj)[1,1] # b is the coefficient found when fitting the model u = b * t
    if (is.vector(Y)) # deflate Y using the linear model fit
      Yj <- Yj - b[1,1] * tj * svd.out$v[1,1]
    else
      Yj <- Yj - b[1,1] * tj %*% t(svd.out$v[,1])
    T[,j] <- tj # save the current latent variable tj
    U[,j] <- uj
    P[,j] <- pj
    C[,j] <- svd.out$v[,1]
  }

  return(list(X.scores = T,
              X.loadings = P,
              Y.scores = U,
              Y.loadings= C))

}
