functions {
  matrix vector_array_to_matrix(vector[] x) {
    matrix[size(x), rows(x[1])] y;
    for (m in 1:size(x))
      y[m] = x[m]';
    return y;
  }
}

data {
  int<lower=1> n;     // number of data points
  int<lower=1> m;     // number of dimensions
  int<lower=1> p;     // number of components
  matrix[n,m] X;      // data matrix
  matrix<lower=0>[n,m] sigma;  // separate sigma for each of the observations (!)
  real<lower=0> minerr; //Minimal error
//   matrix<lower=0>[p,m] F; // F used as input
}

transformed data {
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);
}

parameters {
  matrix<lower=0>[n,p] G; // score matrix
  simplex[m] F[p]; // loadings simplex
  vector[p] alpha_a; // Posterior a
  vector[p] alpha_b; // Posterior b
  real<lower=0> sigma_a[p];
}

model {

  // The priors for G and F are implicitly given in the parameters section,
  //  so explicitly stating them here would only lead to unnecessary calculations
  //  (uniform priors become additive constants (in logspace), that can be ignored).
  
  // Column-wise autocorrelation
  for(j in 1:p) {
    G[1,j] ~ normal(0,100); // the first time point regularised
    G[2:,j] ~ normal(alpha_a[j] + alpha_b[j]*G[:(n-1),j], sigma_a[j]); // gaussian
  }

  // X ~ Z = G * F
  Xv ~ normal(to_vector(G*vector_array_to_matrix(F)), sigmav);

}