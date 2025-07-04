// BAMF + Lasso prior (replacing Horseshoe)
// Modified: 7/5/2024

functions {
  real double_exponential_lpdf(real x, real mu, real sigma) {
    return log(0.5) - log(sigma) - fabs(x - mu) / sigma;
  }

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
  matrix<lower=0>[n,m] sigma;  // separate sigma for each of the observations
  vector<lower=0>[n-1] timesteps; // How much time elapses between values in X
  real<lower=0> minerr; // Minimal error
}

transformed data {
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);
}

parameters {
  matrix[p, m] z;                 // Basis weights (to be sparsified by Lasso)
  matrix<lower=0>[n,p] G;         // Score matrix
  vector<lower=0>[p] alpha_a;     // Posterior a
  vector<lower=minerr>[p] alpha_b;// Posterior b
}

transformed parameters {
  simplex[m] F[p];                // Profiles
  matrix<lower=0>[p, m] mu;       // Unnormalized profile weights
  vector[p] total;

  for (k in 1:p){
    total[k] = 0;
    for (j in 1:m){
      mu[k,j] = fabs(z[k,j]);      // Use magnitude for positivity
      total[k] += mu[k,j];
    }
    for (j in 1:m){
      F[k,j] = mu[k,j] / total[k]; // Normalize
    }
  }
}

model {
  real lasso_scale = 0.1; // Adjust this to control sparsity

  // Lasso prior on z
  for (k in 1:p) {
    for (j in 1:m) {
      target += double_exponential_lpdf(z[k, j] | 0, lasso_scale);
    }
  }

  // Cauchy dynamics on G
  for(j in 1:p) {
    G[2:,j] ~ cauchy(G[:(n-1),j], alpha_a[j]*timesteps + alpha_b[j]);
  }

  // Observation model
  Xv ~ normal(to_vector(G * vector_array_to_matrix(F)), sigmav);
}
