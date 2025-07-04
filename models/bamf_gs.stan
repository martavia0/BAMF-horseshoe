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
  vector<lower=0>[n-1] timesteps; //How much time elapses between values in X (arbitrary unit, but unit choice also affects other priors)
  real<lower=0> minerr; //Minimal error
}

transformed data {
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);
}

parameters {
  simplex[n] G[p]; // score simplex
  matrix<lower=0>[p,m] F; // loadings matrix, changed in 2!
  vector<lower=0>[p] alpha_a; // Posterior a
  vector<lower=minerr>[p] alpha_b; // Posterior b
}

transformed parameters{
  matrix[n,p] G_matrix; // converting G array to matrix
  for (j in 1:p) {
    for (i in 1:n) {
      G_matrix[i,j] = G[j][i];
    }
  }  
}

model {
 
  // Column-wise autocorrelation
  for(j in 1:p) {
    G[j,2:] ~ cauchy(G[j,:(n-1)], alpha_a[j]*timesteps + alpha_b[j]);
  }
  // X ~ Z = G * F
  Xv ~ normal(to_vector(G_matrix * F), sigmav);

}
