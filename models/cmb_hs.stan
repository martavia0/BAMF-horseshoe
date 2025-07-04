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
  matrix[n,p] G;      // G matrix
  vector<lower=0>[n-1] timesteps; //How much time elapses between values in X (arbitrary unit, but unit choice also affects other priors)
  matrix<lower=0>[n,m] sigma;  // separate sigma for each of the observations (!)
  real<lower=0> minerr; //Minimal error
}

transformed data {
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);
  real <lower = 1> nu_global =1; //dof for thau
  real <lower = 1> nu_local =1; //dof for lambda
  real < lower =0> p_zero =1  ; //# slab scale for the regularized horseshoe
  real < lower =0> scale_global;
  real < lower =0> slab_scale =2.5  ; //# slab scale for the regularized horseshoe
  real < lower =0> slab_df = 4; //# slab degrees of freedom for the regularized horseshoe
  scale_global = p_zero / ((p-p_zero)*sqrt(n));
}

parameters {
  matrix<lower=0>[p,m] z;                     // Random-effects auxiliary variable for horseshoe
  vector<lower=0>[m] tau; // Global shrinkage parameter
  matrix < lower =0>[p,m] lambda ; // Local shrinkage parameter // This is only m or sigma?
  real < lower =0> caux ;  //Regularisation parameter 
}

transformed parameters {
  simplex[m] F[p]; // loadings simplex
  matrix < lower =0 >[p,m] lambda_tilde ; //  'truncated ' local shrinkage parameter
  real < lower =0> c; // slab scale
  matrix <lower=0>[p,m] mu; 
  vector[p] total; 

  c = slab_scale * sqrt ( caux );
  for (k in 1:p) {
    lambda_tilde[k,] = sqrt ( c^2 * square (lambda[k,]) ./ (c^2 + square (lambda[k,]) .* to_row_vector(square(tau))) );
  }
  // We construct mu
  for (k in 1:p){
    total[k]=0;
    for (j in 1:m){
        mu[k,j] = z[k,j] * lambda_tilde[k,j] * tau[j]; 
        total[k] += mu[k,j];  // Accumulate the total sum for normalization
    }
    for (j in 1:m){
        F[k,j] = mu[k,j] / total[k];  // Normalize mu to ensure F[k,] is a valid simplex
    }
  }
}

model {
  for (j in 1:m) {
    for (k in 1:p) {
      lambda[k,j] ~ student_t(nu_local, 0, 1);
      z[k,j] ~ cauchy(0,1);
    }
  }
  tau ~ student_t (nu_global, 0, scale_global );
  caux ~ inv_gamma (0.5* slab_df , 0.5* slab_df );
  
  Xv ~ normal(to_vector(G*vector_array_to_matrix(F)), sigmav);
}
