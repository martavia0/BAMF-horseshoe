//
// BAMF + Horseshoe prior all m/zs
//   7/5/2024
// T-students changed to Cauchys as in Piironen and Vehtari
// F modelled as a function of lambda_tilde instead of just lambda ( Horseshoe is regularised)
// Lambda is m/z dependent. 
//
//
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
}

transformed data {
    
  vector[n*m] Xv = to_vector(X);
  vector[n*m] sigmav = to_vector(sigma);  
}

parameters {
  matrix<lower=0>[p,m] z;                     // Random-effects auxiliary variable for horseshoe
  matrix<lower=0>[n,p] G; // score matrix
  simplex[m] F[p]; // loadings simplex
}
  
model {
  Xv ~ normal(to_vector(G*vector_array_to_matrix(F)), sigmav);
}
 