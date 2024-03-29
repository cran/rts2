functions {
  matrix getAD_L(real alpha, real theta, int M, int n,
                 array[] real dists, array[,] int NN, int mod){
    matrix[M+1,n] AD = rep_matrix(0,M+1,n);
    int idxlim;
    int idx1; 
    int idx2;
    real dist;
    AD[M+1,1] = alpha;
    matrix[M,M] smat = rep_matrix(0,M,M);
    vector[M] svec = rep_vector(0,M);
    int index;
    matrix[n,n] L = rep_matrix(0,n,n);
    real lsum;
    
    for(i in 2:n){
      idxlim = i<=(M) ? i-1 : M;
      for(j in 1:idxlim){
        smat[j,j] = alpha;
      }
      if(idxlim > 1){
        for(j in 1:(idxlim-1)){
          for(k in (j+1):idxlim){
            idx1 = NN[j,i] < NN[k,i] ? NN[j,i] : NN[k,i];
            idx2 = NN[j,i] < NN[k,i] ? NN[k,i] : NN[j,i];
            index = (n-1)*(idx1-1)-(((idx1-2)*(idx1-1))%/%2)+(idx2-idx1-1)+1;
            dist = dists[index];
            if(mod == 0){
              smat[j,k] = alpha * exp(-1.0*(dist*dist)/(theta*theta));
              smat[k,j] = smat[j,k];
            } else {
              smat[j,k] = alpha * exp(-1.0*dist/theta);
              smat[k,j] = smat[j,k];
            }
          }
        }
      }
      for(j in 1:idxlim){
        index = (n-1)*(NN[j,i]-1)-(((NN[j,i]-2)*(NN[j,i]-1))%/%2)+(i - NN[j,i] -1)+1;
        dist = dists[index];
        svec[j] = mod == 0 ? alpha * exp(-1.0*(dist*dist)/(theta*theta)) : alpha * exp(-1.0*dist/theta);
      }
      AD[1:idxlim,i] = mdivide_left_spd(smat[1:idxlim,1:idxlim] , svec[1:idxlim]);
      AD[M+1,i] = alpha - dot_product(AD[1:idxlim,i],svec[1:idxlim]);
    }
    // create the cholesky decomposition approximation
    for(k in 1:n){
      for(i in k:n){
        idxlim = i<=(M) ? i-1 : M;
        lsum = 0;
        if(idxlim > 0){
          for(j in 1:idxlim){
            lsum += AD[j,i] * L[NN[j,i],k];
          }
        }
        L[i,k] = i==k ? (1+lsum) : lsum;
      }
    }
    for(k in 1:n){
      L[,k] *= sqrt(AD[M+1,k]);
    }
    return(L);
   }
   real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[start:end]);
  }
   real partial_sum2_lpmf(array[] int y,int start, int end, vector mu){
    return poisson_log_lpmf(y[start:end]|mu[start:end]);
  }
}
data {
  int<lower=1> D; //number of dimensions
  int<lower=1> Q; //number of covariates
  int<lower=1> M; // number of nearest neighbours
  int<lower=1> Nsample; //number of observations per time period
  int nT; //number of time periods
  array[M,Nsample] int NN;
  array[Nsample*nT] int y; //outcome
  matrix[Nsample,D] x_grid; //prediction grid and observations
  vector[Nsample*nT] popdens; //population density
  matrix[Nsample*nT,Q] X;
  array[2] real prior_lscale;
  array[2] real prior_var;
  array[Q] real prior_linpred_mean;
  array[Q] real prior_linpred_sd;
  int mod;
  int<lower = 0, upper = 1> known_cov;
  real<lower=0> sigma_data; 
  real<lower=0> phi_data; 
}

transformed data {
  vector[Nsample*nT] logpopdens = log(popdens);
  matrix[known_cov ? Nsample : 0,known_cov ? Nsample : 0] L_data;
  array[(Nsample*(Nsample-1))%/%2] real dists;
  
  for(i in 1:(Nsample-1)){
    for(j in (i+1):Nsample){
      dists[(Nsample-1)*(i-1)-(((i-2)*(i-1))%/%2)+(j-i-1)+1] = sqrt((x_grid[i,1] - x_grid[j,1]) * (x_grid[i,1] - x_grid[j,1]) +
              (x_grid[i,2] - x_grid[j,2]) * (x_grid[i,2] - x_grid[j,2]));
    }
  }
  if(known_cov){
    L_data = getAD_L(sigma_data, phi_data, M, Nsample, dists, NN, mod);
  }
}

parameters {
  array[known_cov ? 0 : 1] real<lower=1e-05> phi_param; //length scale
  array[known_cov ? 0 : 1] real<lower=1e-05> sigma_param;
  vector[Q] gamma;
  array[nT > 1 ? 1 : 0] real<lower=-1,upper=1> ar;
  vector[Nsample*nT] f_raw;
}

transformed parameters {
  matrix[Nsample,Nsample] L;
  real<lower=1e-05> phi; //length scale
  real<lower=1e-05> sigma;
  vector[Nsample*nT] f;
  if(known_cov){
    sigma = sigma_data;
    phi = phi_data;
    L = L_data;
  } else {
    sigma = sigma_param[1];
    phi = phi_param[1];
    L = getAD_L(sigma, phi, M, Nsample, dists, NN, mod);
  }
  for(t in 1:nT){
    if(nT>1){
      if(t==1){
        f[1:Nsample] = (1/(1-ar[1]^2))*L*f_raw[1:Nsample];
      } else {
        f[(Nsample*(t-1)+1):(t*Nsample)] = ar[1]*f[(Nsample*(t-2)+1):((t-1)*Nsample)] + L*f_raw[(Nsample*(t-1)+1):(t*Nsample)];
      }
    } else {
      f = L*f_raw;
    }
  }
}

model{
  if(!known_cov){
    phi_param[1] ~ normal(prior_lscale[1],prior_lscale[2]);
    sigma_param[1] ~ normal(prior_var[1],prior_var[2]);
  }
  if(nT>1) ar[1] ~ normal(0,1);
  for(q in 1:Q){
    gamma[q] ~ normal(prior_linpred_mean[q],prior_linpred_sd[q]);
  }
  int grainsize = 1;
  target += reduce_sum(partial_sum1_lpdf,to_array_1d(f_raw),grainsize);
  target += reduce_sum(partial_sum2_lpmf,y,grainsize,X*gamma+logpopdens+f);
}

generated quantities{
  vector[Nsample*nT] y_grid_predict;
  
  for(i in 1:(Nsample*nT)){
    y_grid_predict[i] = exp(X[i,]*gamma+logpopdens[i]+f[i]);
  }
}
