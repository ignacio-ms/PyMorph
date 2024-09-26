library(maigesPack)
library(FNN)
library(readxl)
library(aricode)


compute_mutinfo <- function(x, y, k = 2) {
  mutinfo(x, y, k = k)
}

# Function to perform bootstrapping and calculate p-value
bootstrap_pvalue_randomization <- function(X, Y, bRep = 100, k = 2) {
  # Compute the original MI
  mi_original <- compute_mutinfo(X, Y, k)
  
  # Initialize vector to store bootstrap MI values
  mi_null <- numeric(bRep)
  
  # Perform bootstrapping
  for (i in 1:bRep) {
    # Shuffle both datasets independently
    X_permuted <- sample(X)
    Y_permuted <- sample(Y)
    # Compute MI for the permuted data
    mi_null[i] <- compute_mutinfo(X_permuted, Y_permuted, k)
  }
  
  # Compute p-value as the proportion of null MI values greater than or equal to the original MI
  p_value <- mean(mi_null >= mi_original)
  
  return(p_value)
}

get_mu_sigma <- function(data) {
  mu <- mean(data)
  sigma <- sd(data)
  return(list(mu = mu, sigma = sigma))
}

# Define the function to compute Adjusted Mutual Information (AMI)
compute_ami <- function(x, y) {
  AMI(x, y)
}

# Define the function for bootstrapping and calculating p-value
bootstrap_ami_pvalue <- function(X, Y, bRep = 100) {
  # Compute the original AMI
  ami_original <- compute_ami(X, Y)
  
  # Initialize a vector to store the AMI values from the bootstrap samples
  ami_null <- numeric(bRep)
  
  # Perform the bootstrap procedure
  for (i in 1:bRep) {
    # Randomly shuffle both datasets
    X_permuted <- sample(X)
    Y_permuted <- sample(Y)
    
    # Compute the AMI for the permuted datasets
    ami_null[i] <- compute_ami(X_permuted, Y_permuted)
  }
  
  # Calculate the p-value as the proportion of bootstrapped AMI values that are greater than or equal to the original AMI
  p_value <- mean(ami_null >= ami_original)
  
  return(p_value)
}

# Function to normalize data using provided mean and standard deviation
normalize <- function(data, mu, sigma) {
  return((data - mu) / sigma)
}


X <- read_excel("C:/Users/ignac/OneDrive/Nacho/CNIC/Cicerone/Cris/Cris/WT MI/bitot_results(vox).xlsx", sheet = 2)
Y <- read_excel("C:/Users/ignac/OneDrive/Nacho/CNIC/Cicerone/Cris/Cris/WT no MI(New)/bitot_results(vox).xlsx", sheet = 2)

X <- X$IntegratedIntensity
Y <- Y$IntegratedIntensity

params <- get_mu_sigma(X)
mu <- params$mu
sigma <- params$sigma

X_norm <- normalize(X, mu, sigma)
Y_norm <- normalize(Y, mu, sigma)

mi <- mutinfo(X, Y, k=2)
mi_norm <- mutinfo(X_norm, Y_norm, k=2)
# p_val <- bootstrapMI(X, Y, bRep=500, ret='p-value')

cat("MI:", mi, "\n")
cat("MI Normalized:", mi_norm, "\n")

# p_val <- bootstrap_pvalue(X, Y, bRep = 1000, k = 2)
# p_val_norm <- bootstrap_pvalue(X_norm, Y_norm, bRep = 1000, k = 2)
# cat("P-value for Mutual Information:", p_val, "\n")
# cat("P-value for Mutual Information:", p_val_norm, "\n")

p_val <- bootstrap_pvalue_randomization(X, Y, bRep = 1000, k = 2)
p_val_norm <- bootstrap_pvalue_randomization(X_norm, Y_norm, bRep = 1000, k = 2)
cat("P-value for Mutual Information:", p_val, "\n")
cat("P-value for Mutual Information Normalized:", p_val_norm, "\n")


# ami <- compute_ami(X, Y)
# ami_norm <- compute_ami(X_norm, Y_norm)
# 
# p_val <- bootstrap_ami_pvalue(X, Y, bRep = 500)
# p_val_norm <- bootstrap_ami_pvalue(X_norm, Y_norm, bRep = 500)
# cat("P-value for Adjusted Mutual Information:", p_val, "\n")
# cat("P-value for Adjusted Mutual Information Normalized:", p_val_norm, "\n")

