library(ivmodel)
library(AER)  # For alternative 2SLS implementation

# Load test data
no_exog_data <- read.csv("test/data/no_exogenous.csv")
exog_data <- read.csv("test/data/exogenous.csv")

cat("=== R ivmodel verification ===\n\n")

# Test case 1: No exogenous variables
cat("1. NO EXOGENOUS VARIABLES:\n")
Y <- no_exog_data$y
D <- no_exog_data$x
Z <- as.matrix(no_exog_data[, c("z1", "z2", "z3", "z4")])

cat("Data dimensions:\n")
cat("  Y (outcome):", length(Y), "observations\n")
cat("  D (endogenous):", length(D), "observations\n")
cat("  Z (instruments):", nrow(Z), "x", ncol(Z), "matrix\n")

# Fit ivmodel without exogenous variables
iv_result_no_exog <- ivmodel(Y = Y, D = D, Z = Z)

cat("\nChecking what's in the ivmodel object:\n")
cat("  Available slots:", names(iv_result_no_exog), "\n")
if (!is.null(iv_result_no_exog$TSLS)) {
    cat("  TSLS slot names:", names(iv_result_no_exog$TSLS), "\n")
    cat("  TSLS point estimate:", iv_result_no_exog$TSLS$point.est, "\n")
}

# Try manual 2SLS calculation
cat("\nManual 2SLS calculation:\n")
# First stage: D ~ Z (plus constant)
Z_with_const <- cbind(1, Z)
first_stage <- lm.fit(Z_with_const, D)
D_fitted <- Z_with_const %*% first_stage$coefficients

# Second stage: Y ~ D_fitted (plus constant)
X_2sls <- cbind(D_fitted, 1)
second_stage <- lm.fit(X_2sls, Y)
tsls_coef <- second_stage$coefficients[1]  # coefficient on endogenous variable
cat("  Manual TSLS coefficient:", tsls_coef, "\n")

# Alternative using AER package
cat("\nUsing AER package for 2SLS:\n")
data_df <- data.frame(Y = Y, D = D, Z1 = Z[,1], Z2 = Z[,2], Z3 = Z[,3], Z4 = Z[,4])
aer_result <- ivreg(Y ~ D | Z1 + Z2 + Z3 + Z4, data = data_df)
cat("  AER 2SLS coefficient:", coef(aer_result)[2], "\n")  # D coefficient
aer_summary <- summary(aer_result)
cat("  AER 2SLS std error (homoskedastic):", aer_summary$coefficients[2, 2], "\n")
aer_robust <- summary(aer_result, vcov = sandwich)
cat("  AER 2SLS std error (robust):", aer_robust$coefficients[2, 2], "\n")

cat("\nTSLS results (no exogenous):\n")
cat("  Coefficient:", iv_result_no_exog$TSLS$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_no_exog$TSLS$std.err, "\n")
if (!is.null(iv_result_no_exog$TSLS$var.rob) && !is.na(iv_result_no_exog$TSLS$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_no_exog$TSLS$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}

cat("\nLIML results (no exogenous):\n")
cat("  Coefficient:", iv_result_no_exog$LIML$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_no_exog$LIML$std.err, "\n")
if (!is.null(iv_result_no_exog$LIML$var.rob) && !is.na(iv_result_no_exog$LIML$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_no_exog$LIML$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}
cat("  Kappa:", iv_result_no_exog$LIML$kappa, "\n")

cat("\nFuller results (no exogenous):\n")
cat("  Coefficient:", iv_result_no_exog$Fuller$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_no_exog$Fuller$std.err, "\n")
if (!is.null(iv_result_no_exog$Fuller$var.rob) && !is.na(iv_result_no_exog$Fuller$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_no_exog$Fuller$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}
cat("  Kappa:", iv_result_no_exog$Fuller$kappa, "\n")

cat("\n" , rep("=", 50), "\n\n")

# Test case 2: With exogenous variables
cat("2. WITH EXOGENOUS VARIABLES:\n")
Y2 <- exog_data$y
D2 <- exog_data$x
Z2 <- as.matrix(exog_data[, c("z1", "z2", "z3", "z4", "z5")])
X2 <- as.matrix(exog_data[, c("exog1", "exog2")])

cat("Data dimensions:\n")
cat("  Y (outcome):", length(Y2), "observations\n")
cat("  D (endogenous):", length(D2), "observations\n")
cat("  Z (instruments):", nrow(Z2), "x", ncol(Z2), "matrix\n")
cat("  X (exogenous):", nrow(X2), "x", ncol(X2), "matrix\n")

# Manual 2SLS calculation with exogenous variables
cat("\nManual 2SLS calculation (with exogenous):\n")
# For TSLS with exogenous variables:
# 1. Instruments include original Z plus exogenous X
# 2. Regressors include endogenous D plus exogenous X
Z_full <- cbind(1, Z2, X2)  # intercept + instruments + exogenous
X_full <- cbind(D2, X2)     # endogenous + exogenous

# Project X_full onto Z_full
P_Z <- Z_full %*% solve(t(Z_full) %*% Z_full) %*% t(Z_full)
X_fitted <- P_Z %*% X_full

# Second stage: Y ~ X_fitted
second_stage_exog <- lm.fit(X_fitted, Y2)
tsls_coef_exog <- second_stage_exog$coefficients[1]  # coefficient on endogenous variable
cat("  Manual TSLS coefficient (endogenous):", tsls_coef_exog, "\n")

# Alternative using AER package with exogenous variables
cat("\nUsing AER package for 2SLS (with exogenous):\n")
data_df2 <- data.frame(Y = Y2, D = D2,
                      Z1 = Z2[,1], Z2 = Z2[,2], Z3 = Z2[,3], Z4 = Z2[,4], Z5 = Z2[,5],
                      exog1 = X2[,1], exog2 = X2[,2])
aer_result2 <- ivreg(Y ~ D + exog1 + exog2 | Z1 + Z2 + Z3 + Z4 + Z5 + exog1 + exog2, data = data_df2)
cat("  AER 2SLS coefficient (endogenous):", coef(aer_result2)[2], "\n")  # D coefficient
aer_summary2 <- summary(aer_result2)
cat("  AER 2SLS std error (homoskedastic):", aer_summary2$coefficients[2, 2], "\n")
aer_robust2 <- summary(aer_result2, vcov = sandwich)
cat("  AER 2SLS std error (robust):", aer_robust2$coefficients[2, 2], "\n")

# Fit ivmodel with exogenous variables
iv_result_exog <- ivmodel(Y = Y2, D = D2, Z = Z2, X = X2)

cat("\nTSLS results (with exogenous):\n")
cat("  Coefficient:", iv_result_exog$TSLS$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_exog$TSLS$std.err, "\n")
if (!is.null(iv_result_exog$TSLS$var.rob) && !is.na(iv_result_exog$TSLS$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_exog$TSLS$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}

cat("\nLIML results (with exogenous):\n")
cat("  Coefficient:", iv_result_exog$LIML$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_exog$LIML$std.err, "\n")
if (!is.null(iv_result_exog$LIML$var.rob) && !is.na(iv_result_exog$LIML$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_exog$LIML$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}
cat("  Kappa:", iv_result_exog$LIML$kappa, "\n")

cat("\nFuller results (with exogenous):\n")
cat("  Coefficient:", iv_result_exog$Fuller$point.est, "\n")
cat("  Std Error (homoskedastic):", iv_result_exog$Fuller$std.err, "\n")
if (!is.null(iv_result_exog$Fuller$var.rob) && !is.na(iv_result_exog$Fuller$var.rob)) {
    cat("  Std Error (robust):", sqrt(iv_result_exog$Fuller$var.rob), "\n")
} else {
    cat("  Std Error (robust): not available\n")
}
cat("  Kappa:", iv_result_exog$Fuller$kappa, "\n")

cat("\n=== Current Julia test expectations ===\n")
cat("No exogenous - TSLS: 1.1347818853655336\n")
cat("No exogenous - LIML: 1.9336876404887946\n")
cat("No exogenous - Fuller: 1.8056208909589633\n")
cat("With exogenous - TSLS: 1.0762813464043182\n")
cat("With exogenous - LIML: 1.2965861503223310\n")
cat("With exogenous - Fuller: 1.2832402012315129\n")