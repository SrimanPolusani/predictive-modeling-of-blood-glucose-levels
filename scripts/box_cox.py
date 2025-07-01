import numpy as np

class BoxCox:
    def __init__(self, data, lambda_range, lambda_step):
        # Make sure that x is indeed a np array
        self.data = np.asarray(data)

        self.lambda_range = lambda_range
        self.lambda_step = lambda_step
        self.current_lambda = 0

    def box_cox_transform(self):
        """Applies box cox transformation to a given variable by taking x (variable values) and lambda as inputs"""

        # Do a log transformation if lambda_ value is closer to zero
        if np.isclose(self.current_lambda, 0):
            return np.log(self.data)
        else:
            return (np.power(self.data, self.current_lambda) - 1)/self.current_lambda

    def box_cox_log_likelihood(self):
        """Computes and returns log likelihood of the transformed variable data by taking x(variable data) and lambda_ as inputs"""
        # Find total number of values in x
        n = len(self.data)

        # Transform x
        y = self.box_cox_transform()
        # Find the population variance(ddof=0) of y
        y_var = np.var(y, ddof=0)

        if np.isclose(y_var, 0):
            return -np.inf
        else:
            # Log likelihood formula
            normality_check = (-n/2)*np.log(y_var) # Term to check normality
            jacobian_term = (self.current_lambda-1) * np.sum(np.log(self.data)) # Scales the normality check
            log_likelihood = normality_check + jacobian_term
            return log_likelihood

    def find_best_lambda(self):
        """Finds the optimal lambda by maximizing log-likelihood over a range
        :return: current best lambda value, current maximum log likelihood value, array of lambda values in the range, all there log_likelihoods and shift used.
        """
        if np.any(self.data<=0):
            min_val = np.min(self.data)
            shift = abs(min_val) + 0.001 # Example shift
            self.data = self.data + shift
            print(f"Shift of {min_val+0.001} is added to the values in x as some of them are non-positive")

        lambdas = np.linspace(self.lambda_range[0], self.lambda_range[1], self.lambda_step)
        log_likelihoods = []

        for lambda_ in lambdas:
            self.current_lambda = lambda_
            log_likelihood = self.box_cox_log_likelihood()
            log_likelihoods.append(log_likelihood)

        if np.all(np.isinf(log_likelihoods)):
            raise RuntimeError("All log-likelihoods were -ve infinity. Variance of transformed x is converged")

        best_lambda_idx = np.argmax(log_likelihoods)
        best_lambda = lambdas[best_lambda_idx]
        max_log_likelihood = log_likelihoods[best_lambda_idx]

        return best_lambda, max_log_likelihood, lambdas, log_likelihoods






