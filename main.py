import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#calculate R^2
def calc_r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

#models
def model_linear(f, a, b):
    return a*f + b

def model_quadratic(f, a, b, c):
    return a*(f**2) + b*f + c

def model_inverse(f, a, b):
    return a/f + b

def model_inv_square(f, a, b):
    return a/(f**2) + b

#linear and inverse model
def model_mixed(f, a, b, c):
    return a/f + b*f + c

#fit model 
def fit_model(model_function, f, P):
    popt, _ = curve_fit(model_function, f, P) # maxfev=20000
    P_pred = model_function(f, *popt)
    r2 = calc_r2(P, P_pred)
    return popt, r2

# load CSV and fit models
def run_fits(csv_path):
    df = pd.read_csv(csv_path)

    f = df["frequency_MHz"].values.astype(float)
    P = df["avg_power_W"].values.astype(float)

    models = {
        "linear": model_linear,
        "quadratic": model_quadratic,
        "inverse": model_inverse,
        "inverse_square": model_inv_square,
        "mixed": model_mixed
    }

    results = {}

    for name, model in models.items():
        try:
            popt, r2 = fit_model(model, f, P)
            results[name] = {
                "params": popt,
                "r2": r2
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results

# Example usage
if __name__ == "__main__":
    results = run_fits("results2.csv")

    for name, info in results.items():
        print(name.upper())
        if "error" in info:
            print("  error:", info["error"])
        else:
            print("  Params:", info["params"])
            print("  R²:", info["r2"])
        print()

