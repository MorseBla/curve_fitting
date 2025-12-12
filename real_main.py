import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd 
batch_size = 50


#load csv 
#benchmark,frequency_MHz,avg_time_ms,avg_power_W
csv_path = "results.csv"
df = pd.read_csv(csv_path)

#split each column from csv
benchmark = df["benchmark"].values.astype(str)
f = df["frequency_MHz"].values.astype(int)
e = df["avg_time_ms"].values.astype(float)
P = df["avg_power_W"].values.astype(float)

#find  f_max and e_i[f_max]
fmax = np.max(f)
at_fmax = (f == fmax)

unique_benchmarks = np.unique(benchmark)
e_fmax = {}
for b in unique_benchmarks:
    m = (benchmark == b) & at_fmax
    
    e_fmax[b] = float(e[m][0])

e_norm = np.array([e_fmax[b] / fmax for b in benchmark], dtype=float)



#create b_i
bench_to_idx = {b: i for i, b in enumerate(unique_benchmarks)}

task_idx = np.array([bench_to_idx[b] for b in benchmark], dtype=int)
T = len(unique_benchmarks)



# Step 3: Define the JCT model using single-run utilization
def power_model(x, a, *b):
    #P(f) = a * (e_i[fmax]/fmax)(f_i) + b
    #for each task i 
    f_i, e_norm_i, task_idx = x
    task_idx = task_idx.astype(int) 
    b = np.array(b)
    return a * e_norm_i * f_i + b[task_idx] 

def power_model2(x, a, b, *c):
    #P(f) = a * (e_i[fmax]/fmax)(f_i) + b
    #for each task i 
    f_i, e_norm_i, task_idx = x
    task_idx = task_idx.astype(int) 
    c = np.array(c)
    return (a * e_norm_i) * (f_i**2) + b * e_norm_i * f_i + c[task_idx] 
 
#Perform curve fitting to find the best a and b values
#p0 = [1.0] + [np.mean(P)] * T
p0 = [1.0] + [1.0] + [np.mean(P)] * T
#popt, _ = curve_fit(power_model, (f, e_norm, task_idx), P, p0=p0)
popt, _ = curve_fit(power_model2, (f, e_norm, task_idx), P, p0=p0)
#a_hat, b_hat = popt[0], popt[1:]
a_hat, b_hat, c_hat = popt[0], popt[1], popt[2:]
    
print(f"Optimal a: {a_hat}")
print(f"Optimal b: {b_hat}")
print(f"Optimal c: {c_hat}")
 
# Step 6: Visualize the fit using bars
#bar_width = 0.35
#alpha=0.2
#indices = np.arange(len(times_parallel))
 
# Calculate predicted times using the fitted model
#predicted_P = power_model((f, e_norm, task_idx), a_hat, *b_hat)
predicted_P = power_model2((f, e_norm, task_idx), a_hat, b_hat, *c_hat)
# Filter out any invalid values (inf, NaN) from actual and predicted times
#valid_indices = [
#    i for i in range(len(times_parallel)) 
#    if np.isfinite(times_parallel[i]) and np.isfinite(predicted_times[i])
#]
 
# Create filtered arrays for R² calculation
#filtered_actual = np.array(times_parallel)[valid_indices]
#filtered_predicted = np.array(predicted_times)[valid_indices]
 
# Print R² score for actual vs. predicted response times
if len(P) > 0 and len(predicted_P) > 0:
    r_squared = r2_score(P, predicted_P)
    print(f"R² Score: {r_squared:.4f}")
else:
    print("No valid data to compute R² score.")
 
# Plot bars for the actual and predicted times
for b in unique_benchmarks:
    m = (benchmark == b)

    f_b = f[m]
    P_b = P[m]
    Pp_b = predicted_P[m]

    # Sort by frequency for readability
    order = np.argsort(f_b)
    f_b = f_b[order]
    P_b = P_b[order]
    Pp_b = Pp_b[order]

    x = np.arange(len(f_b))
    width = 0.4
    

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, P_b, width, label="Measured")
    plt.bar(x + width/2, Pp_b, width, label="Predicted")

    plt.xticks(x, f_b, rotation=45)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (W)")
    plt.title(f"Power vs Frequency — {b}")
    plt.legend()

    plt.tight_layout()
    plt.show()


 
