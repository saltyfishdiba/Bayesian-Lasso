import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


csv_path = r"D:/biostat article/single cell lab/Dryad/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
markers = [
    "MUC2", "SOX9", "MUC1", "CD31", "Synapto", "CD49f", "CD15", "CHGA", "CDX2", "ITLN1", "CD4", "CD127",
    "Vimentin", "HLADR", "CD8", "CD11c", "CD44", "CD16", "BCL2", "CD3", "CD123", "CD38", "CD90", "aSMA",
    "CD21", "NKG2D", "CD66", "CD57", "CD206", "CD68", "CD34", "aDef5", "CD7", "CD36", "CD138", "CD45RO",
    "Cytokeratin", "CD117", "CD19", "Podoplanin", "CD45", "CD56", "CD69", "Ki67", "CD49a", "CD163",
    "CD161", "OLFM4", "FAP", "CD25", "CollIV", "CK7", "MUC6"
]


data = pd.read_csv(csv_path, usecols=markers)


imputer = SimpleImputer(strategy="mean")
filtered_data = pd.DataFrame(imputer.fit_transform(data), columns=markers)

filtered_data = filtered_data.sample(n=min(1000, len(filtered_data)), random_state=42)


np.random.seed(42)
X = filtered_data.values
y = np.random.choice([0, 1], size=X.shape[0]) 
    #  Define the Bayesian lasso
with pm.Model() as bayesian_lasso:
    # Priors
    tau = pm.HalfCauchy("tau", beta=1)
    beta = pm.Laplace("beta", mu=0, b=tau, shape=X.shape[1])
    mu = pm.math.dot(X, beta)
    
    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=pm.math.sigmoid(mu), observed=y)
    
    # Inference using ADVI for speed
    approx = pm.fit(5000, method="advi")  
    trace = approx.sample(1000)


#  posterior means of beta coefficients
beta_means = trace.posterior["beta"].mean(dim=["chain", "draw"]).values.flatten()


most_significant_index = np.argmax(np.abs(beta_means))
most_significant_variable = markers[most_significant_index]
most_significant_value = beta_means[most_significant_index]


plt.figure(figsize=(14, 8))
plt.bar(range(len(beta_means)), beta_means, color="skyblue")
plt.axhline(0, color="red", linestyle="--", linewidth=1) 
plt.xticks(range(len(markers)), markers, rotation=90)  
plt.xlabel("Variables")
plt.ylabel("Posterior Mean of Coefficients (Beta)")
plt.title("Posterior Means of Coefficients (Beta) in Bayesian LASSO")


plt.text(
    most_significant_index,
    most_significant_value,
    f"Most Significant\n{most_significant_variable}\n({most_significant_value:.2f})",
    ha="center",
    va="bottom",
    color="blue",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"),
)
plt.tight_layout()
plt.show()


print(f"The most significant variable is {most_significant_variable} with a posterior mean of {most_significant_value:.2f}.")
