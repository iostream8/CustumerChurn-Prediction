"""Step 1 - Make a realistic telecom customer dataset (7,000 customers).
Why synthetic? It runs anywhere with no download. If you have the real Kaggle
'Telco Customer Churn' CSV, you can use that instead (same kind of columns)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 7000

tenure = np.clip(rng.exponential(28, N).astype(int) + 1, 1, 72)          # months with company
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[.55, .21, .24])
internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[.34, .44, .22])

def addon(p):
    v = np.where(rng.random(N) < p, "Yes", "No")
    return np.where(internet == "No", "No internet service", v)

tech, security = addon(.35), addon(.35)
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N, p=[.34, .23, .22, .21])
paperless = rng.choice(["Yes", "No"], N, p=[.6, .4])
senior = rng.choice([0, 1], N, p=[.84, .16])
partner = rng.choice(["Yes", "No"], N)
dependents = rng.choice(["Yes", "No"], N, p=[.3, .7])
gender = rng.choice(["Male", "Female"], N)
phone = rng.choice(["Yes", "No"], N, p=[.9, .1])

base = np.select([internet == "No", internet == "DSL"], [20, 45], default=78)
monthly = np.round(base + (tech == "Yes") * 6 + (security == "Yes") * 5 + (phone == "Yes") * 5
                   + rng.normal(0, 6, N), 2).clip(18, None)
total = np.round(monthly * tenure * rng.uniform(.95, 1.05, N), 2)
calls = rng.poisson(np.where(contract == "Month-to-month", 1.6, 0.8))     # support calls in last 6 months

# Hidden "rules" that decide who leaves (the model must discover these on its own)
z = (1.1 * (contract == "Month-to-month") - 0.8 * (contract == "Two year") - 0.035 * tenure
     + 0.7 * (internet == "Fiber optic") + 0.012 * (monthly - 65) + 0.5 * (payment == "Electronic check")
     + 0.3 * (paperless == "Yes") - 0.5 * (tech == "Yes") - 0.4 * (security == "Yes")
     + 0.3 * calls + 0.25 * senior)
lo, hi = -10, 10                                   # tune intercept so about 26.5% of customers churn
for _ in range(50):
    mid = (lo + hi) / 2
    p = 1 / (1 + np.exp(-(z + mid)))
    lo, hi = (mid, hi) if p.mean() < 0.265 else (lo, mid)
churn = np.where(rng.random(N) < p, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(N)], "gender": gender, "SeniorCitizen": senior,
    "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "InternetService": internet, "OnlineSecurity": security, "TechSupport": tech, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": total, "support_calls": calls, "Churn": churn})
df.loc[rng.random(N) < 0.005, "TotalCharges"] = np.nan   # a few missing values on purpose (cleaning practice)
df.to_csv("data/customer_churn.csv", index=False)
print(df.shape, "| churn rate:", (df.Churn == "Yes").mean().round(3))
