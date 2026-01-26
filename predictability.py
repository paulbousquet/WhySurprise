import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

def generate_data(T=300, K=8000, seed=42):
    np.random.seed(seed)
    
    theta_true = 2.0
    
    # Common structural shocks
    u = np.random.randn(T, 3)  # e.g., demand, supply, monetary
    
    # X loads on common shocks
    X = np.random.randn(T, K)
    loadings = np.random.randn(K, 3) * 0.3
    X += u @ loadings.T
    
    D = (0.5*X[:,0] + 0.3*X[:,1] - 0.4*X[:,2] + 
         0.3*X[:,0]*X[:,1] + 0.2*np.sin(X[:,3]) + 0.5*np.random.randn(T))
    
    # Y loads on same shocks directly
    Y = (theta_true*D + 0.4*X[:,0] + 0.3*X[:,1]**2 - 0.5*X[:,2] + 
         0.6*u[:,0] + 0.4*u[:,1] + np.random.randn(T)+u[:,2])
    
    return Y, D, X, theta_true

def dml_lasso(Y, D, X, n_folds=5):
    T = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
    Y_tilde, D_tilde = np.zeros(T), np.zeros(T)
    
    for train_idx, test_idx in kf.split(X):
        lasso_y = LassoCV(cv=3, n_jobs=-1, max_iter=2000).fit(X[train_idx], Y[train_idx])
        lasso_d = LassoCV(cv=3, n_jobs=-1, max_iter=2000).fit(X[train_idx], D[train_idx])
        Y_tilde[test_idx] = Y[test_idx] - lasso_y.predict(X[test_idx])
        D_tilde[test_idx] = D[test_idx] - lasso_d.predict(X[test_idx])
    
    theta_hat = np.sum(D_tilde * Y_tilde) / np.sum(D_tilde**2)
    se = np.sqrt(np.mean((Y_tilde - theta_hat*D_tilde)**2) / np.sum(D_tilde**2))
    return theta_hat, se, Y_tilde, D_tilde

def ols(y, X):
    X = np.c_[np.ones(len(y)), X]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s2 = np.sum(resid**2) / (len(y) - X.shape[1])
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    return beta, beta / se

# Run
Y, D, X, theta_true = generate_data(T=300, K=8000)
print(f"True theta: {theta_true}")

theta_hat, se, Y_tilde, D_tilde = dml_lasso(Y, D, X)
print(f"DML estimate: {theta_hat:.3f} (SE: {se:.3f})")
def run_diagnostics(resid, X, n_regs=500, seed=999):
    np.random.seed(seed)
    sig_counts = np.zeros(n_regs, dtype=int)
    
    for i in range(n_regs):
        rand_idx = np.random.choice(range(3, X.shape[1]), 2, replace=False)
        idx = np.concatenate([[0, 1, 2], rand_idx])
        _, t = ols(resid, X[:, idx])
        sig_counts[i] = np.sum(np.abs(t[1:]) > 1.96)
    
    return sig_counts

print("\n--- Diagnostic: % of regressions with k significant vars (p<0.05) ---")
print("(First 3 vars fixed as X[:,0:3], plus 2 random)")
for resid, name in [(Y_tilde, "Y_tilde"), (D_tilde, "D_tilde")]:
    counts = run_diagnostics(resid, X, n_regs=500)
    print(f"\n{name}:")
    for k in range(6):
        pct = 100 * np.mean(counts >= k)
        print(f"  >= {k} sig: {pct:.1f}%")

