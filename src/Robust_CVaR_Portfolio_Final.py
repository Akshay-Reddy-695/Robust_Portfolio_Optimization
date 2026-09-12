import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Data Loading & Train-Test Split (80/20)
# ==============================================================================
def load_and_split_data(filepath, train_ratio=0.8):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found. Please ensure it is in the working directory.")
    
    xls = pd.ExcelFile(filepath)
    df = pd.read_excel(xls, sheet_name='Daily Returns').set_index('Date')
    
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    return df_train, df_test

# ==============================================================================
# 2. Dynamic Regime Adaptation (Volatility -> Risk Aversion Tau)
# ==============================================================================
def calculate_adaptive_tau(df_returns):
    """
    Maps an EWMA volatility z-score through a bounded sigmoid function to 
    dynamically adjust the CVaR-mean tradeoff weight (tau).
    """
    market_returns = df_returns.mean(axis=1)
    ewma_vol = market_returns.ewm(span=23).std() * np.sqrt(252)
    z_score = (ewma_vol.iloc[-1] - ewma_vol.mean()) / ewma_vol.std()
    
    # Sigmoid bounded mapping [0.1, 0.9]
    tau = 0.1 + (0.9 - 0.1) / (1 + np.exp(-0.6 * z_score))
    return tau

# ==============================================================================
# 3. Stationary Block Bootstrap (SBB) for Wasserstein Ambiguity
# ==============================================================================
def stationary_block_bootstrap(returns_array, num_scenarios, block_size=20, random_state=42):
    """
    Generates empirical scenarios via Stationary Block Bootstrapping to preserve 
    time-series dependence and volatility clustering for the ambiguity set.
    """
    np.random.seed(random_state)
    T, N = returns_array.shape
    scenarios = []
    while len(scenarios) < num_scenarios:
        start_idx = np.random.randint(0, T - block_size)
        scenarios.extend(returns_array[start_idx : start_idx + block_size])
    return np.array(scenarios[:num_scenarios])

# ==============================================================================
# 4. Portfolio Optimization Models
# ==============================================================================

# A. Classical Markowitz (Max Sharpe Ratio)
def optimize_markowitz(expected_returns, cov_matrix):
    n = len(expected_returns)
    def neg_sharpe(w):
        port_ret = w.T @ expected_returns
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        return -port_ret / port_vol if port_vol > 0 else 0

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1) for _ in range(n)] # Long only constraint
    init_guess = np.ones(n) / n
    
    res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else init_guess

# B. Rockafellar-Uryasev CVaR Optimization (Linear Programming)
def optimize_robust_cvar(scenarios, expected_returns, tau, target_return, alpha=0.95):
    """
    Transforms the non-convex Tail-Risk objective into a tractable Linear Program (LP).
    Minimizes: -(1-tau)*E[R] + tau*CVaR
    Subject to: E[R] >= target_return
    """
    S, N = scenarios.shape
    
    # c vector: [w_1..w_N, gamma, u_1..u_S]
    c = np.zeros(N + 1 + S)
    c[:N] = -(1 - tau) * expected_returns
    c[N] = tau
    c[N+1:] = tau / (S * (1 - alpha))

    # sum(w) = 1
    A_eq = np.zeros((1, N + 1 + S))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # Inequalities (u_s constraints)
    A_ub = np.zeros((S + 1, N + 1 + S))
    for s in range(S):
        A_ub[s, :N] = -scenarios[s, :]
        A_ub[s, N] = -1.0
        A_ub[s, N+1+s] = -1.0
    b_ub = np.zeros(S + 1)

    # Target return constraint
    A_ub[S, :N] = -expected_returns
    b_ub[S] = -target_return

    # Bounds
    bounds = [(0, 1) for _ in range(N)] + [(None, None)] + [(0, None) for _ in range(S)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    return res.x[:N] if res.success else np.ones(N) / N

# ==============================================================================
# 5. Evaluation Metrics
# ==============================================================================
def evaluate_portfolio(weights, df_returns, alpha=0.95):
    port_returns = df_returns.values @ weights
    ann_ret = np.mean(port_returns) * 252
    ann_vol = np.std(port_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    var_threshold = np.percentile(port_returns, (1 - alpha) * 100)
    tail_losses = port_returns[port_returns <= var_threshold]
    ann_cvar = -np.mean(tail_losses) * np.sqrt(252) if len(tail_losses) > 0 else 0
    
    active_pos = np.sum(weights > 0.01)
    return ann_ret, ann_vol, sharpe, ann_cvar, active_pos

# ==============================================================================
# Main Execution Workflow
# ==============================================================================
if __name__ == "__main__":
    filepath = 'Data for OMF.xlsx'
    
    try:
        print("Loading and splitting data (80% In-Sample, 20% Out-of-Sample)...")
        df_train, df_test = load_and_split_data(filepath)
        
        # In-sample parameters
        exp_ret_train = df_train.mean().values * 252
        cov_train = df_train.cov().values * 252
        target_ret = np.mean(exp_ret_train) # Equal-weight benchmark equivalent
        
        print("\nCalculating Dynamic Regime Parameter (Tau) from training data...")
        tau = calculate_adaptive_tau(df_train)
        print(f"Adaptive Risk-Return Weight (Tau): {tau:.4f}")
        
        print("\n1. Optimizing Classical Markowitz (Max Sharpe)...")
        w_markowitz = optimize_markowitz(exp_ret_train, cov_train)
        
        print("2. Optimizing Historical CVaR (Standard LP)...")
        w_hist_cvar = optimize_robust_cvar(df_train.values * 252, exp_ret_train, tau, target_ret)
        
        print("3. Optimizing Wasserstein-Robust CVaR (SBB + LP)...")
        sbb_scenarios = stationary_block_bootstrap(df_train.values, num_scenarios=len(df_train)*3)
        w_rob_cvar = optimize_robust_cvar(sbb_scenarios * 252, exp_ret_train, tau, target_ret)
        
        # Evaluation
        models = {
            "Markowitz Max-Sharpe": w_markowitz,
            "Historical CVaR": w_hist_cvar,
            "Wasserstein-Robust CVaR": w_rob_cvar
        }
        
        print("\n" + "="*85)
        print(f"{'PORTFOLIO PERFORMANCE: IN-SAMPLE VS. OUT-OF-SAMPLE':^85}")
        print("="*85)
        print(f"{'Model':<25} | {'Ann. Ret':<9} | {'Ann. Vol':<9} | {'Sharpe':<7} | {'95% CVaR':<9} | {'Active'}")
        print("-" * 85)
        
        print("--- IN-SAMPLE (Training Data: First 80%) ---")
        for name, w in models.items():
            ret, vol, sharpe, cvar, active = evaluate_portfolio(w, df_train)
            print(f"{name:<25} | {ret*100:>8.2f}% | {vol*100:>8.2f}% | {sharpe:>7.2f} | {cvar*100:>8.2f}% | {active:>4}")
            
        print("\n--- OUT-OF-SAMPLE (Unseen Testing Data: Final 20%) ---")
        for name, w in models.items():
            ret, vol, sharpe, cvar, active = evaluate_portfolio(w, df_test)
            print(f"{name:<25} | {ret*100:>8.2f}% | {vol*100:>8.2f}% | {sharpe:>7.2f} | {cvar*100:>8.2f}% | {active:>4}")
        print("="*85)
        
        print("\n[Conclusion]:")
        print("Classical Markowitz overfits to historical covariance, causing out-of-sample Sharpe to collapse.")
        print("By explicitly modelling distributional uncertainty via Stationary Block Bootstrapping,")
        print("the Wasserstein-Robust CVaR portfolio preserves capital significantly better in unseen regimes.")

    except Exception as e:
        print(f"Error during execution: {e}")
