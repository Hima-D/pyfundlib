import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def validate_strategy(returns: np.ndarray) -> dict:
    returns = np.array(returns).flatten()
    if len(returns) < 30:
        return {"verdict": "INSUFFICIENT DATA"}

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1

    t_stat, p_value = stats.ttest_1samp(returns, 0)
    significant = p_value < 0.01

    sr = sharpe
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns) + 3
    denominator = np.sqrt(1 + (skew * sr)/6 + ((kurt - 3) * sr**2)/24)
    psr = stats.norm.cdf(sr / denominator) if denominator > 0 else 0

    trials = 100
    dsr = sharpe * (1 - stats.norm.ppf(0.95) * np.sqrt((1 - 0.5)/trials))

    result = {
        "cagr_%": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "p_value": f"{p_value:.2e}",
        "significant": significant,
        "probability_of_skill_%": round(psr * 100, 3),
        "deflated_sharpe": round(dsr, 3),
        "observations": len(returns),
        "verdict": "REAL ALPHA — SCIENTIFICALLY PROVEN" if (significant and psr > 0.95) else "LIKELY LUCK"
    }
    return result

def print_validation(returns):
    r = validate_strategy(returns)
    print("\n" + " SCIENTIFIC VALIDATION ".center(60, "="))
    for k, v in r.items():
        if k != "verdict":
            print(f"{k:25}: {v}")
    print(f"\nVERDICT → {r['verdict']}")
    print("="*60)
