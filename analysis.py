# analysis.py
import pandas as pd
from scipy.stats import pearsonr


def compute_correlations(df):
    """
    Accepts a pandas DataFrame with columns:
    ['workout_minutes','calories_intake','sleep_hours','mood_score']
    Returns: (corr_matrix, corr_details) or (None, None) if not enough data.
    """
    if df is None or df.empty or len(df) < 3:
        return None, None

    numeric_cols = ["workout_minutes", "calories_intake", "sleep_hours", "mood_score"]
    # keep only required numeric columns, coerce errors to NaN
    df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # if not enough paired rows, bail out
    paired_rows = df_num.dropna()
    if paired_rows.shape[0] < 3:
        return None, None

    # Pearson correlation matrix
    corr_matrix = df_num.corr(method="pearson")

    # compute pairwise r and p-values (only for unique pairs)
    corr_details = []
    cols = numeric_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            x = df_num[c1].dropna()
            y = df_num[c2].dropna()
            # pair them
            paired = pd.concat([x, y], axis=1).dropna()
            if len(paired) > 2:
                try:
                    r, p = pearsonr(paired[c1], paired[c2])
                except Exception:
                    r, p = 0.0, 1.0
                corr_details.append(
                    {
                        "var1": c1,
                        "var2": c2,
                        "r_value": round(r, 2),
                        "p_value": round(p, 4),
                        "n": int(len(paired)),
                    }
                )

    # sort by absolute r value descending for convenience
    corr_details = sorted(corr_details, key=lambda x: abs(x["r_value"]), reverse=True)
    return corr_matrix, corr_details
