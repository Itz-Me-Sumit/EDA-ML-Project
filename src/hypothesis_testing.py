import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from itertools import combinations

# Statistical tests
from scipy import stats
from scipy.stats import (
    ttest_ind,          # Independent samples T-Test
    f_oneway,           # One-way ANOVA
    chi2_contingency,   # Chi-Square test
    mannwhitneyu,       # Mann-Whitney U (non-parametric alt to T-Test)
    kruskal,            # Kruskal-Wallis (non-parametric alt to ANOVA)
    shapiro,            # Shapiro-Wilk normality test
    levene,             # Levene's test for equal variances
)

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
try:
    from src.config import PROCESSED_DATA_DIR
    REPORTS_DIR = Path(str(PROCESSED_DATA_DIR)).parent / "reports" / "hypothesis"
except ImportError:
    PROCESSED_DATA_DIR = Path("data/processed")
    REPORTS_DIR = Path("reports/hypothesis")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class HypothesisTester:
    """
    Day 6 — Hypothesis Testing: Statistically prove/disprove business questions.

    Pipeline Steps:
        1. load_data()              - Load featured_dataset.csv
        2. check_normality()        - Shapiro-Wilk test to choose parametric vs non-parametric
        3. run_ttest()              - Independent T-Test (parametric, 2 groups)
        4. run_mannwhitney()        - Mann-Whitney U (non-parametric, 2 groups)
        5. run_anova()              - One-way ANOVA (parametric, 3+ groups)
        6. run_kruskal()            - Kruskal-Wallis (non-parametric, 3+ groups)
        7. run_chi_square()         - Chi-Square test (categorical vs categorical)
        8. compute_effect_size()    - Cohen's d (T-Test), eta-squared (ANOVA)
        9. run_business_questions() - Pre-defined Olist business hypotheses
        10. plot_results()          - Visualize test outcomes
        11. save_report()           - Save all findings to reports/hypothesis/

    ─────────────────────────────────────────────────────────────────────────
    MATH CHEATSHEET
    ─────────────────────────────────────────────────────────────────────────

    NULL HYPOTHESIS (H₀) vs ALTERNATIVE HYPOTHESIS (H₁):
        H₀: "There is NO significant difference between groups"
        H₁: "There IS a significant difference between groups"
        If p < α (usually 0.05) → Reject H₀ → Difference is statistically significant

    p-VALUE INTUITION (not just < 0.05):
        p-value = probability of observing this result (or more extreme)
                  IF the null hypothesis were true.
        p = 0.03 → "Only 3% chance this result is random noise"
        p = 0.5  → "50% chance — could easily be random"
        ⚠️  p-value does NOT tell you effect SIZE. Always pair with Cohen's d / η².

    T-TEST (Independent Samples):
        t = (mean₁ - mean₂) / sqrt(s²_pooled x (1/n₁ + 1/n₂))
        Assumptions: normality, equal variance (Levene's test), independence.
        Use when: comparing a continuous variable across 2 groups.
        Example: "Is delivery_delay_days different between same-state vs cross-state?"

    MANN-WHITNEY U (Non-parametric alt to T-Test):
        Ranks all values from both groups together, then compares rank sums.
        U = n₁n₂ + n₁(n₁+1)/2 - R₁   where R₁ = sum of ranks for group 1
        Use when: data is NOT normally distributed (most Olist columns!).
        Example: "Is review_score different between late vs on-time deliveries?"

    ONE-WAY ANOVA:
        F = (Variance BETWEEN groups) / (Variance WITHIN groups)
        F = [SS_between/(k-1)] / [SS_within/(N-k)]
        Use when: comparing a continuous variable across 3+ groups.
        Example: "Do review scores differ across payment types?"
        Assumption: normality + homoscedasticity per group.

    KRUSKAL-WALLIS (Non-parametric alt to ANOVA):
        Rank-based extension of Mann-Whitney to 3+ groups.
        H = [12 / N(N+1)] x Σ[Rᵢ²/nᵢ] - 3(N+1)
        Use when ANOVA assumptions are violated (very common in e-commerce data).

    CHI-SQUARE TEST:
        χ² = Σ [(Observed - Expected)² / Expected]
        Use when: both variables are CATEGORICAL.
        Example: "Is payment type associated with order status?"
        Assumption: expected cell count ≥ 5 in 80% of cells.

    COHEN'S d (Effect Size for T-Test):
        d = (mean₁ - mean₂) / pooled_std
        0.2 → small,  0.5 → medium,  0.8 → large
        Critical: a statistically significant result with d=0.1 is practically useless.

    ETA-SQUARED η² (Effect Size for ANOVA):
        η² = SS_between / SS_total
        0.01 → small,  0.06 → medium,  0.14 → large

    TYPE I ERROR (α): Rejecting H₀ when it's true (false positive). Set α = 0.05.
    TYPE II ERROR (β): Failing to reject H₀ when H₁ is true (false negative).
    ─────────────────────────────────────────────────────────────────────────
    """

    # Significance level
    ALPHA = 0.05

    # Olist-specific business questions to test
    BUSINESS_QUESTIONS = [
        {
            "id":          "BQ1",
            "description": "Late deliveries lower review scores",
            "test":        "mannwhitney",
            "group_col":   "is_late_delivery",
            "metric_col":  "review_score",
            "H0":          "Review scores are equal for late vs on-time deliveries",
            "H1":          "Late deliveries have significantly lower review scores",
        },
        {
            "id":          "BQ2",
            "description": "Delivery delay varies by customer state",
            "test":        "kruskal",
            "group_col":   "customer_state",
            "metric_col":  "delivery_delay_days",
            "H0":          "Delivery delay is equal across all states",
            "H1":          "At least one state has significantly different delivery delay",
            "top_n_groups": 10,   # limit to top-10 states by order count
        },
        {
            "id":          "BQ3",
            "description": "Same-state orders are faster than cross-state",
            "test":        "mannwhitney",
            "group_col":   "is_same_state",
            "metric_col":  "actual_delivery_days",
            "H0":          "Delivery days are equal for same-state vs cross-state",
            "H1":          "Same-state orders are delivered faster",
        },
        {
            "id":          "BQ4",
            "description": "Review scores differ across payment types",
            "test":        "kruskal",
            "group_col":   "payment_type",
            "metric_col":  "review_score",
            "H0":          "Review scores are equal across all payment types",
            "H1":          "At least one payment type has different review scores",
        },
        {
            "id":          "BQ5",
            "description": "Long-distance orders have higher freight cost",
            "test":        "mannwhitney",
            "group_col":   "is_long_distance",
            "metric_col":  "freight_value",
            "H0":          "Freight value is equal for long vs short distance orders",
            "H1":          "Long-distance orders have significantly higher freight",
        },
        {
            "id":          "BQ6",
            "description": "Weekend purchases have different review scores",
            "test":        "mannwhitney",
            "group_col":   "is_weekend_purchase",
            "metric_col":  "review_score",
            "H0":          "Review scores are equal for weekend vs weekday purchases",
            "H1":          "Weekend purchases have different review scores",
        },
        {
            "id":          "BQ7",
            "description": "Payment type and order status are associated",
            "test":        "chi_square",
            "col1":        "payment_type",
            "col2":        "order_status",
            "H0":          "Payment type and order status are independent",
            "H1":          "Payment type and order status are associated",
        },
    ]

    def __init__(self, filename: str = "featured_dataset.csv"):
        self.filepath = Path(str(PROCESSED_DATA_DIR)) / filename
        self.df: pd.DataFrame | None = None
        self.results: list[dict] = []          # all test results accumulated here
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

    def _section(self, title: str) -> None:
        print(f"\n{'═' * 65}")
        print(f"  {title}")
        print(f"{'═' * 65}")

    def _verdict(self, p_value: float) -> str:
        """Return formatted verdict string based on p-value."""
        if p_value < 0.001:
            return f"✅ REJECT H₀  (p={p_value:.2e} *** highly significant)"
        elif p_value < 0.01:
            return f"✅ REJECT H₀  (p={p_value:.4f} ** very significant)"
        elif p_value < self.ALPHA:
            return f"✅ REJECT H₀  (p={p_value:.4f} * significant)"
        else:
            return f"❌ FAIL TO REJECT H₀  (p={p_value:.4f}, not significant)"

    def _effect_label(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        d = abs(d)
        if d >= 0.8:   return "Large"
        elif d >= 0.5: return "Medium"
        elif d >= 0.2: return "Small"
        else:          return "Negligible"

    def _eta_label(self, eta: float) -> str:
        """Interpret η² magnitude."""
        if eta >= 0.14:  return "Large"
        elif eta >= 0.06: return "Medium"
        elif eta >= 0.01: return "Small"
        else:             return "Negligible"

    # ── Step 1: Load ─────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """Load featured_dataset.csv."""
        self._section("Step 1 — Loading Data")

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.filepath}\n"
                "Run Day 3 feature engineering pipeline first."
            )

        self.df = pd.read_csv(self.filepath)

        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        print(f"  Path    : {self.filepath}")
        print(f"  Shape   : {self.df.shape}")
        print(f"  Columns : {list(self.df.columns[:8])} ...")
        print(f"\n  Available binary group columns:")
        for col in ["is_late_delivery", "is_same_state", "is_long_distance", "is_weekend_purchase"]:
            if col in self.df.columns:
                vc = self.df[col].value_counts()
                print(f"    {col:<35} {dict(vc)}")

    # ── Step 2: Normality Check ───────────────────────────────────────────────

    def check_normality(
        self,
        col: str,
        group_col: str | None = None,
        sample_size: int = 3000,
    ) -> dict:
        """
        Shapiro-Wilk normality test.

        Why this matters:
            T-Test and ANOVA assume normality.
            If data is NOT normal → use Mann-Whitney U or Kruskal-Wallis.
            Shapiro-Wilk is most powerful for n < 5000.
            For larger samples, normality violations are common — use non-parametric.

        Parameters
        ----------
        col         : Column to test for normality
        group_col   : Optional — test each group separately
        sample_size : Shapiro-Wilk max recommended is 5000; we sample to keep it fast

        Returns
        -------
        dict: {col, statistic, p_value, is_normal, recommendation}
        """
        self._ensure_loaded()

        if col not in self.df.columns:
            return {"col": col, "error": "Column not found"}

        series = self.df[col].dropna()

        # Sample for speed (Shapiro-Wilk is O(n log n) but slow for n > 5000)
        if len(series) > sample_size:
            series = series.sample(n=sample_size, random_state=42)

        stat, p = shapiro(series)
        is_normal = p > self.ALPHA

        result = {
            "col":            col,
            "n":              len(series),
            "statistic":      round(stat, 4),
            "p_value":        round(p, 6),
            "is_normal":      is_normal,
            "recommendation": "T-Test / ANOVA" if is_normal else "Mann-Whitney U / Kruskal-Wallis",
        }

        print(f"  Shapiro-Wilk [{col}]: W={stat:.4f}, p={p:.4f} "
              f"→ {'Normal ✅' if is_normal else 'NOT Normal ⚠️  → use non-parametric'}")

        return result

    # ── Step 3: T-Test ────────────────────────────────────────────────────────

    def run_ttest(
        self,
        group_col: str,
        metric_col: str,
        equal_var: bool | None = None,
        description: str = "",
    ) -> dict:
        """
        Independent samples T-Test.

        t = (μ₁ - μ₂) / SE_pooled

        When to use:
            - Continuous metric_col
            - Binary group_col (exactly 2 groups)
            - Metric is approximately normal (check with check_normality first)
            - If normality fails → use run_mannwhitney() instead

        equal_var: None → auto-detect via Levene's test (recommended)
                   True → Student's T-Test (assumes equal variance)
                   False → Welch's T-Test (safer, does not assume equal variance)

        Parameters
        ----------
        group_col   : Binary column (0/1 or two distinct values)
        metric_col  : Continuous column to compare
        equal_var   : None = auto, True = Student's, False = Welch's
        description : Human-readable label for reports

        Returns
        -------
        dict with test results and Cohen's d effect size
        """
        self._ensure_loaded()

        if group_col not in self.df.columns:
            raise ValueError(f"group_col '{group_col}' not found.")
        if metric_col not in self.df.columns:
            raise ValueError(f"metric_col '{metric_col}' not found.")

        groups = self.df[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(
                f"T-Test requires exactly 2 groups, found {len(groups)}: {groups}"
            )

        group_vals = sorted(groups)
        g1 = self.df.loc[self.df[group_col] == group_vals[0], metric_col].dropna()
        g2 = self.df.loc[self.df[group_col] == group_vals[1], metric_col].dropna()

        # Auto-detect equal variance via Levene's test
        if equal_var is None:
            _, levene_p = levene(g1, g2)
            equal_var = levene_p > self.ALPHA  # True if variances are equal
            test_type = "Student's T" if equal_var else "Welch's T"
        else:
            test_type = "Student's T" if equal_var else "Welch's T"

        t_stat, p_value = ttest_ind(g1, g2, equal_var=equal_var)

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
            / (len(g1) + len(g2) - 2)
        )
        cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

        result = {
            "test":          f"{test_type} T-Test",
            "description":   description or f"{metric_col} by {group_col}",
            "group_col":     group_col,
            "metric_col":    metric_col,
            "group_0":       group_vals[0],
            "group_1":       group_vals[1],
            "mean_0":        round(g1.mean(), 4),
            "mean_1":        round(g2.mean(), 4),
            "n_0":           len(g1),
            "n_1":           len(g2),
            "t_statistic":   round(t_stat, 4),
            "p_value":       round(p_value, 6),
            "significant":   p_value < self.ALPHA,
            "cohens_d":      round(cohens_d, 4),
            "effect_size":   self._effect_label(cohens_d),
        }

        self._print_test_result(result)
        self.results.append(result)
        return result

    # ── Step 4: Mann-Whitney U ────────────────────────────────────────────────

    def run_mannwhitney(
        self,
        group_col: str,
        metric_col: str,
        description: str = "",
        H0: str = "",
        H1: str = "",
    ) -> dict:
        """
        Mann-Whitney U Test (non-parametric alternative to T-Test).

        Tests whether one group tends to have larger values than another,
        WITHOUT assuming normality.

        U = n₁n₂ + n₁(n₁+1)/2 - R₁
        where R₁ = sum of ranks for group 1 (after ranking all combined values)

        Effect size: r = Z / sqrt(N)
            r ≥ 0.1 → small,  r ≥ 0.3 → medium,  r ≥ 0.5 → large

        Parameters
        ----------
        group_col   : Binary column (2 groups)
        metric_col  : Continuous or ordinal column to compare
        description : Human-readable label

        Returns
        -------
        dict with U statistic, p-value, and rank-biserial effect size
        """
        self._ensure_loaded()

        if group_col not in self.df.columns:
            raise ValueError(f"group_col '{group_col}' not found.")
        if metric_col not in self.df.columns:
            print(f"  ⚠️  Skipping: metric_col '{metric_col}' not found in dataset.")
            return {"test": "Mann-Whitney U", "error": f"{metric_col} not found"}

        groups = self.df[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"Mann-Whitney requires 2 groups, found {len(groups)}")

        group_vals = sorted(groups)
        g1 = self.df.loc[self.df[group_col] == group_vals[0], metric_col].dropna()
        g2 = self.df.loc[self.df[group_col] == group_vals[1], metric_col].dropna()

        u_stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")

        # Rank-biserial correlation as effect size (r = 1 - 2U/n1n2)
        n1, n2 = len(g1), len(g2)
        effect_r = 1 - (2 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0

        result = {
            "test":          "Mann-Whitney U",
            "description":   description or f"{metric_col} by {group_col}",
            "H0":            H0,
            "H1":            H1,
            "group_col":     group_col,
            "metric_col":    metric_col,
            "group_0":       group_vals[0],
            "group_1":       group_vals[1],
            "median_0":      round(g1.median(), 4),
            "median_1":      round(g2.median(), 4),
            "mean_0":        round(g1.mean(), 4),
            "mean_1":        round(g2.mean(), 4),
            "n_0":           n1,
            "n_1":           n2,
            "u_statistic":   round(u_stat, 2),
            "p_value":       round(p_value, 6),
            "significant":   p_value < self.ALPHA,
            "effect_r":      round(effect_r, 4),
            "effect_size":   (
                "Large"  if abs(effect_r) >= 0.5 else
                "Medium" if abs(effect_r) >= 0.3 else
                "Small"  if abs(effect_r) >= 0.1 else "Negligible"
            ),
        }

        self._print_test_result(result)
        self.results.append(result)
        return result

    # ── Step 5: One-Way ANOVA ─────────────────────────────────────────────────

    def run_anova(
        self,
        group_col: str,
        metric_col: str,
        top_n_groups: int | None = None,
        description: str = "",
        H0: str = "",
        H1: str = "",
    ) -> dict:
        """
        One-way ANOVA: compare a continuous metric across 3+ groups.

        F = MS_between / MS_within
          = [SS_between / (k-1)] / [SS_within / (N-k)]

        Post-hoc: if ANOVA is significant, run Tukey HSD to find WHICH
        groups differ. (Covered here as group mean summary.)

        Assumptions:
            1. Independence of observations
            2. Normality within each group
            3. Homoscedasticity (equal variances) — check with Levene's

        Parameters
        ----------
        group_col     : Categorical column with 3+ groups
        metric_col    : Continuous metric to compare
        top_n_groups  : Limit to top-N groups by count (avoids tiny groups)

        Returns
        -------
        dict with F statistic, p-value, eta-squared effect size
        """
        self._ensure_loaded()

        if group_col not in self.df.columns:
            raise ValueError(f"group_col '{group_col}' not found.")
        if metric_col not in self.df.columns:
            print(f"  ⚠️  Skipping: metric_col '{metric_col}' not found.")
            return {"test": "ANOVA", "error": f"{metric_col} not found"}

        # Optionally limit to top-N groups by order count
        if top_n_groups:
            top_groups = (
                self.df[group_col].value_counts()
                .head(top_n_groups)
                .index.tolist()
            )
            df_subset = self.df[self.df[group_col].isin(top_groups)]
        else:
            df_subset = self.df

        group_data = [
            df_subset.loc[df_subset[group_col] == g, metric_col].dropna().values
            for g in df_subset[group_col].dropna().unique()
        ]
        group_data = [g for g in group_data if len(g) >= 5]   # skip tiny groups

        if len(group_data) < 3:
            print(f"  ⚠️  ANOVA requires 3+ groups with n≥5. Found only {len(group_data)}. Skipping.")
            return {"test": "ANOVA", "error": "Not enough groups"}

        f_stat, p_value = f_oneway(*group_data)

        # Eta-squared effect size
        all_vals  = np.concatenate(group_data)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_data)
        ss_total   = sum((v - grand_mean) ** 2 for v in all_vals)
        eta_sq     = ss_between / ss_total if ss_total > 0 else 0.0

        # Group means summary (top 5 highest and lowest)
        group_means = (
            df_subset.groupby(group_col)[metric_col]
            .agg(["mean", "median", "count"])
            .sort_values("mean", ascending=False)
        )

        result = {
            "test":          "One-Way ANOVA",
            "description":   description or f"{metric_col} by {group_col}",
            "H0":            H0,
            "H1":            H1,
            "group_col":     group_col,
            "metric_col":    metric_col,
            "n_groups":      len(group_data),
            "f_statistic":   round(f_stat, 4),
            "p_value":       round(p_value, 6),
            "significant":   p_value < self.ALPHA,
            "eta_squared":   round(eta_sq, 4),
            "effect_size":   self._eta_label(eta_sq),
            "group_means":   group_means,
        }

        self._print_test_result(result)
        self.results.append(result)
        return result

    # ── Step 6: Kruskal-Wallis ────────────────────────────────────────────────

    def run_kruskal(
        self,
        group_col: str,
        metric_col: str,
        top_n_groups: int | None = None,
        description: str = "",
        H0: str = "",
        H1: str = "",
    ) -> dict:
        """
        Kruskal-Wallis H Test (non-parametric alternative to ANOVA).

        Rank-based: compare medians across 3+ groups without normality assumption.
        H = [12 / N(N+1)] × Σᵢ[Rᵢ²/nᵢ] - 3(N+1)

        Use when:
            - ANOVA assumptions (normality / equal variance) are violated
            - Ordinal outcome (like review_score 1-5)
            - Small group sizes

        Effect size: ε² = (H - k + 1) / (N - k)
            Interpreted same as η²: 0.01 small, 0.06 medium, 0.14 large

        Parameters
        ----------
        group_col     : Categorical column with 3+ groups
        metric_col    : Continuous or ordinal metric
        top_n_groups  : Limit to top-N groups by count

        Returns
        -------
        dict with H statistic, p-value, epsilon-squared effect size
        """
        self._ensure_loaded()

        if group_col not in self.df.columns:
            raise ValueError(f"group_col '{group_col}' not found.")
        if metric_col not in self.df.columns:
            print(f"  ⚠️  Skipping: metric_col '{metric_col}' not found.")
            return {"test": "Kruskal-Wallis", "error": f"{metric_col} not found"}

        if top_n_groups:
            top_groups = (
                self.df[group_col].value_counts()
                .head(top_n_groups)
                .index.tolist()
            )
            df_subset = self.df[self.df[group_col].isin(top_groups)]
        else:
            df_subset = self.df

        groups_unique = df_subset[group_col].dropna().unique()
        group_data = [
            df_subset.loc[df_subset[group_col] == g, metric_col].dropna().values
            for g in groups_unique
        ]
        group_data = [g for g in group_data if len(g) >= 5]

        if len(group_data) < 3:
            print(f"  ⚠️  Kruskal requires 3+ groups with n≥5. Found {len(group_data)}.")
            return {"test": "Kruskal-Wallis", "error": "Not enough groups"}

        h_stat, p_value = kruskal(*group_data)

        # Epsilon-squared effect size
        N = sum(len(g) for g in group_data)
        k = len(group_data)
        epsilon_sq = (h_stat - k + 1) / (N - k) if (N - k) > 0 else 0.0

        # Group medians summary
        group_medians = (
            df_subset.groupby(group_col)[metric_col]
            .agg(["median", "mean", "count"])
            .sort_values("median", ascending=False)
        )

        result = {
            "test":          "Kruskal-Wallis",
            "description":   description or f"{metric_col} by {group_col}",
            "H0":            H0,
            "H1":            H1,
            "group_col":     group_col,
            "metric_col":    metric_col,
            "n_groups":      k,
            "h_statistic":   round(h_stat, 4),
            "p_value":       round(p_value, 6),
            "significant":   p_value < self.ALPHA,
            "epsilon_sq":    round(epsilon_sq, 4),
            "effect_size":   self._eta_label(epsilon_sq),
            "group_medians": group_medians,
        }

        self._print_test_result(result)
        self.results.append(result)
        return result

    # ── Step 7: Chi-Square ────────────────────────────────────────────────────

    def run_chi_square(
        self,
        col1: str,
        col2: str,
        description: str = "",
        H0: str = "",
        H1: str = "",
    ) -> dict:
        """
        Pearson Chi-Square Test of Independence.

        χ² = Σ [(Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ]
        where Eᵢⱼ = (row total × col total) / grand total

        Tests whether two categorical variables are INDEPENDENT.

        Cramér's V (effect size):
            V = sqrt(χ² / (n × min(r-1, c-1)))
            0–0.1: negligible, 0.1–0.3: weak, 0.3–0.5: moderate, >0.5: strong

        Assumption: Expected cell count ≥ 5 in ≥ 80% of cells.
        If violated → use Fisher's exact test (for 2×2) or merge categories.

        Parameters
        ----------
        col1 : First categorical column
        col2 : Second categorical column

        Returns
        -------
        dict with chi2 statistic, p-value, Cramér's V, degrees of freedom
        """
        self._ensure_loaded()

        for col in [col1, col2]:
            if col not in self.df.columns:
                print(f"  ⚠️  Skipping: '{col}' not found.")
                return {"test": "Chi-Square", "error": f"{col} not found"}

        contingency = pd.crosstab(
            self.df[col1].dropna(),
            self.df[col2].dropna(),
        )

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Check assumption: expected ≥ 5 in 80% of cells
        cells_ok = (expected >= 5).mean()
        assumption_ok = cells_ok >= 0.8

        # Cramér's V effect size
        n = contingency.sum().sum()
        r, c = contingency.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else 0.0

        # Top associations in contingency table
        observed_pct = contingency.div(contingency.sum(axis=1), axis=0).round(3)

        result = {
            "test":             "Chi-Square",
            "description":      description or f"{col1} vs {col2}",
            "H0":               H0,
            "H1":               H1,
            "col1":             col1,
            "col2":             col2,
            "chi2_statistic":   round(chi2, 4),
            "p_value":          round(p_value, 6),
            "dof":              dof,
            "significant":      p_value < self.ALPHA,
            "cramers_v":        round(cramers_v, 4),
            "effect_size":      (
                "Strong"     if cramers_v > 0.5  else
                "Moderate"   if cramers_v > 0.3  else
                "Weak"       if cramers_v > 0.1  else "Negligible"
            ),
            "assumption_ok":    assumption_ok,
            "cells_expected_ge5_pct": round(cells_ok * 100, 1),
            "contingency_table": contingency,
            "observed_pct":     observed_pct,
        }

        self._print_test_result(result)
        self.results.append(result)
        return result

    # ── Internal result printer ───────────────────────────────────────────────

    def _print_test_result(self, result: dict) -> None:
        """Unified pretty-printer for all test results."""
        print(f"\n  ┌─ {result['test']} ─────────────────────────────────")
        print(f"  │  {result.get('description', '')}")

        if result.get("H0"):
            print(f"  │  H₀: {result['H0']}")
        if result.get("H1"):
            print(f"  │  H₁: {result['H1']}")

        print(f"  │")

        test = result["test"]

        if test in ("Mann-Whitney U", "Student's T-Test", "Welch's T-Test") or "T-Test" in test:
            g0, g1 = result.get("group_0"), result.get("group_1")
            stat_key = "u_statistic" if "Mann" in test else "t_statistic"
            print(f"  │  Group {g0}: mean={result.get('mean_0'):.4f}, median={result.get('median_0', result.get('mean_0')):.4f}, n={result.get('n_0')}")
            print(f"  │  Group {g1}: mean={result.get('mean_1'):.4f}, median={result.get('median_1', result.get('mean_1')):.4f}, n={result.get('n_1')}")
            print(f"  │  Statistic: {result.get(stat_key):.4f}")
            effect_key = "cohens_d" if "T" in test else "effect_r"
            print(f"  │  Effect size ({effect_key}): {result.get(effect_key):.4f} → {result.get('effect_size')}")

        elif test in ("One-Way ANOVA", "Kruskal-Wallis"):
            stat_key = "f_statistic" if test == "One-Way ANOVA" else "h_statistic"
            eff_key  = "eta_squared"  if test == "One-Way ANOVA" else "epsilon_sq"
            print(f"  │  Groups: {result.get('n_groups')}")
            print(f"  │  Statistic: {result.get(stat_key):.4f}")
            print(f"  │  Effect size ({eff_key}): {result.get(eff_key):.4f} → {result.get('effect_size')}")

            # Print top 5 group means/medians
            gm = result.get("group_means") or result.get("group_medians")
            if gm is not None:
                print(f"  │  Top 5 groups:")
                for idx, row_data in gm.head(5).iterrows():
                    val_col = "mean" if test == "One-Way ANOVA" else "median"
                    print(f"  │    {str(idx):<20} {val_col}={row_data[val_col]:.4f}  n={int(row_data['count'])}")

        elif test == "Chi-Square":
            print(f"  │  χ² = {result.get('chi2_statistic'):.4f},  df={result.get('dof')}")
            print(f"  │  Cramér's V: {result.get('cramers_v'):.4f} → {result.get('effect_size')}")
            assumption = "✅" if result.get("assumption_ok") else "⚠️ "
            print(f"  │  Assumption (E≥5): {assumption} {result.get('cells_expected_ge5_pct'):.1f}% of cells")

        p = result.get("p_value", 1.0)
        print(f"  │  p-value: {p:.6f}")
        print(f"  │  Verdict: {self._verdict(p)}")
        print(f"  └{'─' * 52}")

    # ── Step 8: Effect Size (standalone) ─────────────────────────────────────

    def compute_effect_size(
        self,
        group1: pd.Series,
        group2: pd.Series,
    ) -> dict:
        """
        Compute Cohen's d between two Series.

        d = (μ₁ - μ₂) / s_pooled
        s_pooled = sqrt([(n₁-1)s₁² + (n₂-1)s₂²] / (n₁+n₂-2))

        Returns dict with d, magnitude label, and summary stats.
        """
        g1, g2 = group1.dropna(), group2.dropna()
        pooled_std = np.sqrt(
            ((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
            / (len(g1) + len(g2) - 2)
        )
        d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

        return {
            "cohens_d":    round(d, 4),
            "magnitude":   self._effect_label(d),
            "mean_g1":     round(g1.mean(), 4),
            "mean_g2":     round(g2.mean(), 4),
            "std_g1":      round(g1.std(), 4),
            "std_g2":      round(g2.std(), 4),
            "n_g1":        len(g1),
            "n_g2":        len(g2),
        }

    # ── Step 9: Business Questions ────────────────────────────────────────────

    def run_business_questions(self) -> list[dict]:
        """
        Run all pre-defined Olist business hypothesis tests.

        Each question tests a real business hypothesis about the Olist dataset.
        Results are collected in self.results for reporting.

        Returns
        -------
        list of result dicts, one per business question
        """
        self._ensure_loaded()
        self._section("Step 9 — Business Questions")

        bq_results = []

        for bq in self.BUSINESS_QUESTIONS:
            print(f"\n  ▶ {bq['id']}: {bq['description']}")

            test = bq.get("test")

            try:
                if test == "mannwhitney":
                    r = self.run_mannwhitney(
                        group_col=bq["group_col"],
                        metric_col=bq["metric_col"],
                        description=f"{bq['id']}: {bq['description']}",
                        H0=bq["H0"],
                        H1=bq["H1"],
                    )

                elif test == "ttest":
                    r = self.run_ttest(
                        group_col=bq["group_col"],
                        metric_col=bq["metric_col"],
                        description=f"{bq['id']}: {bq['description']}",
                    )

                elif test == "kruskal":
                    r = self.run_kruskal(
                        group_col=bq["group_col"],
                        metric_col=bq["metric_col"],
                        top_n_groups=bq.get("top_n_groups"),
                        description=f"{bq['id']}: {bq['description']}",
                        H0=bq["H0"],
                        H1=bq["H1"],
                    )

                elif test == "anova":
                    r = self.run_anova(
                        group_col=bq["group_col"],
                        metric_col=bq["metric_col"],
                        top_n_groups=bq.get("top_n_groups"),
                        description=f"{bq['id']}: {bq['description']}",
                        H0=bq["H0"],
                        H1=bq["H1"],
                    )

                elif test == "chi_square":
                    r = self.run_chi_square(
                        col1=bq["col1"],
                        col2=bq["col2"],
                        description=f"{bq['id']}: {bq['description']}",
                        H0=bq["H0"],
                        H1=bq["H1"],
                    )

                else:
                    print(f"  ⚠️  Unknown test type: {test}")
                    continue

                bq_results.append(r)

            except Exception as e:
                print(f"  ❌ {bq['id']} failed: {e}")
                bq_results.append({
                    "id":    bq["id"],
                    "error": str(e),
                })

        return bq_results

    # ── Step 10: Plot Results ─────────────────────────────────────────────────

    def plot_results(self, save: bool = True) -> None:
        """
        Visualize hypothesis test outcomes.

        Plots:
            1. p-value forest plot — all tests at a glance
            2. Box plots for key Mann-Whitney / Kruskal comparisons
            3. Effect size comparison bar chart
        """
        self._ensure_loaded()

        if not self.results:
            print("No results to plot. Run tests first.")
            return

        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")

        # Filter to results that have p_value and description
        valid_results = [r for r in self.results if "p_value" in r and "description" in r]

        if not valid_results:
            print("No plottable results found.")
            return

        # ── 1. p-Value Forest Plot ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, max(5, len(valid_results) * 0.7)))

        descriptions = [r["description"][:55] for r in valid_results]
        p_values     = [r["p_value"] for r in valid_results]
        significant  = [r.get("significant", False) for r in valid_results]

        colors = ["#1D9E75" if s else "#D85A30" for s in significant]
        bars   = ax.barh(
            descriptions,
            [-np.log10(max(p, 1e-300)) for p in p_values],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )

        # Significance threshold line
        ax.axvline(
            x=-np.log10(self.ALPHA),
            color="#534AB7",
            linestyle="--",
            lw=1.5,
            label=f"α = {self.ALPHA} (-log₁₀p = {-np.log10(self.ALPHA):.1f})",
        )

        # Annotate each bar with p-value
        for bar, p in zip(bars, p_values):
            label = f"p={p:.4f}" if p >= 0.001 else f"p={p:.2e}"
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="left", fontsize=8,
            )

        ax.set_xlabel("-log₁₀(p-value)  [longer bar = more significant]", fontsize=10)
        ax.set_title(
            "Hypothesis Tests — p-Value Forest Plot\n"
            "🟢 Significant (Reject H₀)   🔴 Not Significant",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()

        if save:
            path = plots_dir / "pvalue_forest_plot.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            print(f"  Saved: {path.name}")
        plt.close(fig)

        # ── 2. Box plots for binary comparisons ───────────────────────────
        binary_results = [
            r for r in valid_results
            if r.get("test") == "Mann-Whitney U" and "group_col" in r and "metric_col" in r
        ]

        if binary_results:
            ncols = min(3, len(binary_results))
            nrows = (len(binary_results) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes = np.array(axes).flatten() if len(binary_results) > 1 else [axes]
            fig.suptitle(
                "Mann-Whitney U — Group Comparisons\n(medians compared)",
                fontsize=13, fontweight="bold",
            )

            for ax, result in zip(axes, binary_results):
                gc  = result["group_col"]
                mc  = result["metric_col"]
                g0  = result["group_0"]
                g1  = result["group_1"]

                if gc not in self.df.columns or mc not in self.df.columns:
                    ax.set_visible(False)
                    continue

                data_g0 = self.df.loc[self.df[gc] == g0, mc].dropna()
                data_g1 = self.df.loc[self.df[gc] == g1, mc].dropna()

                # Clip to 1st-99th pct for readability
                q_lo = min(data_g0.quantile(0.01), data_g1.quantile(0.01))
                q_hi = max(data_g0.quantile(0.99), data_g1.quantile(0.99))
                data_g0 = data_g0.clip(q_lo, q_hi)
                data_g1 = data_g1.clip(q_lo, q_hi)

                bp = ax.boxplot(
                    [data_g0, data_g1],
                    patch_artist=True,
                    widths=0.5,
                    flierprops=dict(marker=".", markersize=2, alpha=0.2),
                    medianprops=dict(color="white", lw=2),
                )

                bp["boxes"][0].set_facecolor("#7F77DD")
                bp["boxes"][1].set_facecolor("#D85A30")

                sig_label = "✅ Significant" if result["significant"] else "❌ Not Significant"
                ax.set_title(
                    f"{mc}\nby {gc}\n{sig_label}  p={result['p_value']:.4f}",
                    fontsize=9,
                )
                ax.set_xticklabels([f"Group {g0}", f"Group {g1}"], fontsize=8)
                ax.set_ylabel(mc, fontsize=8)

            # Hide unused axes
            for ax in axes[len(binary_results):]:
                ax.set_visible(False)

            plt.tight_layout()
            if save:
                path = plots_dir / "mannwhitney_boxplots.png"
                plt.savefig(path, dpi=130, bbox_inches="tight")
                print(f"  Saved: {path.name}")
            plt.close(fig)

        # ── 3. Effect Size Summary ─────────────────────────────────────────
        eff_results = [
            r for r in valid_results
            if "effect_size" in r and r.get("significant")
        ]

        if eff_results:
            size_map = {"Large": 3, "Medium": 2, "Small": 1, "Negligible": 0}
            fig, ax = plt.subplots(figsize=(10, max(4, len(eff_results) * 0.7)))

            descs = [r["description"][:50] for r in eff_results]
            sizes = [size_map.get(r["effect_size"], 0) for r in eff_results]
            eff_colors = {3: "#1D9E75", 2: "#BA7517", 1: "#D85A30", 0: "#AAAAAA"}
            bar_colors = [eff_colors[s] for s in sizes]

            ax.barh(descs, sizes, color=bar_colors, edgecolor="white", height=0.6)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["Negligible", "Small", "Medium", "Large"])
            ax.set_title(
                "Effect Sizes — Significant Tests Only\n"
                "🟢 Large  🟡 Medium  🔴 Small",
                fontsize=12, fontweight="bold",
            )
            ax.invert_yaxis()
            plt.tight_layout()

            if save:
                path = plots_dir / "effect_sizes.png"
                plt.savefig(path, dpi=130, bbox_inches="tight")
                print(f"  Saved: {path.name}")
            plt.close(fig)

        print(f"\n  All plots saved to: {plots_dir}")

    # ── Step 11: Save Report ──────────────────────────────────────────────────

    def save_report(self) -> None:
        """
        Save all hypothesis test results to reports/hypothesis/.

        Files:
            hypothesis_results.csv   — machine-readable results table
            summary.txt              — human-readable findings
        """
        self._section("Step 11 — Saving Report")

        if not self.results:
            print("No results to save.")
            return

        # Flatten results to CSV-safe format (exclude DataFrame objects)
        flat_results = []
        for r in self.results:
            flat = {k: v for k, v in r.items()
                    if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))}
            flat_results.append(flat)

        results_df = pd.DataFrame(flat_results)
        csv_path = REPORTS_DIR / "hypothesis_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

        # ── Human-readable summary ────────────────────────────────────────
        lines = [
            "Day 6 — Hypothesis Testing Summary",
            f"Run at: {self._run_timestamp}",
            f"Dataset: {self.filepath.name}",
            f"Shape: {self.df.shape}",
            f"Alpha (significance level): {self.ALPHA}",
            "",
            f"Total tests run      : {len(self.results)}",
            f"Significant results  : {sum(r.get('significant', False) for r in self.results)}",
            f"Non-significant      : {sum(not r.get('significant', True) for r in self.results)}",
            "",
            "─" * 60,
            "FINDINGS",
            "─" * 60,
        ]

        for r in self.results:
            if "p_value" not in r:
                continue
            verdict   = "✅ SIGNIFICANT" if r.get("significant") else "❌ NOT SIGNIFICANT"
            eff       = r.get("effect_size", "N/A")
            lines.append(f"\n{r.get('description', '')}")
            if r.get("H0"):
                lines.append(f"  H₀: {r['H0']}")
            lines.append(f"  Test     : {r.get('test', '')}")
            lines.append(f"  p-value  : {r.get('p_value', '')}")
            lines.append(f"  Verdict  : {verdict}")
            lines.append(f"  Effect   : {eff}")

        lines += [
            "",
            "─" * 60,
            "KEY INSIGHTS (Significant + Large/Medium effect)",
            "─" * 60,
        ]

        key_findings = [
            r for r in self.results
            if r.get("significant") and r.get("effect_size") in ("Large", "Medium")
        ]

        if key_findings:
            for r in key_findings:
                lines.append(f"  ★ {r.get('description', '')} (effect={r.get('effect_size')})")
        else:
            lines.append("  No large/medium effect findings. Results are statistically significant")
            lines.append("  but practically small — common with large n datasets.")

        lines += [
            "",
            "─" * 60,
            "NEXT STEPS → Day 7: NLP on Review Text",
            "  TF-IDF features from Portuguese review text",
            "  Sentiment polarity per review_score group",
            "─" * 60,
        ]

        summary_path = REPORTS_DIR / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Saved: {summary_path.name}")

        # ── Console highlight ─────────────────────────────────────────────
        print(f"\n{'─' * 65}")
        print("  Day 6 Complete — Key Findings")
        print(f"{'─' * 65}")

        sig = [r for r in self.results if r.get("significant") and "p_value" in r]
        not_sig = [r for r in self.results if not r.get("significant") and "p_value" in r]

        print(f"\n  ✅ Significant ({len(sig)}):")
        for r in sig:
            print(f"     {r.get('description', '')[:60]:<60}  effect={r.get('effect_size', 'N/A')}")

        if not_sig:
            print(f"\n  ❌ Not Significant ({len(not_sig)}):")
            for r in not_sig:
                print(f"     {r.get('description', '')[:60]:<60}  p={r.get('p_value'):.4f}")

        print(f"\n  Reports saved to: {REPORTS_DIR}")
        print(f"\n  Next → Day 7: NLP on Review Text")
        print(f"    TF-IDF, sentiment polarity, bigrams per review_score group")


# ── Orchestrator ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        tester = HypothesisTester()

        # Step 1: Load
        tester.load_data()

        # Step 2: Quick normality check on key columns
        print("\n── Normality Checks (decide parametric vs non-parametric) ───")
        for col in ["review_score", "delivery_delay_days", "freight_value"]:
            tester.check_normality(col)

        # Steps 3-8: Run all business questions
        tester.run_business_questions()

        # Step 9: Visualize
        tester.plot_results(save=True)

        # Step 10: Save report
        tester.save_report()

        print("\n  Day 6 complete! Check reports/hypothesis/ for outputs.")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise