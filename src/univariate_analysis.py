import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from scipy.stats import boxcox, yeojohnson
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
# Adjust these paths to match your project's src/config.py
try:
    from src.config import PROCESSED_DATA_DIR
    REPORTS_DIR = Path(str(PROCESSED_DATA_DIR)).parent / "reports" / "univariate"
except ImportError:
    PROCESSED_DATA_DIR = Path("data/processed")
    REPORTS_DIR = Path("reports/univariate")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class UnivariateAnalyzer:
    """
    Day 4 — Univariate Statistics & Distribution Analysis.

    Pipeline Steps:
        1. load_data()             - Load featured_dataset.csv
        2. compute_stats()         - Skewness, Kurtosis, mean/median/std per column
        3. transform_skewed()      - Box-Cox (positive-only) / Yeo-Johnson (any sign)
        4. plot_distributions()    - KDE, Q-Q, PDF vs CDF for every numerical column
        5. target_deep_dive()      - review_score full breakdown
        6. save_report()           - CSV stats summary + all plots to reports/univariate/

    Math Cheatsheet (inline so you never need to look elsewhere):
    ─────────────────────────────────────────────────────────────
    Skewness   = E[(X-μ)³] / σ³
                 >  1  → right-skewed (long right tail, e.g. price)
                 < -1  → left-skewed  (long left tail)
                  ~0   → symmetric

    Kurtosis   = E[(X-μ)⁴] / σ⁴  (excess kurtosis = this - 3)
                 >0 (leptokurtic)  → heavy tails, more outlier-prone
                 <0 (platykurtic)  → thin tails, fewer extremes
                 =0 (mesokurtic)   → Normal distribution baseline

    Box-Cox    λ=0 → log(x),  λ=1 → x (no change), λ=0.5 → √x
               Finds optimal λ to maximise normality. ONLY positive values.

    Yeo-Johnson: same idea but handles zero & negative values.
               Preferred when columns can be negative (e.g. delivery_delay_days).

    Q-Q Plot   : If points lie on the diagonal → Normal.
               S-curve → heavy tails. C-curve → skewed.

    PDF        : P(X = x) density at a point
    CDF        : P(X ≤ x) cumulative — useful for percentile reasoning
    """

    # Columns that might be negative — use Yeo-Johnson
    POSSIBLY_NEGATIVE = {
        "delivery_delay_days",
        "approval_delay_hrs",
        "carrier_pickup_delay_hrs",
        "customer_zip_lat",
        "customer_zip_lng",
        "seller_zip_lat",
        "seller_zip_lng",
        "purchase_hour_sin",
        "purchase_hour_cos",
        "purchase_dayofweek_sin",
        "purchase_dayofweek_cos",
        "purchase_month_sin",
        "purchase_month_cos",
    }

    # Skewness threshold — columns beyond this get transformed
    SKEW_THRESHOLD = 1.0

    def __init__(self, filename: str = "featured_dataset.csv"):
        self.filepath = Path(str(PROCESSED_DATA_DIR)) / filename
        self.df: pd.DataFrame | None = None
        self.numerical_cols: list[str] = []
        self.stats_df: pd.DataFrame | None = None      # populated by compute_stats()
        self.transformed_df: pd.DataFrame | None = None  # populated by transform_skewed()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

    def _is_positive_only(self, col: str) -> bool:
        """True if the column has no zero or negative values (Box-Cox safe)."""
        return (self.df[col].dropna() > 0).all()

    # ── Step 1: Load ────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """Load featured_dataset.csv and identify numerical columns."""
        print(f"Loading: {self.filepath}")

        if not self.filepath.exists():
            raise FileNotFoundError(f"Not found: {self.filepath}")

        self.df = pd.read_csv(self.filepath)

        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # Numerical columns — exclude IDs and binary flags (they're not continuous)
        exclude_patterns = ("_id", "is_")
        self.numerical_cols = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if not any(col.endswith(p) or col.startswith(p) for p in exclude_patterns)
        ]

        print(f"Shape              : {self.df.shape}")
        print(f"Numerical cols     : {len(self.numerical_cols)}")
        print(f"Cols               : {self.numerical_cols[:8]} {'...' if len(self.numerical_cols) > 8 else ''}")

    # ── Step 2: Compute Stats ───────────────────────────────────────────────

    def compute_stats(self) -> pd.DataFrame:
        """
        Compute per-column distribution statistics.

        Returns
        -------
        pd.DataFrame with columns:
            mean, median, std, min, max,
            skewness, kurtosis (excess),
            missing_pct, iqr, cv (coefficient of variation)
        """
        self._ensure_loaded()
        print("\nComputing distribution statistics...")

        records = []
        for col in self.numerical_cols:
            series = self.df[col].dropna()

            if len(series) == 0:
                continue

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            mean = series.mean()

            records.append({
                "column":       col,
                "mean":         round(mean, 4),
                "median":       round(series.median(), 4),
                "std":          round(series.std(), 4),
                "min":          round(series.min(), 4),
                "max":          round(series.max(), 4),
                "skewness":     round(series.skew(), 4),       # scipy uses Fisher's def
                "kurtosis":     round(series.kurtosis(), 4),   # excess kurtosis (normal=0)
                "iqr":          round(q3 - q1, 4),
                "cv":           round(series.std() / mean, 4) if mean != 0 else np.nan,
                "missing_pct":  round(self.df[col].isnull().mean() * 100, 2),
                "n_unique":     series.nunique(),
            })

        self.stats_df = pd.DataFrame(records).set_index("column")

        # Pretty print the most skewed columns
        skewed = self.stats_df[self.stats_df["skewness"].abs() > self.SKEW_THRESHOLD]
        print(f"\nHighly skewed columns (|skew| > {self.SKEW_THRESHOLD}): {len(skewed)}")
        if not skewed.empty:
            print(skewed[["mean", "skewness", "kurtosis"]].to_string())

        return self.stats_df

    # ── Step 3: Transform Skewed Columns ───────────────────────────────────

    def transform_skewed(self) -> pd.DataFrame:
        """
        Apply Box-Cox or Yeo-Johnson to highly skewed columns.

        Decision rule:
            - Column in POSSIBLY_NEGATIVE  →  Yeo-Johnson (safe for any sign)
            - All values > 0               →  Box-Cox (finds optimal λ)
            - Otherwise                    →  Yeo-Johnson

        Transformed columns are suffixed with '_transformed' and added to
        a copy of self.df (self.transformed_df). The original self.df is
        never modified here so you can compare before/after.
        """
        self._ensure_loaded()
        if self.stats_df is None:
            self.compute_stats()

        print("\nTransforming skewed columns...")
        self.transformed_df = self.df.copy()

        skewed_cols = self.stats_df[
            self.stats_df["skewness"].abs() > self.SKEW_THRESHOLD
        ].index.tolist()

        for col in skewed_cols:
            series = self.df[col].dropna()

            # Choose transformer
            use_yeo = (
                col in self.POSSIBLY_NEGATIVE
                or not self._is_positive_only(col)
            )

            if use_yeo:
                transformed, lam = yeojohnson(series)
                method = f"Yeo-Johnson (λ={lam:.3f})"
            else:
                # Box-Cox requires strictly positive values
                transformed, lam = boxcox(series)
                method = f"Box-Cox (λ={lam:.3f})"

            new_col = f"{col}_transformed"
            self.transformed_df.loc[series.index, new_col] = transformed

            # Check skew improvement
            new_skew = pd.Series(transformed).skew()
            old_skew = self.stats_df.loc[col, "skewness"]
            print(f"  {col:<40} skew: {old_skew:+.3f} → {new_skew:+.3f}  [{method}]")

        print(f"\nTransformed {len(skewed_cols)} columns.")
        return self.transformed_df

    # ── Step 4: Plot Distributions ─────────────────────────────────────────

    def plot_distributions(
        self,
        max_cols: int = 20,
        save: bool = True,
    ) -> None:
        """
        For each numerical column, create a 4-panel figure:
            [1] Histogram + KDE     — shape of distribution
            [2] Q-Q plot            — normality check
            [3] CDF                 — cumulative probabilities
            [4] Box plot            — outlier summary

        Parameters
        ----------
        max_cols : int
            Limit number of columns to avoid generating 50+ files.
        save : bool
            If True, saves each figure to reports/univariate/plots/.
        """
        self._ensure_loaded()
        if self.stats_df is None:
            self.compute_stats()

        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        cols_to_plot = self.numerical_cols[:max_cols]
        print(f"\nPlotting distributions for {len(cols_to_plot)} columns...")

        plt.style.use("seaborn-v0_8-whitegrid")

        for col in cols_to_plot:
            series = self.df[col].dropna()
            if len(series) < 10:
                continue

            fig = plt.figure(figsize=(14, 9))
            fig.suptitle(
                f"Distribution Analysis — {col}\n"
                f"skew={self.stats_df.loc[col,'skewness']:+.3f}  "
                f"kurt={self.stats_df.loc[col,'kurtosis']:+.3f}  "
                f"n={len(series):,}",
                fontsize=13, fontweight="bold", y=1.01,
            )
            gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

            # ── Panel 1: Histogram + KDE ───────────────────────────────────
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(series, bins=50, density=True, alpha=0.55,
                     color="#7F77DD", edgecolor="white", linewidth=0.4,
                     label="Histogram")
            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 300)
            ax1.plot(x_range, kde(x_range), color="#D85A30", lw=2, label="KDE")
            ax1.axvline(series.mean(),   color="#1D9E75", lw=1.5, ls="--", label="Mean")
            ax1.axvline(series.median(), color="#BA7517", lw=1.5, ls=":",  label="Median")
            ax1.set_title("Histogram + KDE (PDF)", fontsize=10)
            ax1.set_xlabel(col, fontsize=8)
            ax1.legend(fontsize=7)

            # ── Panel 2: Q-Q Plot ──────────────────────────────────────────
            ax2 = fig.add_subplot(gs[0, 1])
            (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm")
            ax2.scatter(osm, osr, s=8, alpha=0.4, color="#7F77DD")
            line_x = np.array([osm.min(), osm.max()])
            ax2.plot(line_x, slope * line_x + intercept,
                     color="#D85A30", lw=1.5, label=f"R²={r**2:.3f}")
            ax2.set_title("Q-Q Plot (Normality Check)", fontsize=10)
            ax2.set_xlabel("Theoretical quantiles", fontsize=8)
            ax2.set_ylabel("Sample quantiles", fontsize=8)
            ax2.legend(fontsize=7)

            # ── Panel 3: CDF ───────────────────────────────────────────────
            ax3 = fig.add_subplot(gs[1, 0])
            sorted_vals = np.sort(series)
            cdf_vals    = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax3.plot(sorted_vals, cdf_vals, color="#7F77DD", lw=1.5)
            ax3.axhline(0.5, color="#D85A30", lw=1, ls="--", alpha=0.7, label="50th pct")
            ax3.axhline(0.95, color="#BA7517", lw=1, ls=":", alpha=0.7, label="95th pct")
            ax3.fill_between(sorted_vals, cdf_vals, alpha=0.12, color="#7F77DD")
            ax3.set_title("CDF (Cumulative Distribution)", fontsize=10)
            ax3.set_xlabel(col, fontsize=8)
            ax3.set_ylabel("P(X ≤ x)", fontsize=8)
            ax3.legend(fontsize=7)

            # ── Panel 4: Box Plot ──────────────────────────────────────────
            ax4 = fig.add_subplot(gs[1, 1])
            bp = ax4.boxplot(
                series,
                vert=True,
                patch_artist=True,
                widths=0.4,
                boxprops=dict(facecolor="#CECBF6", color="#534AB7"),
                medianprops=dict(color="#D85A30", lw=2),
                flierprops=dict(marker=".", markersize=3, alpha=0.3,
                                markerfacecolor="#534AB7"),
            )
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            ax4.set_title(
                f"Box Plot\nIQR={iqr:.3f}  Q1={q1:.3f}  Q3={q3:.3f}",
                fontsize=10,
            )
            ax4.set_xticks([])

            plt.tight_layout()

            if save:
                safe_name = col.replace("/", "_").replace(" ", "_")
                out_path  = plots_dir / f"{safe_name}_distribution.png"
                plt.savefig(out_path, dpi=120, bbox_inches="tight")
                print(f"  Saved: {out_path.name}")

            plt.close(fig)

        print(f"All distribution plots saved to: {plots_dir}")

    # ── Step 5: Target Variable Deep Dive ──────────────────────────────────

    def target_deep_dive(
        self,
        target_col: str = "review_score",
        save: bool = True,
    ) -> None:
        """
        Detailed analysis of the target variable (review_score).

        Plots:
            1. Full value_counts bar chart with % labels
            2. KDE by score (5 overlapping curves)
            3. Cumulative review score distribution
            4. Score vs delivery_delay_days box plots (business insight)
        """
        self._ensure_loaded()

        if target_col not in self.df.columns:
            print(f"Column '{target_col}' not found. Skipping target deep dive.")
            return

        print(f"\nTarget deep dive: {target_col}")
        series = self.df[target_col].dropna()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(
            f"Target Variable Deep Dive — {target_col}",
            fontsize=14, fontweight="bold",
        )

        colors = ["#A32D2D", "#D85A30", "#BA7517", "#1D9E75", "#534AB7"]

        # ── 1. Value counts bar chart ──────────────────────────────────────
        ax = axes[0, 0]
        vc = series.value_counts().sort_index()
        bars = ax.bar(vc.index, vc.values,
                      color=colors, edgecolor="white", linewidth=0.5)
        total = vc.sum()
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{val/total*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title("Review Score Distribution", fontsize=10)
        ax.set_xlabel("Score (1-5)")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])

        # ── 2. KDE per score ───────────────────────────────────────────────
        if "delivery_delay_days" in self.df.columns:
            ax = axes[0, 1]
            from scipy.stats import gaussian_kde
            for score, color in zip(sorted(series.unique()), colors):
                subset = self.df.loc[
                    self.df[target_col] == score, "delivery_delay_days"
                ].dropna()
                if len(subset) > 30:
                    kde = gaussian_kde(subset)
                    x_range = np.linspace(subset.quantile(0.01),
                                          subset.quantile(0.99), 200)
                    ax.plot(x_range, kde(x_range), color=color,
                            lw=2, label=f"Score {int(score)}")
            ax.set_title("Delivery Delay by Review Score (KDE)", fontsize=10)
            ax.set_xlabel("delivery_delay_days")
            ax.legend(fontsize=8)
        else:
            axes[0, 1].set_visible(False)

        # ── 3. CDF of review scores ────────────────────────────────────────
        ax = axes[1, 0]
        sorted_vals = np.sort(series)
        cdf_vals    = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf_vals, color="#534AB7", lw=2)
        ax.fill_between(sorted_vals, cdf_vals, alpha=0.15, color="#534AB7")
        ax.set_title("Cumulative Distribution of Review Scores", fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("P(Score ≤ x)")
        ax.set_xticks([1, 2, 3, 4, 5])

        # ── 4. Box plots of delay per score group ─────────────────────────
        if "delivery_delay_days" in self.df.columns:
            ax = axes[1, 1]
            score_groups = [
                self.df.loc[self.df[target_col] == s, "delivery_delay_days"].dropna()
                for s in sorted(series.unique())
            ]
            bp = ax.boxplot(
                [g.clip(*g.quantile([0.02, 0.98])) for g in score_groups],
                patch_artist=True,
                widths=0.5,
                flierprops=dict(marker=".", markersize=3, alpha=0.2),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title("Delivery Delay vs Review Score\n(Business Insight)", fontsize=10)
            ax.set_xlabel("Review Score")
            ax.set_ylabel("delivery_delay_days")
            ax.set_xticklabels([int(s) for s in sorted(series.unique())])
        else:
            axes[1, 1].set_visible(False)

        plt.tight_layout()

        if save:
            out_path = REPORTS_DIR / "target_deep_dive.png"
            plt.savefig(out_path, dpi=130, bbox_inches="tight")
            print(f"Saved: {out_path}")

        plt.close(fig)

        # Summary stats
        print(f"\n{target_col} summary:")
        print(f"  Mean   : {series.mean():.3f}")
        print(f"  Median : {series.median():.0f}")
        print(f"  Skew   : {series.skew():.3f}  (negative = left-skewed = most reviews are 5★)")
        print(f"  Value counts:\n{vc.to_string()}")

    # ── Step 6: Save Report ─────────────────────────────────────────────────

    def save_report(self) -> None:
        """
        Save the statistics DataFrame to a CSV and print a markdown-style
        summary of the most interesting findings.
        """
        self._ensure_loaded()
        if self.stats_df is None:
            self.compute_stats()

        out_path = REPORTS_DIR / "univariate_stats.csv"
        self.stats_df.to_csv(out_path)
        print(f"\nStats saved: {out_path}")

        # Top 5 most skewed
        top_skewed = self.stats_df.nlargest(5, "skewness")[["mean", "skewness", "kurtosis"]]
        print("\nTop 5 most right-skewed columns:")
        print(top_skewed.to_string())

        # Columns needing transformation
        needs_transform = self.stats_df[
            self.stats_df["skewness"].abs() > self.SKEW_THRESHOLD
        ]
        print(f"\n{len(needs_transform)} columns recommended for transformation "
              f"(|skew| > {self.SKEW_THRESHOLD}):")
        for col in needs_transform.index:
            skew = needs_transform.loc[col, "skewness"]
            method = "Yeo-Johnson" if col in self.POSSIBLY_NEGATIVE else "Box-Cox"
            print(f"  {col:<42} skew={skew:+.3f}  → use {method}")


# ── Orchestrator ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        analyzer = UnivariateAnalyzer()

        analyzer.load_data()
        analyzer.compute_stats()
        analyzer.transform_skewed()

        # Plot all (set max_cols lower during dev to save time)
        analyzer.plot_distributions(max_cols=20, save=True)

        analyzer.target_deep_dive(target_col="review_score", save=True)
        analyzer.save_report()

        print("\nDay 4 complete! Check reports/univariate/ for outputs.")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise