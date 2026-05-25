import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
try:
    from src.config import PROCESSED_DATA_DIR
    REPORTS_DIR = Path(str(PROCESSED_DATA_DIR)).parent / "reports" / "bivariate"
except ImportError:
    PROCESSED_DATA_DIR = Path("data/processed")
    REPORTS_DIR = Path("reports/bivariate")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class BivariateAnalyzer:
    """
    Day 5 - Bivariate & Multivariate Analysis.

    Pipeline Steps:
        1. load_data()          - Load featured_dataset.csv
        2. correlation()        - Pearson + Spearman matrices
        3. plot_heatmap()       - Correlation heatmap + top-feature pairplot
        4. cramers_v()          - Categorical vs categorical association
        5. point_biserial()     - Continuous vs categorical correlation
        6. compute_vif()        - Variance Inflation Factor (multicollinearity)
        7. save_report()        - All results to reports/bivariate/
        8. run_full_pipeline()  - Orchestrate all steps in one call

    --- Math Cheatsheet ---------------------------------------------------------
    Pearson r   = Sum[(Xi - X_mean)(Yi - Y_mean)] / (n * sigma_x * sigma_y)
                  Measures LINEAR relationship. Assumes normality.
                  Range: -1 to +1. Sensitive to outliers.

    Spearman rho = Pearson r applied on RANKS of X and Y
                  Measures MONOTONIC relationship (not just linear).
                  Use when: data is skewed, has outliers, or is ordinal.
                  Rule of thumb: price/delay columns -> use Spearman.

    Cramer's V  = sqrt(chi^2 / (n * min(r-1, c-1)))
                  chi^2 = chi-square statistic from contingency table.
                  Range: 0 (no association) to 1 (perfect association).
                  Use for: two categorical columns (state vs category etc.)

    Point-biserial r = Pearson r where one variable is binary (0/1)
                  Mathematically identical to Pearson, but semantically
                  correct when one variable is a flag (is_late_delivery etc.)

    VIF         = 1 / (1 - R^2_i)
                  R^2_i = how well column i is predicted by ALL other columns.
                  VIF > 5  -> moderate multicollinearity (investigate)
                  VIF > 10 -> severe multicollinearity (drop or combine)
                  Why it matters: multicollinear features make linear model
                  coefficients unstable and uninterpretable.
    -----------------------------------------------------------------------------
    """

    # Columns to treat as categorical even if encoded as int
    CATEGORICAL_COLS = [
        "customer_state",
        "seller_state",
        "product_category_name_english",
        "order_status",
        "payment_type",
        "job_type",
        "job_mode",
    ]

    # Binary flag columns — best analysed with point-biserial
    BINARY_COLS = [
        "is_late_delivery",
        "is_long_distance",
        "is_same_state",
        "is_weekend_purchase",
    ]

    # Target variable
    TARGET = "review_score"

    def __init__(self, filename: str = "featured_dataset.csv"):
        self.filepath = Path(str(PROCESSED_DATA_DIR)) / filename
        self.df: pd.DataFrame | None = None
        self.numerical_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.pearson_matrix: pd.DataFrame | None = None
        self.spearman_matrix: pd.DataFrame | None = None
        self.vif_df: pd.DataFrame | None = None
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __repr__(self) -> str:
        shape = self.df.shape if self.df is not None else "not loaded"
        loaded = "✅" if self.df is not None else "❌"
        corr   = "✅" if self.pearson_matrix is not None else "❌"
        vif    = "✅" if self.vif_df is not None else "❌"
        return (
            f"BivariateAnalyzer(\n"
            f"  data       : {loaded} {shape}\n"
            f"  correlation: {corr}\n"
            f"  vif        : {vif}\n"
            f"  timestamp  : {self._run_timestamp}\n"
            f")"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

    def _section(self, title: str) -> None:
        """Pretty section divider for console output."""
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    # ── Step 1: Load ─────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """
        Load featured_dataset.csv and separate numerical/categorical columns.

        Notes
        -----
        - ID columns (ending in _id) are excluded from numerical analysis
          because they carry no statistical meaning.
        - Columns in CATEGORICAL_COLS are forced into categorical even if
          pandas read them as int (e.g., encoded state codes).
        """
        self._section("Step 1 — Loading Data")
        print(f"Path: {self.filepath}")

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.filepath}\n"
                f"Run Day 3 feature engineering pipeline first."
            )

        self.df = pd.read_csv(self.filepath)

        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # Numerical: exclude IDs and known binary/categorical columns
        exclude = set(self.CATEGORICAL_COLS) | set(self.BINARY_COLS)
        self.numerical_cols = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if not col.endswith("_id") and col not in exclude
        ]

        # Categorical: object dtype + predefined list that may be int-encoded
        self.categorical_cols = list(
            set(self.df.select_dtypes(include=["object", "category"]).columns.tolist())
            | set([c for c in self.CATEGORICAL_COLS if c in self.df.columns])
        )

        # Runtime info
        null_pct = self.df.isnull().mean().mul(100).round(1)
        cols_with_nulls = null_pct[null_pct > 0]

        print(f"\nShape             : {self.df.shape}")
        print(f"Numerical cols    : {len(self.numerical_cols)}")
        print(f"Categorical cols  : {len(self.categorical_cols)}")
        print(f"Binary flag cols  : {sum(c in self.df.columns for c in self.BINARY_COLS)}")
        print(f"Target present    : {'✅' if self.TARGET in self.df.columns else '❌ (missing!)'}")

        if not cols_with_nulls.empty:
            print(f"\nColumns with nulls (will be dropped in pairwise ops):")
            for col, pct in cols_with_nulls.items():
                print(f"  {col:<42} {pct:.1f}%")

    # ── Step 2: Pearson & Spearman Correlation ────────────────────────────────

    def correlation(self, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute Pearson and Spearman correlation matrices.

        Pearson  → linear relationships, normally-distributed data
        Spearman → monotonic relationships, skewed/ordinal data

        For Olist: price, freight, and delay columns are heavily right-skewed
        → Spearman is more reliable for those. We compute both and compare.

        Parameters
        ----------
        top_n : int
            Number of columns (by variance) to include in the matrix.
            Full matrix with 50+ features is unreadable.

        Returns
        -------
        (pearson_df, spearman_df)

        Notes
        -----
        Pearson vs Spearman disagreement (diff > 0.1):
            Means relationship is NON-LINEAR or OUTLIER-DRIVEN.
            Spearman is more trustworthy in those cases.
        """
        self._ensure_loaded()
        self._section("Step 2 — Pearson & Spearman Correlations")

        # ── Select top_n highest-variance columns ─────────────────────────
        # High-variance columns carry the most signal for correlation.
        # Always include TARGET even if it falls outside top_n.
        variances = self.df[self.numerical_cols].var().sort_values(ascending=False)
        top_cols  = variances.head(top_n).index.tolist()

        if self.TARGET in self.numerical_cols and self.TARGET not in top_cols:
            # Replace the lowest-variance col to keep matrix size at top_n
            top_cols[-1] = self.TARGET
            print(f"  Note: {self.TARGET} had low variance but was added (it's the target).")

        subset = self.df[top_cols].dropna()
        print(f"  Matrix size: {len(top_cols)} × {len(top_cols)}  (rows after dropna: {len(subset):,})")

        self.pearson_matrix  = subset.corr(method="pearson")
        self.spearman_matrix = subset.corr(method="spearman")

        # ── Print top correlations WITH target ────────────────────────────
        if self.TARGET in top_cols:
            pearson_with_target  = (
                self.pearson_matrix[self.TARGET]
                .drop(self.TARGET)
                .sort_values(key=abs, ascending=False)
            )
            spearman_with_target = (
                self.spearman_matrix[self.TARGET]
                .drop(self.TARGET)
                .sort_values(key=abs, ascending=False)
            )

            print(f"\nTop 10 Pearson correlations with '{self.TARGET}':")
            for feat, val in pearson_with_target.head(10).items():
                bar = "█" * int(abs(val) * 20)
                direction = "+" if val > 0 else "-"
                print(f"  {feat:<42} {direction}{abs(val):.3f}  {bar}")

            print(f"\nTop 10 Spearman correlations with '{self.TARGET}':")
            for feat, val in spearman_with_target.head(10).items():
                bar = "█" * int(abs(val) * 20)
                direction = "+" if val > 0 else "-"
                print(f"  {feat:<42} {direction}{abs(val):.3f}  {bar}")

            # ── Flag Pearson vs Spearman disagreements ────────────────────
            # Large difference → non-linear relationship or outlier-driven
            # → trust Spearman more, and investigate with scatter plots
            diff = (pearson_with_target - spearman_with_target).abs()
            big_diff = diff[diff > 0.1].sort_values(ascending=False)

            if not big_diff.empty:
                print(f"\n⚠️  Pearson vs Spearman diff > 0.1 (non-linear/outlier-driven):")
                print(f"   → For these, PREFER Spearman. Consider log-transform or robust scaler.")
                for feat, val in big_diff.items():
                    print(f"  {feat:<42} Δ={val:.3f}")
            else:
                print(f"\n✅ Pearson ≈ Spearman for all features — relationships are fairly linear.")

        return self.pearson_matrix, self.spearman_matrix

    # ── Step 3: Heatmap + Pairplot ────────────────────────────────────────────

    def plot_heatmap(self, save: bool = True) -> None:
        """
        Plot side-by-side Pearson and Spearman heatmaps.
        Also produces a pairplot for the top 6 features correlated with target.

        Dynamic figsize ensures annotations don't overflow regardless of
        how many columns are in the matrix.
        """
        self._ensure_loaded()
        if self.pearson_matrix is None:
            self.correlation()

        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        n = len(self.pearson_matrix)
        cell_size = max(0.6, min(1.0, 14.0 / n))   # shrink cells for large matrices

        # ── Side-by-side heatmaps ─────────────────────────────────────────
        fig_w = cell_size * n * 2 + 4
        fig_h = cell_size * n + 2
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        fig.suptitle("Pearson vs Spearman Correlation Matrices", fontsize=14, fontweight="bold")

        # Only show lower triangle — upper is redundant
        mask = np.triu(np.ones_like(self.pearson_matrix, dtype=bool))

        annot_size = max(5, min(9, int(90 / n)))   # scale font with matrix size

        for ax, matrix, title in zip(
            axes,
            [self.pearson_matrix, self.spearman_matrix],
            ["Pearson (linear)", "Spearman (monotonic)"],
        ):
            sns.heatmap(
                matrix,
                ax=ax,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.3,
                annot_kws={"size": annot_size},
                cbar_kws={"shrink": 0.6},
            )
            ax.set_title(title, fontsize=11)
            ax.tick_params(axis="x", rotation=45, labelsize=annot_size)
            ax.tick_params(axis="y", rotation=0,  labelsize=annot_size)

        plt.tight_layout()

        if save:
            path = plots_dir / "correlation_heatmaps.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            print(f"  Saved: {path.name}")
        plt.close(fig)

        # ── Pairplot: top correlated features with target ─────────────────
        if self.TARGET in self.spearman_matrix.columns:
            top_features = (
                self.spearman_matrix[self.TARGET]
                .drop(self.TARGET)
                .abs()
                .sort_values(ascending=False)
                .head(5)
                .index
                .tolist()
            )
            top_features.append(self.TARGET)

            pairplot_df = self.df[top_features].dropna().sample(
                n=min(3000, len(self.df)), random_state=42
            )

            # Use hue only if target has <= 6 unique values (classification)
            # For regression targets, skip hue to avoid overplotting
            use_hue = pairplot_df[self.TARGET].nunique() <= 6

            pp = sns.pairplot(
                pairplot_df,
                hue=self.TARGET if use_hue else None,
                diag_kind="kde",
                plot_kws={"alpha": 0.4, "s": 12},
                palette="viridis" if use_hue else None,
            )
            pp.fig.suptitle(
                f"Pairplot — top 5 features correlated with '{self.TARGET}'",
                y=1.02, fontsize=12, fontweight="bold",
            )

            if save:
                path = plots_dir / "pairplot_top_features.png"
                pp.savefig(path, dpi=110, bbox_inches="tight")
                print(f"  Saved: {path.name}")
            plt.close("all")

        # ── Target distribution plot ──────────────────────────────────────
        if self.TARGET in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f"Target Distribution: '{self.TARGET}'", fontsize=13, fontweight="bold")

            target_series = self.df[self.TARGET].dropna()

            # Histogram + KDE
            axes[0].hist(target_series, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
            axes[0].set_xlabel(self.TARGET)
            axes[0].set_ylabel("Count")
            axes[0].set_title("Histogram")

            # Add stats as text
            skew = target_series.skew()
            kurt = target_series.kurt()
            axes[0].text(
                0.97, 0.95,
                f"Skew : {skew:.2f}\nKurt : {kurt:.2f}\nMean : {target_series.mean():.2f}",
                transform=axes[0].transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

            # Q-Q plot to check normality
            stats.probplot(target_series, dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot (normality check)")

            plt.tight_layout()
            if save:
                path = plots_dir / "target_distribution.png"
                plt.savefig(path, dpi=130, bbox_inches="tight")
                print(f"  Saved: {path.name}")
            plt.close(fig)

    # ── Step 4: Cramér's V (categorical vs categorical) ───────────────────────

    def cramers_v(self, save: bool = True) -> pd.DataFrame:
        """
        Compute Cramér's V for all categorical column pairs.

        Formula:
            χ²  = sum of (observed - expected)² / expected  (chi-square)
            V   = sqrt(χ² / (n · min(rows-1, cols-1)))

        Interpretation:
            0.0 – 0.1  → negligible
            0.1 – 0.3  → weak
            0.3 – 0.5  → moderate
            > 0.5      → strong

        Notes
        -----
        High-cardinality columns (e.g. product categories with 70+ values)
        will naturally inflate χ², so V should be read with cardinality in mind.

        Returns
        -------
        DataFrame of Cramér's V values (symmetric matrix).
        """
        self._ensure_loaded()
        self._section("Step 4 — Cramér's V (Categorical Associations)")

        cat_cols = [c for c in self.categorical_cols if c in self.df.columns]
        if len(cat_cols) < 2:
            print("Not enough categorical columns for Cramér's V. Skipping.")
            return pd.DataFrame()

        # THE FIX: Actually remove high-cardinality columns instead of just warning
        MAX_CATEGORIES = 50
        filtered_cat_cols = []
        
        for col in cat_cols:
            unique_count = self.df[col].nunique()
            if unique_count <= MAX_CATEGORIES:
                filtered_cat_cols.append(col)
            else:
                print(f"  🚫 Skipping '{col}' (Too many categories: {unique_count})")
                
        cat_cols = filtered_cat_cols
        
        if len(cat_cols) < 2:
            print("Not enough low-cardinality categorical columns for Cramér's V. Skipping.")
            return pd.DataFrame()

        print(f"  Computing V for {len(cat_cols)} valid categorical columns...")

        def _cramers_v_pair(x: pd.Series, y: pd.Series) -> float:
            contingency = pd.crosstab(x, y)
            chi2, _, _, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            r, c = contingency.shape
            return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0

        v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

        for col1 in cat_cols:
            for col2 in cat_cols:
                if col1 == col2:
                    v_matrix.loc[col1, col2] = 1.0
                else:
                    paired = self.df[[col1, col2]].dropna()
                    v_matrix.loc[col1, col2] = _cramers_v_pair(paired[col1], paired[col2])

        # ── Print associations by strength ───────────────────────────────
        print("\nCramér's V results (above diagonal only):")
        rows = []
        for i, col1 in enumerate(cat_cols):
            for col2 in cat_cols[i+1:]:
                v = v_matrix.loc[col1, col2]
                strength = (
                    "🔴 Strong"   if v > 0.5 else
                    "🟠 Moderate" if v > 0.3 else
                    "🟡 Weak"     if v > 0.1 else
                    "⚪ Negligible"
                )
                rows.append({"col1": col1, "col2": col2, "V": v, "strength": strength})

        summary_df = pd.DataFrame(rows).sort_values("V", ascending=False)

        # Print at least top 15 OR all non-negligible ones
        notable = summary_df[summary_df["V"] > 0.1]
        to_print = notable if not notable.empty else summary_df.head(10)
        for _, row in to_print.iterrows():
            print(f"  {row['col1']:<35} vs {row['col2']:<35} V={row['V']:.3f}  {row['strength']}")

        if notable.empty:
            print("  No notable categorical associations found (all V < 0.1).")

        # ── Heatmap ───────────────────────────────────────────────────────
        if save and len(cat_cols) >= 2:
            cell = max(0.7, min(1.2, 10.0 / len(cat_cols)))
            fig_size = max(6, len(cat_cols)) * cell
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

            sns.heatmap(
                v_matrix.astype(float),
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                vmin=0, vmax=1,
                square=True,
                linewidths=0.3,
                annot_kws={"size": max(6, min(9, int(70 / len(cat_cols))))},
            )
            ax.set_title("Cramér's V — Categorical Associations", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.tick_params(axis="y", rotation=0,  labelsize=8)
            plt.tight_layout()

            path = REPORTS_DIR / "plots" / "cramers_v_heatmap.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            print(f"\n  Saved: {path.name}")
            plt.close(fig)

        return v_matrix

    # ── Step 5: Point-Biserial (continuous vs binary flag) ───────────────────

    def point_biserial(self) -> pd.DataFrame:
        """
        Compute point-biserial correlation between binary flag columns
        and all numerical columns.

        Mathematically = Pearson r, but semantically appropriate when
        one variable is binary (0/1).

        Business interpretation for Olist:
            is_late_delivery  vs delivery_days    → expected positive r
            is_late_delivery  vs review_score     → expected negative r
            is_same_state     vs freight_value    → expected negative r

        Returns
        -------
        DataFrame with columns: binary_col, numerical_col, r, p_value, significant
        """
        self._ensure_loaded()
        self._section("Step 5 — Point-Biserial Correlations (Binary vs Numerical)")

        binary_cols = [c for c in self.BINARY_COLS if c in self.df.columns]
        if not binary_cols:
            print("No binary columns found. Skipping.")
            return pd.DataFrame()

        print(f"  Binary columns found: {binary_cols}")

        results = []
        for bin_col in binary_cols:
            for num_col in self.numerical_cols:
                if num_col == bin_col:
                    continue
                paired = self.df[[bin_col, num_col]].dropna()
                if len(paired) < 30:
                    continue
                # Verify binary column is actually 0/1 (could be label-encoded differently)
                unique_vals = paired[bin_col].unique()
                if len(unique_vals) > 2:
                    continue
                r, p = pointbiserialr(paired[bin_col], paired[num_col])
                results.append({
                    "binary_col":    bin_col,
                    "numerical_col": num_col,
                    "r":             round(r, 4),
                    "p_value":       round(p, 6),
                    "significant":   p < 0.05,
                    "n":             len(paired),
                })

        if not results:
            print("  No valid pairs found.")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        # ── Print results grouped by binary col ──────────────────────────
        print(f"\nPoint-biserial results (|r| > 0.1 AND p < 0.05):")
        for bin_col in binary_cols:
            subset = result_df[
                (result_df["binary_col"] == bin_col) &
                (result_df["r"].abs() > 0.1) &
                (result_df["significant"])
            ].sort_values("r", key=abs, ascending=False)

            if subset.empty:
                print(f"\n  {bin_col}: no strong correlations found (|r| > 0.1)")
                continue

            print(f"\n  ── {bin_col} ─────────────────")
            for _, row in subset.head(10).iterrows():
                direction = "↑" if row["r"] > 0 else "↓"
                print(
                    f"    {row['numerical_col']:<40} r={row['r']:+.4f} {direction}"
                    f"  p={row['p_value']:.4f}  n={row['n']:,}"
                )

        return result_df

    # ── Step 6: VIF (Variance Inflation Factor) ───────────────────────────────

    def compute_vif(self, vif_threshold: float = 5.0) -> pd.DataFrame:
        """
        Compute VIF for all numerical columns to detect multicollinearity.

        VIF_i = 1 / (1 - R²_i)
        where R²_i = coefficient of determination when column i is regressed
        on all other numerical columns.

        Interpretation:
            VIF = 1     → no multicollinearity
            1 < VIF < 5 → moderate (acceptable)
            VIF > 5     → high multicollinearity (investigate)
            VIF > 10    → severe (drop or combine features)

        Why this matters for Day 12+ modeling:
            Multicollinear features destabilize linear model coefficients.
            Tree-based models are less affected but feature importance
            gets split across correlated features.

        Strategy for high VIF:
            1. Drop the feature with higher VIF in a correlated pair
            2. Combine correlated features (e.g. ratio or sum)
            3. Use PCA to orthogonalize (covered Day 10)
            4. Use Ridge/Lasso which handles multicollinearity implicitly

        Parameters
        ----------
        vif_threshold : float
            Flag features at or above this VIF value. Default: 5.0

        Returns
        -------
        DataFrame with columns: feature, vif, flag
        """
        self._ensure_loaded()
        self._section("Step 6 — VIF (Multicollinearity Detection)")

        # Exclude target and binary cols from VIF (they are outcomes/flags)
        vif_cols = [
            c for c in self.numerical_cols
            if c != self.TARGET and c not in self.BINARY_COLS
        ]
        clean_df = self.df[vif_cols].dropna()

        if clean_df.shape[1] < 2:
            print("Need at least 2 numerical columns for VIF.")
            return pd.DataFrame()

        # Remove zero-variance columns (VIF is undefined for constants)
        varying_cols = [c for c in vif_cols if clean_df[c].std() > 0]
        removed = set(vif_cols) - set(varying_cols)
        if removed:
            print(f"  Removed zero-variance cols: {removed}")

        X = clean_df[varying_cols].values
        print(f"  Computing VIF for {len(varying_cols)} features on {len(clean_df):,} rows...")

        vif_records = []
        for i, col in enumerate(varying_cols):
            try:
                vif = variance_inflation_factor(X, i)
                vif_records.append({
                    "feature": col,
                    "vif":     round(vif, 2),
                    "flag":    (
                        "🔴 SEVERE ≥10" if vif >= 10 else
                        "🟠 HIGH ≥5"    if vif >= vif_threshold else
                        "✅ ok"
                    ),
                })
            except Exception as e:
                vif_records.append({"feature": col, "vif": np.nan, "flag": f"error: {e}"})

        self.vif_df = (
            pd.DataFrame(vif_records)
            .sort_values("vif", ascending=False)
            .reset_index(drop=True)
        )

        # ── Print summary ─────────────────────────────────────────────────
        severe   = self.vif_df[self.vif_df["vif"] >= 10]
        high     = self.vif_df[(self.vif_df["vif"] >= vif_threshold) & (self.vif_df["vif"] < 10)]
        ok_count = len(self.vif_df) - len(severe) - len(high)

        print(f"\n  VIF Summary:")
        print(f"    ✅ OK (VIF < {vif_threshold})  : {ok_count}")
        print(f"    🟠 High (≥{vif_threshold}, <10): {len(high)}")
        print(f"    🔴 Severe (≥10)         : {len(severe)}")

        print(f"\n  Top 20 VIF scores:")
        print(f"  {'Feature':<42} {'VIF':>8}  Flag")
        print(f"  {'─'*42} {'─'*8}  {'─'*15}")
        for _, row in self.vif_df.head(20).iterrows():
            vif_str = f"{row['vif']:.2f}" if not np.isnan(row["vif"]) else " NaN"
            print(f"  {row['feature']:<42} {vif_str:>8}  {row['flag']}")

        # ── Actionable recommendations ────────────────────────────────────
        if not severe.empty or not high.empty:
            print(f"\n  Recommendations:")
            for _, row in pd.concat([severe, high]).iterrows():
                action = (
                    "→ DROP or combine with correlated feature (Day 10 PCA)"
                    if row["vif"] >= 10 else
                    "→ Investigate correlation; consider dropping if redundant"
                )
                print(f"    {row['feature']:<42} VIF={row['vif']:.1f}  {action}")

        return self.vif_df

    # ── Step 7: Save Report ───────────────────────────────────────────────────

    def save_report(
        self,
        point_biserial_df: pd.DataFrame | None = None,
        cramers_v_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Save all computed matrices and stats to reports/bivariate/.

        Files saved:
            pearson_matrix.csv       - Full Pearson correlation matrix
            spearman_matrix.csv      - Full Spearman correlation matrix
            vif_scores.csv           - VIF per feature
            point_biserial.csv       - Binary vs numerical correlations
            cramers_v_matrix.csv     - Categorical association matrix
            summary.txt              - Human-readable Day 5 summary
        """
        self._ensure_loaded()
        self._section("Step 7 — Saving Reports")

        saved = []

        if self.pearson_matrix is not None:
            path = REPORTS_DIR / "pearson_matrix.csv"
            self.pearson_matrix.to_csv(path)
            saved.append(path.name)

        if self.spearman_matrix is not None:
            path = REPORTS_DIR / "spearman_matrix.csv"
            self.spearman_matrix.to_csv(path)
            saved.append(path.name)

        if self.vif_df is not None:
            path = REPORTS_DIR / "vif_scores.csv"
            self.vif_df.to_csv(path, index=False)
            saved.append(path.name)

        if point_biserial_df is not None and not point_biserial_df.empty:
            path = REPORTS_DIR / "point_biserial.csv"
            point_biserial_df.to_csv(path, index=False)
            saved.append(path.name)

        if cramers_v_df is not None and not cramers_v_df.empty:
            path = REPORTS_DIR / "cramers_v_matrix.csv"
            cramers_v_df.to_csv(path)
            saved.append(path.name)

        # ── Human-readable summary ────────────────────────────────────────
        summary_lines = [
            f"Day 5 — Bivariate Analysis Summary",
            f"Run at: {self._run_timestamp}",
            f"Dataset: {self.filepath.name}",
            f"Shape: {self.df.shape}",
            "",
        ]

        if self.spearman_matrix is not None and self.TARGET in self.spearman_matrix.columns:
            top5 = (
                self.spearman_matrix[self.TARGET]
                .drop(self.TARGET)
                .abs()
                .sort_values(ascending=False)
                .head(5)
            )
            summary_lines.append(f"Top 5 Spearman correlations with '{self.TARGET}':")
            for feat, val in top5.items():
                summary_lines.append(f"  {feat:<42} ρ={val:.3f}")
            summary_lines.append("")

        if self.vif_df is not None:
            severe = self.vif_df[self.vif_df["vif"] >= 10]
            high   = self.vif_df[(self.vif_df["vif"] >= 5) & (self.vif_df["vif"] < 10)]
            summary_lines.append(f"VIF: {len(severe)} severe (≥10), {len(high)} high (5–10)")
            if not severe.empty:
                for _, row in severe.iterrows():
                    summary_lines.append(f"  SEVERE: {row['feature']} (VIF={row['vif']:.1f})")
            summary_lines.append("")

        summary_lines.append(f"Files saved: {', '.join(saved)}")
        summary_lines.append(f"Plots dir  : {REPORTS_DIR / 'plots'}")

        summary_path = REPORTS_DIR / "summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        saved.append(summary_path.name)

        print(f"\nSaved {len(saved)} files to: {REPORTS_DIR}")
        for f in saved:
            print(f"  ✅ {f}")

        # ── Final console summary ─────────────────────────────────────────
        print("\n── Day 5 Complete ─────────────────────────────────────────")
        if self.spearman_matrix is not None and self.TARGET in self.spearman_matrix.columns:
            top3 = (
                self.spearman_matrix[self.TARGET]
                .drop(self.TARGET)
                .abs()
                .sort_values(ascending=False)
                .head(3)
            )
            print(f"\nKey finding — Top 3 predictors of '{self.TARGET}':")
            for rank, (feat, val) in enumerate(top3.items(), 1):
                print(f"  {rank}. {feat:<40} ρ={val:.3f}")

        if self.vif_df is not None:
            severe = self.vif_df[self.vif_df["vif"] >= 10]
            if not severe.empty:
                print(f"\n⚠️  Action needed — {len(severe)} features with VIF ≥ 10:")
                for _, row in severe.iterrows():
                    print(f"   Drop/combine: {row['feature']} (VIF={row['vif']:.1f})")
            else:
                print(f"\n✅ No severe multicollinearity detected.")

        print(f"\nNext → Day 6: Hypothesis Testing")
        print(f"  T-Test, ANOVA, Mann-Whitney U on business questions")
        print(f"  e.g. 'Does late delivery significantly lower review score?'")

    # ── Step 8: Full Pipeline Orchestrator ───────────────────────────────────

    def run_full_pipeline(
        self,
        top_n: int = 15,
        vif_threshold: float = 5.0,
        save_plots: bool = True,
    ) -> dict:
        """
        Run all Day 5 steps in sequence with one method call.

        Parameters
        ----------
        top_n         : Number of top-variance columns for correlation matrix
        vif_threshold : VIF cutoff for flagging multicollinearity
        save_plots    : Whether to save plots to disk

        Returns
        -------
        dict with keys: pearson, spearman, cramers_v, point_biserial, vif
        """
        self._section("Day 5 — Full Pipeline")
        print(f"  Config: top_n={top_n}, vif_threshold={vif_threshold}, save={save_plots}")

        self.load_data()
        self.correlation(top_n=top_n)
        self.plot_heatmap(save=save_plots)

        cramers_df = self.cramers_v(save=save_plots)
        pb_df      = self.point_biserial()
        vif_df     = self.compute_vif(vif_threshold=vif_threshold)

        self.save_report(point_biserial_df=pb_df, cramers_v_df=cramers_df)

        return {
            "pearson":         self.pearson_matrix,
            "spearman":        self.spearman_matrix,
            "cramers_v":       cramers_df,
            "point_biserial":  pb_df,
            "vif":             vif_df,
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        analyzer = BivariateAnalyzer()
        results  = analyzer.run_full_pipeline(
            top_n=15,
            vif_threshold=5.0,
            save_plots=True,
        )
        print(f"\n{analyzer}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise