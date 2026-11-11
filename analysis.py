"""
Used-car sale price regression and auction buy/pass rule.

Reproduces every number used in the assignment slides.
Run:  .venv/bin/python analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30
RECON_COST = 1500

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Accident"] = out["Accident"].str.strip().str.lower()
    return out


def predict_price(model, cars: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(cars))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def apply_rule(predicted_sale, auction_price) -> pd.Series:
    """Buy if predicted lot price covers auction bid plus reconditioning."""
    return predicted_sale > auction_price + RECON_COST


def score_rule(lots: pd.DataFrame, buy: pd.Series) -> dict:
    bought = lots.loc[buy].copy()
    n_bought = int(buy.sum())
    if n_bought == 0:
        return {
            "n_bought": 0,
            "n_losers": 0,
            "total_profit": 0.0,
            "profit_per_bought": np.nan,
        }
    profit = bought["Later_Sale_Price"] - bought["Auction_Price"] - RECON_COST
    return {
        "n_bought": n_bought,
        "n_losers": int((profit < 0).sum()),
        "total_profit": float(profit.sum()),
        "profit_per_bought": float(profit.mean()),
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    sales = prep(pd.read_csv(ROOT / "used_car_sales.csv"))
    lots = prep(pd.read_csv(ROOT / "auction_lots.csv"))

    train, test = train_test_split(
        sales, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train = train.copy()
    test = test.copy()

    full_formula = (
        "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners"
    )
    reduced_formula = (
        "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)"
    )

    full = smf.ols(full_formula, data=train).fit()
    reduced = smf.ols(reduced_formula, data=train).fit()

    # Hand over the version with higher training adjusted R².
    if reduced.rsquared_adj >= full.rsquared_adj:
        v1 = reduced
        v1_name = "v1 (dropped Num_Owners)"
        dropped = "Num_Owners"
    else:
        v1 = full
        v1_name = "v1 (all features)"
        dropped = "none"

    train_mean = float(train["Sale_Price"].mean())

    y_test = test["Sale_Price"]
    pred_v1_test = predict_price(v1, test)
    pred_base_test = np.full(len(test), train_mean)

    test_base_r2 = r2_score(y_test, pred_base_test)
    test_v1_r2 = r2_score(y_test, pred_v1_test)
    test_base_rmse = rmse(y_test, pred_base_test)
    test_v1_rmse = rmse(y_test, pred_v1_test)

    lots = lots.copy()
    lots["pred_v1"] = predict_price(v1, lots)
    lots["pred_base"] = train_mean
    lots["buy_v1"] = apply_rule(lots["pred_v1"], lots["Auction_Price"])
    lots["buy_base"] = apply_rule(lots["pred_base"], lots["Auction_Price"])
    lots["realized_profit"] = (
        lots["Later_Sale_Price"] - lots["Auction_Price"] - RECON_COST
    )

    base_score = score_rule(lots, lots["buy_base"])
    v1_score = score_rule(lots, lots["buy_v1"])

    # --- console report (source of slide numbers) ---
    print("=" * 72)
    print("DATA")
    print("=" * 72)
    print(f"Sales file: used_car_sales.csv  n={len(sales)} cars sold last year")
    print(f"Split: random {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, seed={RANDOM_STATE}")
    print(f"  training n={len(train)}  test n={len(test)}")
    print(f"Auction file: auction_lots.csv  n={len(lots)} lots (held out of the fit)")
    print(f"Training-set average sale price (baseline prediction): ${train_mean:,.2f}")

    print("\n" + "=" * 72)
    print("TRAINING FIT — full model")
    print("=" * 72)
    print(full.summary())
    print(f"\nFull-model adjusted R²: {full.rsquared_adj:.4f}")

    print("\n" + "=" * 72)
    print("TRAINING FIT — without Num_Owners")
    print("=" * 72)
    print(reduced.summary())
    print(f"\nReduced-model adjusted R²: {reduced.rsquared_adj:.4f}")
    print(f"\nHanded-over model: {v1_name}")
    print(f"Feature dropped: {dropped}")
    print(f"v1 training adjusted R²: {v1.rsquared_adj:.4f}")

    print("\nCoefficient reading (v1, dollars of sale price):")
    params = v1.params
    pvals = v1.pvalues
    for name in params.index:
        print(
            f"  {name:24s}  {params[name]:10.2f}   p={pvals[name]:.4g}"
        )

    print("\n" + "=" * 72)
    print("TEST-SET FIT  (cars the model never saw)")
    print("=" * 72)
    print(f"{'':22s}  {'R²':>10s}  {'RMSE ($)':>12s}")
    print(f"{'Baseline (train mean)':22s}  {test_base_r2:10.3f}  {test_base_rmse:12,.0f}")
    print(f"{'v1':22s}  {test_v1_r2:10.3f}  {test_v1_rmse:12,.0f}")

    print("\n" + "=" * 72)
    print("AUCTION DECISION  (40 lots, not used in the fit)")
    print("=" * 72)
    print("Rule: BUY if predicted sale price > auction price + $1,500")
    print(f"{'':22s}  {'Bought':>8s}  {'Losers':>8s}  {'Total profit':>14s}")
    print(
        f"{'Baseline':22s}  {base_score['n_bought']:8d}  "
        f"{base_score['n_losers']:8d}  ${base_score['total_profit']:13,.0f}"
    )
    print(
        f"{'v1':22s}  {v1_score['n_bought']:8d}  "
        f"{v1_score['n_losers']:8d}  ${v1_score['total_profit']:13,.0f}"
    )
    print(
        f"Profit lift (v1 − baseline): "
        f"${v1_score['total_profit'] - base_score['total_profit']:,.0f}"
    )

    # --- artifacts ---
    lots_out = lots[
        [
            "Lot",
            "Mileage_K",
            "Age_Years",
            "Horsepower",
            "Accident",
            "Num_Owners",
            "Auction_Price",
            "Later_Sale_Price",
            "pred_base",
            "pred_v1",
            "buy_base",
            "buy_v1",
            "realized_profit",
        ]
    ]
    lots_out.to_csv(OUT_DIR / "auction_decisions.csv", index=False)

    coef_table = pd.DataFrame(
        {
            "term": v1.params.index,
            "coefficient": v1.params.values,
            "std_err": v1.bse.values,
            "p_value": v1.pvalues.values,
        }
    )
    coef_table.to_csv(OUT_DIR / "v1_coefficients.csv", index=False)

    # Chart 1: predicted vs actual on the test cars
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    lo = min(y_test.min(), pred_v1_test.min(), train_mean) - 1500
    hi = max(y_test.max(), pred_v1_test.max(), train_mean) + 1500
    ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=1, label="Perfect prediction")
    ax.scatter(
        y_test,
        pred_base_test,
        s=28,
        c="#bbbbbb",
        label="Baseline (same guess for every car)",
        zorder=2,
    )
    ax.scatter(
        y_test,
        pred_v1_test,
        s=36,
        c="#1f4e79",
        label="v1",
        zorder=3,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted sale price ($)")
    ax.set_title("Test cars: predicted vs actual sale price")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_predicted_vs_actual.png", dpi=200)
    plt.close(fig)

    # Chart 2: auction value
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))

    labels = ["Baseline", "v1"]
    bought = [base_score["n_bought"], v1_score["n_bought"]]
    losers = [base_score["n_losers"], v1_score["n_losers"]]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, bought, w, color="#1f4e79", label="Cars bought")
    axes[0].bar(x + w / 2, losers, w, color="#c0504d", label="Money-losing buys")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Number of cars")
    axes[0].set_title("What each rule buys at auction")
    axes[0].legend(frameon=False)
    style_axes(axes[0])

    profits = [base_score["total_profit"], v1_score["total_profit"]]
    colors = ["#888888", "#1f4e79"]
    axes[1].bar(labels, profits, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Total profit ($)")
    axes[1].set_title("Profit on the 40 auction lots")
    style_axes(axes[1])
    for i, val in enumerate(profits):
        offset = 400 if val >= 0 else -1200
        axes[1].text(i, val + offset, f"${val:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "auction_value.png", dpi=200)
    plt.close(fig)

    print("\nWrote:")
    print(f"  {FIG_DIR / 'test_predicted_vs_actual.png'}")
    print(f"  {FIG_DIR / 'auction_value.png'}")
    print(f"  {OUT_DIR / 'auction_decisions.csv'}")
    print(f"  {OUT_DIR / 'v1_coefficients.csv'}")


if __name__ == "__main__":
    main()
