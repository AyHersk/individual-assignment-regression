# Used car prices: should the dealer buy this car?

Linear regression of lot sale price, then a buy/pass rule at auction.

## Reproduce the numbers

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

`analysis.py` fits the model, evaluates it on held-out cars, applies the auction rule, and writes:

- `figures/test_predicted_vs_actual.png`
- `figures/auction_value.png`
- `outputs/v1_coefficients.csv`
- `outputs/auction_decisions.csv`

The random split uses `random_state=42` (70% train / 30% test). Every printed number is determined by that seed.

## Data

- `used_car_sales.csv` — 240 cars the dealer sold in the last year. Used to fit and evaluate the model.
- `auction_lots.csv` — 40 recent auction lots. Not used in the fit. Used only to score the buy/pass rule.
