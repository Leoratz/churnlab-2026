# ChurnLab 2026

## 1. Business question

The goal of this project is to predict whether a customer will leave a company (customer churn) based on their characteristics and service information.

The target column is `Churn`. It indicates whether a customer has left the company. It must not be used as a model input because it is the value we want to predict.

## 2. Python environment

This project requires Python 3.11 or later.

To install the dependencies and set up the environment, run:

```bash
uv sync
```

To run Python commands in the project environment, use:

```bash
uv run python --version
```

The dependencies are declared in `pyproject.toml`, and their resolved versions are recorded in `uv.lock`.

## 3. Dataset

The project uses the IBM Telco Customer Churn dataset.

- **Source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **SHA-256:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Local file:** `data/raw/telco.csv`
- **Source metadata:** `data/raw/telco.source.json`

To download and verify the dataset, run:

```bash
uv run python scripts/download_data.py
```

To inspect the dataset, run:

```bash
uv run python -c "import pandas as pd; df = pd.read_csv('data/raw/telco.csv'); print(df.info()); print(df.head()); print(df['Churn'].value_counts(dropna=False))"
```

The SHA-256 hash identifies the exact file contents. It does not guarantee that the labels are correct or that the dataset is free of data leakage.

## 4. Project structure

- `src/`: Python source code for the project.
- `scripts/`: utility scripts, including the dataset download script.
- `data/raw/`: original dataset and its source metadata.
- `evidence/tp01/`: evidence and reports retained for the TP01 submission.
- `reports/`: generated reports, excluded from Git.
- `tests/`: automated tests.

Some directories may be created as the project develops.

## 5. Dataset limitations

This dataset is a snapshot of customer information and churn labels. It does not necessarily provide the exact event times when customers decided to leave or the times when predictions would have been made.

It is not an event-time dataset that records a chronological sequence of customer events, nor an intervention dataset that records actions taken by a company and their effects on customers.

Therefore, this dataset alone cannot establish whether an intervention, such as a retention offer, would prevent a customer from leaving. Such questions would require appropriate timestamps and intervention/outcome data.