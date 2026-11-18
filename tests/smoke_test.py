# tests/test_smoke.py
from sklearn import metrics

from churnlab.data import FEATURES, load_clean, make_split
from churnlab.evaluate import evaluate
from churnlab.model import build_pipeline


def test_pipeline_trains_and_evaluates():
    df = load_clean()
    train_df, val_df, _ = make_split(df)

    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURES], train_df["target"])
    metrics = evaluate(pipeline, val_df[FEATURES], val_df["target"])

    assert set(metrics) == {
        "val_average_precision", "val_precision", "val_recall", "val_f1",
    }
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["val_average_precision"] > 0.4 

    print(metrics)