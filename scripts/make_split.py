from churnlab.data import load_clean, make_split, save_split

df = load_clean()
train_df, val_df, test_df = make_split(df)
save_split(train_df, val_df, test_df)

for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"{name:<11} n={len(part):>5}  churn rate={part['target'].mean():.4f}")