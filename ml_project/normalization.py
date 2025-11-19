def normalize_column(df, column):
    """Scales a column to the range [0, 1]."""
    if column not in df.columns:
        raise KeyError(f"{column} not found")
    col = df[column]
    return (col - col.min()) / (col.max() - col.min())