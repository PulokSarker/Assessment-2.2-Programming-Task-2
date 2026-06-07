import numpy as np
import csv
import os


def load_csv(
    path,
    delimiter=",",
    has_header=True,
    target_col=-1,
    dtype=float,
    skip_rows=0,
    encoding="utf-8",
):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")

    rows = []
    feature_names = None

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i == 0 and has_header:
                feature_names = row
                continue
            if i < skip_rows + (1 if has_header else 0):
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path!r}")

    # Convert to numpy, coercing to float (NaN for non-numeric)
    data = []
    for row in rows:
        numeric_row = []
        for val in row:
            try:
                numeric_row.append(float(val))
            except (ValueError, TypeError):
                numeric_row.append(np.nan)
        data.append(numeric_row)

    data = np.array(data, dtype=float)
    n_cols = data.shape[1]
    col = target_col if target_col >= 0 else n_cols + target_col

    X = np.delete(data, col, axis=1)
    y = data[:, col]

    if feature_names is not None:
        feat_names = [name for i, name in enumerate(feature_names) if i != col]
    else:
        feat_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, y, feat_names


def save_csv(
    X,
    y,
    path,
    feature_names=None,
    target_name="target",
    delimiter=",",
    encoding="utf-8",
):
    
    X = np.array(X, dtype=float)
    y = np.array(y)
    n_features = X.shape[1] if X.ndim > 1 else 1

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(feature_names + [target_name])
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])


def stream_csv(
    path,
    chunk_size=100,
    delimiter=",",
    has_header=True,
    target_col=-1,
    encoding="utf-8",
):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header_skipped = False
        n_cols = None
        col = None
        buffer = []

        for row in reader:
            if not header_skipped and has_header:
                header_skipped = True
                continue

            # Coerce
            numeric_row = []
            for val in row:
                try:
                    numeric_row.append(float(val))
                except (ValueError, TypeError):
                    numeric_row.append(np.nan)
            buffer.append(numeric_row)

            if len(buffer) >= chunk_size:
                data = np.array(buffer, dtype=float)
                if col is None:
                    n_cols = data.shape[1]
                    col = target_col if target_col >= 0 else n_cols + target_col
                X = np.delete(data, col, axis=1)
                y = data[:, col]
                yield X, y
                buffer = []

        # Yield remaining rows
        if buffer:
            data = np.array(buffer, dtype=float)
            if col is None:
                n_cols = data.shape[1]
                col = target_col if target_col >= 0 else n_cols + target_col
            X = np.delete(data, col, axis=1)
            y = data[:, col]
            yield X, y


def generate_synthetic_dataset(
    n_samples=1000,
    n_features=5,
    n_classes=3,
    noise=0.1,
    random_state=42,
    save_path=None,
):
    
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    # Assign class based on weighted sum of features
    weights = rng.standard_normal((n_features, n_classes))
    scores = X @ weights + rng.standard_normal((n_samples, n_classes)) * noise
    y = np.argmax(scores, axis=1)

    if save_path is not None:
        save_csv(X, y, save_path)
    return X, y