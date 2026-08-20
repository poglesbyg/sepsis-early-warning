"""TensorFlow models.

Two networks, for two different reasons.

``build_mlp`` is a like-for-like comparison: same 345-column matrix as the
booster, different function class. It exists to answer whether a dense network
buys anything over gradient boosting on engineered tabular features. Usually it
does not, and reporting that honestly is more useful than omitting it.

``build_sequence_model`` is the one that does something the tabular models
cannot. It reads the admission as a sequence and learns its own temporal
representation instead of consuming hand-specified 6h and 24h windows.

Two constraints shape the architecture:

*Everything is causal.* No bidirectional layers, no non-causal convolutions. A
bidirectional GRU would score beautifully offline and be undeployable, because at
hour 20 it would be reading hour 40. The ``Conv1D`` layers use
``padding="causal"`` and the GRUs run forward only.

*Batches are bucketed by length, not padded to a global maximum.* Stays run from
8 hours to 336. Padding everything to 336 would make >90% of the compute mask,
and truncating to a fixed window would discard a quarter of the positive hours,
which arrive late in long stays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, CHANNELS, Config

STATIC_FEATURES = [
    "age", "gender", "unit_micu", "unit_sicu", "unit_unknown",
    "hosp_adm_time", "log_iculos",
]


def _column_stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and scale, safe for columns that are entirely missing.

    A channel nobody in this cohort ever had measured yields an empty slice; it
    is standardised to a constant zero and carried by its mask channel instead.
    """
    finite = np.isfinite(v)
    counts = finite.sum(axis=0)
    filled = np.where(finite, v, 0.0)
    mean = np.divide(filled.sum(axis=0), counts, out=np.zeros(v.shape[1]), where=counts > 0)
    var = np.divide(
        (np.where(finite, v - mean, 0.0) ** 2).sum(axis=0),
        counts,
        out=np.zeros(v.shape[1]),
        where=counts > 0,
    )
    std = np.sqrt(var)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _import_keras():
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    return tf, tf.keras


# --------------------------------------------------------------------------
# Sequence tensors
# --------------------------------------------------------------------------
class SequenceEncoder:
    """Turns the hourly frame into per-admission arrays and standardises them.

    Channel layout per hour: carried-forward values, then a binary
    "measured this hour" flag per channel, then the static covariates broadcast
    across time. The mask channels are what let the network distinguish a real
    reading from a stale carry-forward -- the same information the tabular
    recency features encode, handed over in a form an RNN can use.
    """

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or CHANNELS
        self.value_cols = [f"{c}_locf" for c in self.channels]
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return 2 * len(self.channels) + len(STATIC_FEATURES)

    def fit(self, frame: pd.DataFrame) -> "SequenceEncoder":
        self.mean_, self.std_ = _column_stats(frame[self.value_cols].to_numpy(dtype=np.float32))
        self.static_mean_, self.static_std_ = _column_stats(
            frame[STATIC_FEATURES].to_numpy(dtype=np.float32)
        )
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        """Return per-admission (features, labels, patient_id), ordered by hour."""
        frame = frame.sort_values(["patient_id", "hour"], ignore_index=True)

        v = frame[self.value_cols].to_numpy(dtype=np.float32)
        v = (v - self.mean_) / self.std_
        # A channel never measured for this patient has no meaningful value; zero
        # is the post-standardisation cohort mean, and the mask channel says so.
        observed = np.isfinite(v).astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        s = frame[STATIC_FEATURES].to_numpy(dtype=np.float32)
        s = np.nan_to_num((s - self.static_mean_) / self.static_std_)

        block = np.concatenate([v, observed, s], axis=1)
        y = frame["SepsisLabel"].to_numpy(dtype=np.float32)

        pids = frame["patient_id"].to_numpy()
        edges = np.flatnonzero(pids[1:] != pids[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(pids)]))
        xs, ys, ids = [], [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            xs.append(block[a:b])
            ys.append(y[a:b])
            ids.append(pids[a])
        return xs, ys, ids


def make_batches(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    batch_size: int = 64,
    positive_weight: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Length-bucketed padded batches.

    Admissions are sorted by length before batching, so each batch pads to its own
    longest member rather than to the cohort maximum. On this dataset that cuts
    padded timesteps by roughly an order of magnitude versus a fixed window.
    """
    order = np.argsort([len(x) for x in xs])
    batches = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        T = max(len(xs[i]) for i in idx)
        n_feat = xs[idx[0]].shape[1]
        xb = np.zeros((len(idx), T, n_feat), dtype=np.float32)
        yb = np.zeros((len(idx), T, 1), dtype=np.float32)
        wb = np.zeros((len(idx), T), dtype=np.float32)
        for r, i in enumerate(idx):
            n = len(xs[i])
            xb[r, :n] = xs[i]
            yb[r, :n, 0] = ys[i]
            # Padded steps get weight 0; real positive hours get upweighted to
            # offset a 1.8% positive rate.
            wb[r, :n] = np.where(ys[i] > 0, positive_weight, 1.0)
        batches.append((xb, yb, wb))
    return batches


class BatchSequence:
    """Keras data source over pre-built batches, reshuffled every epoch."""

    def __init__(self, batches, shuffle=True, seed=0):
        self.batches = batches
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(batches))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
        for i in self.order:
            yield self.batches[i]


def as_keras_dataset(batches, n_features: int, shuffle: bool = True, seed: int = 0):
    tf, _ = _import_keras()
    gen = BatchSequence(batches, shuffle=shuffle, seed=seed)
    signature = (
        tf.TensorSpec(shape=(None, None, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(lambda: iter(gen), output_signature=signature)
    # ``repeat`` plus an explicit ``steps_per_epoch`` at fit time: without it Keras
    # exhausts the generator after one pass and warns that it ran out of data.
    return ds.repeat().prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------
def build_sequence_model(
    n_features: int,
    conv_filters: int = 64,
    gru_units: tuple[int, int] = (96, 48),
    dropout: float = 0.25,
    learning_rate: float = 1e-3,
):
    """Causal Conv1D front end into stacked GRUs, one risk score per hour.

    The convolutions pick up short-horizon shape -- a rising respiratory rate over
    three hours -- and the recurrent layers carry longer-range state, such as how
    far this patient has drifted from where they started. ``padding="causal"``
    is what keeps the convolution from peeking one step ahead.
    """
    tf, keras = _import_keras()
    L = keras.layers

    inp = L.Input(shape=(None, n_features), name="hourly_channels")
    x = L.Conv1D(conv_filters, 3, padding="causal", activation="relu")(inp)
    x = L.LayerNormalization()(x)
    x = L.Conv1D(conv_filters, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = L.LayerNormalization()(x)
    x = L.Dropout(dropout)(x)
    x = L.GRU(gru_units[0], return_sequences=True)(x)
    x = L.Dropout(dropout)(x)
    x = L.GRU(gru_units[1], return_sequences=True)(x)
    out = L.Dense(1, activation="sigmoid", name="hourly_risk")(x)

    model = keras.Model(inp, out, name="causal_gru")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        weighted_metrics=[keras.metrics.AUC(name="auprc", curve="PR")],
    )
    return model


def build_mlp(n_features: int, units=(256, 128, 64), dropout: float = 0.3, learning_rate: float = 1e-3):
    """Dense network on the engineered tabular matrix, for a same-inputs comparison."""
    tf, keras = _import_keras()
    L = keras.layers

    inp = L.Input(shape=(n_features,))
    x = inp
    for u in units:
        x = L.Dense(u, activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name="tabular_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        weighted_metrics=[keras.metrics.AUC(name="auprc", curve="PR")],
    )
    return model


# --------------------------------------------------------------------------
# Training / prediction
# --------------------------------------------------------------------------
def train_sequence_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    cfg: Config = CFG,
    epochs: int = 25,
    batch_size: int = 64,
    positive_weight: float | None = None,
    verbose: int = 0,
):
    tf, keras = _import_keras()
    keras.utils.set_random_seed(cfg.seed)

    encoder = SequenceEncoder().fit(train_frame)
    xtr, ytr, _ = encoder.transform(train_frame)
    xva, yva, _ = encoder.transform(valid_frame)

    if positive_weight is None:
        rate = train_frame["SepsisLabel"].mean()
        # Square-root tempering rather than full inverse frequency: weighting
        # positives 45:1 destabilises the recurrent layers and pushes calibration
        # far past anything the isotonic step can pull back.
        positive_weight = float(((1 - rate) / rate) ** 0.5)

    train_batches = make_batches(xtr, ytr, batch_size, positive_weight)
    valid_batches = make_batches(xva, yva, batch_size, positive_weight)

    model = build_sequence_model(encoder.n_features)
    history = model.fit(
        as_keras_dataset(train_batches, encoder.n_features, shuffle=True, seed=cfg.seed),
        validation_data=as_keras_dataset(valid_batches, encoder.n_features, shuffle=False),
        steps_per_epoch=len(train_batches),
        validation_steps=len(valid_batches),
        epochs=epochs,
        shuffle=False,  # BatchSequence reshuffles batch order itself each epoch
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auprc", mode="max", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auprc", mode="max", factor=0.5, patience=2, min_lr=1e-5
            ),
        ],
    )
    return model, encoder, history


def predict_sequences(model, encoder: SequenceEncoder, frame: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    """Hourly risk scores aligned to ``frame`` sorted by (patient_id, hour)."""
    xs, ys, _ = encoder.transform(frame)
    order = np.argsort([len(x) for x in xs])
    out: list[tuple[int, np.ndarray]] = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        T = max(len(xs[i]) for i in idx)
        xb = np.zeros((len(idx), T, xs[idx[0]].shape[1]), dtype=np.float32)
        for r, i in enumerate(idx):
            xb[r, : len(xs[i])] = xs[i]
        preds = model.predict(xb, verbose=0)[..., 0]
        for r, i in enumerate(idx):
            out.append((i, preds[r, : len(xs[i])]))
    out.sort(key=lambda t: t[0])
    return np.concatenate([p for _, p in out])
