"""
src/data/split.py
------------------
Patient-level train/val/test split.

The #1 data-leakage mistake in medical imaging: splitting by *image* means
the same patient's two X-rays can land one in train and one in test, so the
model partially memorizes patient-specific anatomy instead of learning the
disease signal -- inflating validation metrics in a way that won't hold up
on genuinely unseen patients. We split by `patient_id` instead, so every
image from a given patient stays entirely within one split.
"""
import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits by unique patient_id so no patient appears in more than one split."""
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1

    rng = np.random.default_rng(seed)
    patients = df[patient_col].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_df = df[df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].reset_index(drop=True)

    # Hard guarantee, not just a hope: assert zero patient overlap between splits.
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from pathlib import Path

    df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "labels.csv")
    train_df, val_df, test_df = patient_level_split(df)
    print(f"Train: {len(train_df)} images / {train_df['patient_id'].nunique()} patients")
    print(f"Val:   {len(val_df)} images / {val_df['patient_id'].nunique()} patients")
    print(f"Test:  {len(test_df)} images / {test_df['patient_id'].nunique()} patients")

    overlap_tv = set(train_df.patient_id) & set(val_df.patient_id)
    overlap_tt = set(train_df.patient_id) & set(test_df.patient_id)
    overlap_vt = set(val_df.patient_id) & set(test_df.patient_id)
    print(f"Patient overlap train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)}")
