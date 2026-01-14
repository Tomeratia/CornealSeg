#!/usr/bin/env python3
"""
Build dataset index for corneal ulcer segmentation.
Creates reproducible train/val/test splits with stratification by grade.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_numeric_id(filename: str) -> int | None:
    """Extract numeric ID from filename like '12.jpg', '001.png', etc."""
    match = re.match(r"(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None


def get_image_ids_from_dir(directory: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> set[int]:
    """Get set of numeric IDs for images in directory."""
    ids = set()
    for ext in extensions:
        for f in directory.glob(f"*{ext}"):
            num_id = parse_numeric_id(f.name)
            if num_id is not None:
                ids.add(num_id)
    return ids


def load_excel_metadata(excel_path: Path) -> pd.DataFrame:
    """Load and parse Excel metadata, extracting numeric IDs."""
    df = pd.read_excel(excel_path)
    
    # First column contains filenames like "12.jpg"
    filename_col = df.columns[0]
    
    # Extract numeric ID from filename
    df["id"] = df[filename_col].apply(parse_numeric_id)
    
    # Drop rows without valid ID
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    
    # Handle duplicates: keep first occurrence
    duplicate_ids = df[df.duplicated(subset=["id"], keep="first")]["id"].tolist()
    if duplicate_ids:
        warnings.warn(f"Found {len(duplicate_ids)} duplicate IDs in Excel, keeping first occurrence: {duplicate_ids[:10]}...")
        df = df.drop_duplicates(subset=["id"], keep="first")
    
    return df


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float, tol: float = 1e-6) -> None:
    """Validate that split ratios sum to 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tol:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def stratified_split_with_fallback(
    ids: list[int],
    grades: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Perform stratified split by grade if possible, otherwise fall back to random.
    Returns (train_ids, val_ids, test_ids).
    """
    # Count samples per grade
    grade_counts = pd.Series(grades).value_counts()
    min_count = grade_counts.min()
    
    # Need at least 2 samples per grade for stratification to work with train_test_split
    try_stratified = min_count >= 2
    
    if try_stratified:
        try:
            # First split: separate test set
            train_val_ids, test_ids, train_val_grades, _ = train_test_split(
                ids, grades,
                test_size=test_ratio,
                stratify=grades,
                random_state=seed,
            )
            
            # Second split: separate train and val from remaining
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=val_ratio_adjusted,
                stratify=train_val_grades,
                random_state=seed,
            )
            
            return train_ids, val_ids, test_ids
            
        except ValueError as e:
            warnings.warn(f"Stratified split failed ({e}), falling back to random split.")
    else:
        warnings.warn(
            f"Stratification not possible: minimum samples per grade = {min_count}. "
            "Falling back to random split."
        )
    
    # Fallback: random split without stratification
    train_val_ids, test_ids = train_test_split(
        ids,
        test_size=test_ratio,
        random_state=seed,
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio_adjusted,
        random_state=seed,
    )
    
    return train_ids, val_ids, test_ids


def print_distribution_table(df: pd.DataFrame, split_col: str, group_col: str, title: str) -> None:
    """Print distribution table for a given grouping."""
    print(f"\n{title}")
    print("-" * 50)
    pivot = pd.crosstab(df[group_col], df[split_col], margins=True, margins_name="Total")
    # Reorder columns
    col_order = [c for c in ["train", "val", "test", "Total"] if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset index for corneal ulcer segmentation."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory containing rawImages/, ulcerLabels/, and Category information.xlsx",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: data_root/outputs)",
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Set up paths
    data_root = args.data_root.resolve()
    raw_images_dir = data_root / "rawImages"
    ulcer_labels_dir = data_root / "ulcerLabels"
    excel_path = data_root / "Category information.xlsx"
    out_dir = (args.out_dir or data_root / "outputs").resolve()
    
    # Validate paths exist
    for path, name in [(raw_images_dir, "rawImages"), (ulcer_labels_dir, "ulcerLabels"), (excel_path, "Excel file")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Load data
    print("=" * 60)
    print("BUILDING DATASET INDEX")
    print("=" * 60)
    
    image_ids = get_image_ids_from_dir(raw_images_dir)
    mask_ids = get_image_ids_from_dir(ulcer_labels_dir)
    excel_df = load_excel_metadata(excel_path)
    excel_id_set = set(excel_df["id"])
    
    print(f"\n[1] Raw counts:")
    print(f"    - rawImages:   {len(image_ids)}")
    print(f"    - ulcerLabels: {len(mask_ids)}")
    print(f"    - Excel rows:  {len(excel_df)}")
    
    # Intersection: IDs present in both images and masks
    common_ids = image_ids & mask_ids
    print(f"\n[2] After intersection (image ∩ mask):")
    print(f"    - Common IDs:  {len(common_ids)}")
    
    # Report IDs in masks but missing in images
    mask_only = mask_ids - image_ids
    if mask_only:
        print(f"    - Masks without images (dropped): {len(mask_only)}")
    
    # Report IDs in images but missing in masks
    image_only = image_ids - mask_ids
    if image_only:
        print(f"    - Images without masks (dropped): {len(image_only)}")
    
    # Filter to IDs in Excel
    ids_missing_excel = common_ids - excel_id_set
    if ids_missing_excel:
        print(f"    - IDs not in Excel (dropped): {len(ids_missing_excel)}")
    
    valid_ids = common_ids & excel_id_set
    
    # Build working dataframe
    working_df = excel_df[excel_df["id"].isin(valid_ids)].copy()
    
    # Rename columns for consistency
    col_map = {}
    for col in working_df.columns:
        if col.lower() == "category":
            col_map[col] = "category"
        elif col.lower() == "type":
            col_map[col] = "type"
        elif col.lower() == "grade":
            col_map[col] = "grade"
    working_df = working_df.rename(columns=col_map)
    
    # Filter by Category (1 or 2 only)
    invalid_categories = working_df[~working_df["category"].isin([0, 1, 2])]
    if len(invalid_categories) > 0:
        print(f"    - Invalid category values (dropped): {len(invalid_categories)}")
        working_df = working_df[working_df["category"].isin([0, 1, 2])]
    
    # Category 0 excluded (no ulcer masks for Category 0)
    cat_0_count = len(working_df[working_df["category"] == 0])
    working_df = working_df[working_df["category"].isin([1, 2])]
    
    print(f"\n[3] After Category filter (1, 2 only):")
    print(f"    - Category 0 excluded: {cat_0_count}")
    print(f"    - Final sample count:  {len(working_df)}")
    
    if len(working_df) == 0:
        print("\nERROR: No valid samples remain after filtering!")
        sys.exit(1)
    
    # Perform split
    ids_list = working_df["id"].tolist()
    grades_list = working_df["grade"].tolist()
    
    train_ids, val_ids, test_ids = stratified_split_with_fallback(
        ids_list, grades_list,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed,
    )
    
    # Assign splits
    id_to_split = {}
    for id_ in train_ids:
        id_to_split[id_] = "train"
    for id_ in val_ids:
        id_to_split[id_] = "val"
    for id_ in test_ids:
        id_to_split[id_] = "test"
    
    working_df["split"] = working_df["id"].map(id_to_split)
    
    print(f"\n[4] Split sizes:")
    print(f"    - Train: {len(train_ids)} ({len(train_ids)/len(working_df)*100:.1f}%)")
    print(f"    - Val:   {len(val_ids)} ({len(val_ids)/len(working_df)*100:.1f}%)")
    print(f"    - Test:  {len(test_ids)} ({len(test_ids)/len(working_df)*100:.1f}%)")
    
    # Distribution tables
    print_distribution_table(working_df, "split", "category", "Distribution by Category")
    print_distribution_table(working_df, "split", "grade", "Distribution by Grade")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Select and order output columns (without paths)
    output_cols = ["id", "category", "type", "grade", "split"]
    output_df = working_df[output_cols].sort_values("id").reset_index(drop=True)
    
    # Save CSV
    csv_path = out_dir / "dataset_index.csv"
    output_df.to_csv(csv_path, index=False)
    print(f"\n[5] Saved outputs to: {out_dir}")
    print(f"    - {csv_path.name}")
    
    # Save ID text files
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        txt_path = out_dir / f"{split_name}_ids.txt"
        with open(txt_path, "w") as f:
            for id_ in sorted(split_ids):
                f.write(f"{id_}\n")
        print(f"    - {txt_path.name}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
