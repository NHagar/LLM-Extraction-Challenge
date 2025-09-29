import json
from typing import Dict, Tuple

import pandas as pd
from difflib import SequenceMatcher
import re


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching by removing extra spaces, articles, and normalizing case"""
    if not text or text == "<PARSING ERROR>":
        return text

    # Convert to lowercase and strip
    normalized = text.lower().strip()

    # Remove common articles and prefixes
    normalized = re.sub(r'^(the|a|an)\s+', '', normalized)

    # Normalize multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized)

    # Remove common suffixes that might vary
    normalized = re.sub(r'\s+(inc|llc|corp|committee|for\s+\w+)\.?$', '', normalized)

    return normalized.strip()


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts using multiple metrics"""
    if not text1 or not text2 or text1 == "<PARSING ERROR>" or text2 == "<PARSING ERROR>":
        return 0.0

    # Exact match after normalization
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if norm1 == norm2:
        return 1.0

    # SequenceMatcher ratio (accounts for character-level differences)
    sequence_ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    # Check for missing space patterns (e.g., "Rosenfor" vs "Rosen for")
    missing_space_score = 0.0
    text1_no_spaces = re.sub(r'\s+', '', text1.lower())
    text2_no_spaces = re.sub(r'\s+', '', text2.lower())
    if text1_no_spaces == text2_no_spaces:
        missing_space_score = 0.95  # Very high similarity for missing spaces

    # Return the highest similarity score
    return max(sequence_ratio, missing_space_score)


def is_close_match(inferred: str, expected: str, similarity_threshold: float = 0.85) -> Tuple[bool, float]:
    """
    Determine if two committee names are close matches that should be automatically accepted.

    Returns:
        Tuple of (is_close_match, similarity_score)
    """
    similarity = calculate_similarity_score(inferred, expected)
    return similarity >= similarity_threshold, similarity


def calculate_accuracy_metrics(df: pd.DataFrame, include_close_matches: bool = False) -> Dict:
    """Calculate overall accuracy, parsing error rate, and mismatch statistics"""

    # Handle NaN values by converting to string
    df["committee_name_inferred"] = (
        df["committee_name_inferred"].fillna("<PARSING ERROR>").astype(str)
    )
    df["committee_name_expected"] = df["committee_name_expected"].fillna("").astype(str)

    # Calculate overall accuracy (case-insensitive exact match)
    df["is_exact_match"] = df.apply(
        lambda row: row["committee_name_inferred"].lower().strip()
        == row["committee_name_expected"].lower().strip(),
        axis=1,
    )

    # Identify parsing errors
    df["is_parsing_error"] = df["committee_name_inferred"] == "<PARSING ERROR>"

    # Calculate close matches if requested
    close_matches = 0
    if include_close_matches:
        df["is_close_match"] = df.apply(
            lambda row: is_close_match(
                row["committee_name_inferred"], row["committee_name_expected"]
            )[0] if not row["is_exact_match"] and not row["is_parsing_error"] else False,
            axis=1,
        )
        close_matches = df["is_close_match"].sum()

    # Calculate metrics
    total_inferences = len(df)
    exact_matches = df["is_exact_match"].sum()
    parsing_errors = df["is_parsing_error"].sum()

    # Total correct includes exact matches and close matches if enabled
    total_correct = exact_matches + (close_matches if include_close_matches else 0)

    # Mismatches are those that are not exact matches, not close matches, and not parsing errors
    mismatches = total_inferences - total_correct - parsing_errors

    metrics = {
        "total_inferences": total_inferences,
        "exact_matches": exact_matches,
        "parsing_errors": parsing_errors,
        "mismatches": mismatches,
        "accuracy_rate": total_correct / total_inferences
        if total_inferences > 0
        else 0,
        "parsing_error_rate": parsing_errors / total_inferences
        if total_inferences > 0
        else 0,
        "mismatch_rate": mismatches / total_inferences if total_inferences > 0 else 0,
    }

    if include_close_matches:
        metrics["close_matches"] = close_matches
        metrics["close_match_rate"] = close_matches / total_inferences if total_inferences > 0 else 0

    return metrics


def analyze_by_groups(df: pd.DataFrame) -> Dict:
    """Analyze metrics by model and prompt type"""
    results = {}

    # Group by model and prompt_type
    for (model, prompt_type), group in df.groupby(["model", "prompt_type"]):
        metrics = calculate_accuracy_metrics(group)
        results[f"{model}_{prompt_type}"] = metrics

    return results


def get_mismatches_for_review(df: pd.DataFrame, exclude_close_matches: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get mismatches for manual review, optionally excluding close matches

    Returns:
        Tuple of (mismatches_for_review, close_matches_auto_accepted)
    """

    # Handle NaN values by converting to string
    df["committee_name_inferred"] = (
        df["committee_name_inferred"].fillna("<PARSING ERROR>").astype(str)
    )
    df["committee_name_expected"] = df["committee_name_expected"].fillna("").astype(str)

    # Add helper columns
    df["is_exact_match"] = df.apply(
        lambda row: row["committee_name_inferred"].lower().strip()
        == row["committee_name_expected"].lower().strip(),
        axis=1,
    )
    df["is_parsing_error"] = df["committee_name_inferred"] == "<PARSING ERROR>"

    # Filter to mismatches only (non-exact matches that aren't parsing errors)
    mismatches = df[~df["is_exact_match"] & ~df["is_parsing_error"]].copy()

    if mismatches.empty:
        return mismatches, pd.DataFrame()

    # Calculate similarity scores and identify close matches
    similarity_data = mismatches.apply(
        lambda row: is_close_match(row["committee_name_inferred"], row["committee_name_expected"]),
        axis=1,
        result_type='expand'
    )

    mismatches["is_close_match"] = similarity_data[0]
    mismatches["similarity_score"] = similarity_data[1]

    # Split into close matches (auto-accepted) and remaining mismatches
    if exclude_close_matches:
        close_matches_auto = mismatches[mismatches["is_close_match"]].copy()
        mismatches_for_review = mismatches[~mismatches["is_close_match"]].copy()
    else:
        close_matches_auto = pd.DataFrame()
        mismatches_for_review = mismatches.copy()

    # Add length difference for sorting
    if not mismatches_for_review.empty:
        mismatches_for_review["length_diff"] = abs(
            mismatches_for_review["committee_name_inferred"].str.len()
            - mismatches_for_review["committee_name_expected"].str.len()
        )
        mismatches_for_review = mismatches_for_review.sort_values("length_diff")

    # Select columns for output
    base_columns = [
        "newsletter_id",
        "model",
        "prompt_type",
        "committee_name_inferred",
        "committee_name_expected",
        "similarity_score"
    ]

    # Prepare output DataFrames with appropriate columns
    if not mismatches_for_review.empty:
        review_columns = base_columns.copy()
        if "length_diff" in mismatches_for_review.columns:
            review_columns.append("length_diff")
        mismatches_output = mismatches_for_review[review_columns]
    else:
        mismatches_output = mismatches_for_review

    if not close_matches_auto.empty:
        close_matches_output = close_matches_auto[base_columns]
    else:
        close_matches_output = close_matches_auto

    return mismatches_output, close_matches_output


def interactive_mismatch_review(mismatches_df: pd.DataFrame) -> Dict:
    """Lightweight CLI for reviewing mismatches"""

    print(f"\n🔍 Manual Review of {len(mismatches_df)} Mismatches")
    print("=" * 60)
    print("For each mismatch, classify as:")
    print("  ✅ 'c' = Correct (minor formatting difference)")
    print("  ❌ 'i' = Incorrect")
    print("  ⏭️  's' = Skip this one")
    print("  🛑 'q' = Quit review")
    print("=" * 60)

    results = {
        "correct_with_formatting": [],
        "incorrect": [],
        "skipped": [],
        "reviewed_count": 0,
    }

    for idx, row in mismatches_df.iterrows():
        print(f"\n📧 Newsletter: {row['newsletter_id'][:8]}...")
        print(f"🤖 Model: {row['model']} | Prompt: {row['prompt_type']}")
        print(f"📝 Expected:  '{row['committee_name_expected']}'")
        print(f"🎯 Inferred:  '{row['committee_name_inferred']}'")
        print(f"📏 Length diff: {row['length_diff']}")

        while True:
            choice = input("\nClassify [c/i/s/q]: ").lower().strip()

            if choice == "c":
                results["correct_with_formatting"].append(idx)
                results["reviewed_count"] += 1
                break
            elif choice == "i":
                results["incorrect"].append(idx)
                results["reviewed_count"] += 1
                break
            elif choice == "s":
                results["skipped"].append(idx)
                break
            elif choice == "q":
                print(
                    f"\n🏁 Review stopped. Reviewed {results['reviewed_count']} items."
                )
                return results
            else:
                print("❓ Please enter 'c', 'i', 's', or 'q'")

    print(f"\n🎉 Review complete! Reviewed {results['reviewed_count']} items.")
    return results


def save_review_results(
    review_results: Dict, filename: str = "benchmarking/data/manual_review.json"
):
    """Save manual review results to JSON file"""
    with open(filename, "w") as f:
        json.dump(review_results, f, indent=2)
    print(f"💾 Review results saved to {filename}")


def print_summary_report(
    df: pd.DataFrame,
    grouped_metrics: Dict,
    review_results: Dict = None,
    close_matches_auto: pd.DataFrame = None,
    overall_with_close: Dict = None
):
    """Print a comprehensive summary report"""

    overall_metrics = calculate_accuracy_metrics(df)

    print("\n" + "=" * 80)
    print("📊 INFERENCE EVALUATION SUMMARY REPORT")
    print("=" * 80)

    # Overall metrics
    print("\n🎯 OVERALL PERFORMANCE (Exact Matches Only)")
    print(f"   Total inferences: {overall_metrics['total_inferences']:,}")
    print(
        f"   Exact matches: {overall_metrics['exact_matches']:,} ({overall_metrics['accuracy_rate']:.2%})"
    )
    print(
        f"   Parsing errors: {overall_metrics['parsing_errors']:,} ({overall_metrics['parsing_error_rate']:.2%})"
    )
    print(
        f"   Mismatches: {overall_metrics['mismatches']:,} ({overall_metrics['mismatch_rate']:.2%})"
    )

    # Close match analysis
    if overall_with_close and close_matches_auto is not None:
        print("\n🎯 ADJUSTED PERFORMANCE (Including Close Matches)")
        print(f"   Total inferences: {overall_with_close['total_inferences']:,}")
        print(f"   Exact matches: {overall_with_close['exact_matches']:,}")
        print(f"   Close matches: {overall_with_close['close_matches']:,} ({overall_with_close['close_match_rate']:.2%})")
        print(f"   Combined accuracy: {overall_with_close['accuracy_rate']:.2%}")
        print(f"   Parsing errors: {overall_with_close['parsing_errors']:,} ({overall_with_close['parsing_error_rate']:.2%})")
        print(f"   Remaining mismatches: {overall_with_close['mismatches']:,} ({overall_with_close['mismatch_rate']:.2%})")

    # Breakdown by model and prompt
    print("\n📈 BREAKDOWN BY MODEL & PROMPT TYPE")
    print("-" * 80)
    for group_name, metrics in grouped_metrics.items():
        print(
            f"{group_name:50} | Accuracy: {metrics['accuracy_rate']:6.2%} | "
            f"Parse Errors: {metrics['parsing_error_rate']:6.2%} | "
            f"Total: {metrics['total_inferences']:4,}"
        )

    # Manual review summary if available
    if review_results:
        print("\n🔍 MANUAL REVIEW RESULTS")
        total_reviewed = review_results["reviewed_count"]
        correct_formatting = len(review_results["correct_with_formatting"])
        incorrect = len(review_results["incorrect"])

        if total_reviewed > 0:
            adj_accuracy = correct_formatting / total_reviewed
            print(f"   Reviewed: {total_reviewed} remaining mismatches")
            print(
                f"   Correct with formatting issues: {correct_formatting} ({adj_accuracy:.2%})"
            )
            print(f"   Actually incorrect: {incorrect} ({1 - adj_accuracy:.2%})")

            # Calculate final adjusted overall accuracy
            estimated_correct = (correct_formatting / total_reviewed) * overall_with_close["mismatches"]
            final_total_correct = (
                overall_with_close["exact_matches"] +
                overall_with_close["close_matches"] +
                estimated_correct
            )
            final_adjusted_accuracy = (
                final_total_correct / overall_with_close["total_inferences"]
            )
            print(f"   📈 Final estimated accuracy: {final_adjusted_accuracy:.2%}")


def analyze_close_matches_only():
    """Non-interactive analysis to show the impact of close match detection"""

    # Load inference results
    print("📁 Loading inference results...")
    df = pd.read_csv("benchmarking/data/inferences.csv")

    # Calculate metrics without close match detection
    print("📊 Calculating original accuracy metrics...")
    original_metrics = calculate_accuracy_metrics(df, include_close_matches=False)

    # Get mismatches for review with close match detection
    print("🔍 Identifying mismatches and close matches...")
    mismatches_for_review, close_matches_auto = get_mismatches_for_review(df, exclude_close_matches=True)

    # Calculate adjusted metrics with close matches included
    adjusted_metrics = calculate_accuracy_metrics(df, include_close_matches=True)

    print("\n" + "=" * 80)
    print("📊 CLOSE MATCH DETECTION ANALYSIS")
    print("=" * 80)

    print(f"\n🎯 IMPACT OF CLOSE MATCH DETECTION:")
    print(f"   Original accuracy (exact matches only): {original_metrics['accuracy_rate']:.2%}")
    print(f"   Adjusted accuracy (with close matches): {adjusted_metrics['accuracy_rate']:.2%}")
    print(f"   Improvement: {adjusted_metrics['accuracy_rate'] - original_metrics['accuracy_rate']:.2%}")

    print(f"\n📈 BREAKDOWN:")
    print(f"   Total inferences: {original_metrics['total_inferences']:,}")
    print(f"   Exact matches: {original_metrics['exact_matches']:,}")
    print(f"   Close matches (auto-accepted): {adjusted_metrics['close_matches']:,}")
    print(f"   Remaining for manual review: {len(mismatches_for_review):,}")
    print(f"   Parsing errors: {original_metrics['parsing_errors']:,}")

    reduction_pct = (1 - len(mismatches_for_review) / original_metrics['mismatches']) * 100
    print(f"\n🎉 MANUAL REVIEW REDUCTION:")
    print(f"   Original mismatches needing review: {original_metrics['mismatches']:,}")
    print(f"   Remaining mismatches needing review: {len(mismatches_for_review):,}")
    print(f"   Reduction: {reduction_pct:.1f}%")

    if len(close_matches_auto) > 0:
        print(f"\n📋 Sample close matches (auto-accepted):")
        print("-" * 80)
        for _, row in close_matches_auto.head(10).iterrows():
            print(f"Expected:  '{row['committee_name_expected']}'")
            print(f"Inferred:  '{row['committee_name_inferred']}'")
            print(f"Similarity: {row['similarity_score']:.3f}")
            print("-" * 40)

    return mismatches_for_review, close_matches_auto


def main():
    """Main evaluation workflow"""

    # Load inference results
    print("📁 Loading inference results...")
    df = pd.read_csv("benchmarking/data/inferences.csv")

    # Calculate metrics without close match detection first
    print("📊 Calculating accuracy metrics...")
    grouped_metrics = analyze_by_groups(df)

    # Get mismatches for review with close match detection
    print("🔍 Identifying mismatches and close matches...")
    mismatches_for_review, close_matches_auto = get_mismatches_for_review(df, exclude_close_matches=True)

    print(f"✅ Found {len(close_matches_auto)} close matches (auto-accepted)")
    print(f"🔍 Found {len(mismatches_for_review)} remaining mismatches for manual review")

    if len(close_matches_auto) > 0:
        print("\n📋 Sample close matches (auto-accepted):")
        print("-" * 80)
        for _, row in close_matches_auto.head(5).iterrows():
            print(f"Expected:  '{row['committee_name_expected']}'")
            print(f"Inferred:  '{row['committee_name_inferred']}'")
            print(f"Similarity: {row['similarity_score']:.3f}")
            print("-" * 40)

    # Ask if user wants to do manual review of remaining mismatches
    if len(mismatches_for_review) > 0:
        do_review = (
            input(f"\nDo manual review of {len(mismatches_for_review)} remaining mismatches? [y/N]: ")
            .lower()
            .strip()
        )

        if do_review == "y":
            review_results = interactive_mismatch_review(mismatches_for_review)
            save_review_results(review_results)
        else:
            review_results = None
            print("⏭️  Skipping manual review")
    else:
        review_results = None
        print("✅ No remaining mismatches found!")

    # Calculate adjusted metrics with close matches included
    print("📊 Calculating adjusted accuracy metrics (including close matches)...")
    overall_with_close = calculate_accuracy_metrics(df, include_close_matches=True)

    # Print final report
    print_summary_report(df, grouped_metrics, review_results, close_matches_auto, overall_with_close)


if __name__ == "__main__":
    main()
