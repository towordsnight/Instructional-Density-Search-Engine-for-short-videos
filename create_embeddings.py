#!/usr/bin/env python3
"""
Create embeddings for Short Compilation search engine.
Loads all-MiniLM-L6-v2 and generates embeddings for titles and transcripts
from YouTube, TikTok, Instagram shorts.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from instructional_score import compute_instructional_scores


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load pre-trained sentence transformer model."""
    return SentenceTransformer(model_name)


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of text strings (title + transcript combined)
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        numpy array of shape (n_samples, embedding_dim)
    """
    # Handle empty strings and NaN
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def load_dataset(input_path: Path, title_col: str = "title", transcript_col: str = "transcript"):
    """Load dataset from JSON or CSV file. Returns (df, title_col, transcript_col)."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        df = pd.read_json(input_path)
    elif suffix in (".csv", ".tsv"):
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(input_path, sep=sep)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .json or .csv")

    # Map lowercase to actual column names
    col_map = {c.lower(): c for c in df.columns}
    title_col = col_map.get(title_col.lower()) or col_map.get("title") or df.columns[0]
    transcript_col = col_map.get(transcript_col.lower()) or col_map.get("transcript") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    return df, title_col, transcript_col


def main():
    parser = argparse.ArgumentParser(description="Create embeddings for short video titles and transcripts")
    parser.add_argument("--input", "-i", type=str, default="shorts_data.json", help="Input file path (JSON or CSV)")
    parser.add_argument("--output", "-o", type=str, default="embeddings.npy", help="Output embeddings file (.npy)")
    parser.add_argument("--density-output", type=str, default="density_scores.npy", help="Output instructional density scores (.npy)")
    parser.add_argument("--metadata-output", type=str, default="metadata.json", help="Output metadata for mapping embeddings to videos")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--title-col", type=str, default="title", help="Column name for title")
    parser.add_argument("--transcript-col", type=str, default="transcript", help="Column name for transcript")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--combine", type=str, default="concat", choices=["concat", "title_only", "transcript_only"],
                        help="How to combine title and transcript")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Loading dataset: {args.input}")
    df, title_col, transcript_col = load_dataset(args.input, args.title_col, args.transcript_col)

    # Build text for embedding (use resolved column names from load_dataset)
    if args.combine == "concat":
        df["_text"] = df[title_col].fillna("").astype(str) + " [SEP] " + df[transcript_col].fillna("").astype(str)
    elif args.combine == "title_only":
        df["_text"] = df[title_col].fillna("").astype(str)
    else:
        df["_text"] = df[transcript_col].fillna("").astype(str)

    texts = df["_text"].tolist()
    print(f"Creating embeddings for {len(texts)} items...")

    embeddings = create_embeddings(model, texts, batch_size=args.batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    density_scores = np.array(compute_instructional_scores(texts))
    print(f"Density scores: min={density_scores.min():.3f}, max={density_scores.max():.3f}")

    np.save(args.output, embeddings)
    print(f"Saved embeddings to {args.output}")

    np.save(args.density_output, density_scores)
    print(f"Saved density scores to {args.density_output}")

    meta_cols = [c for c in df.columns if c != "_text"]
    metadata = df[meta_cols].to_dict(orient="records")
    with open(args.metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
