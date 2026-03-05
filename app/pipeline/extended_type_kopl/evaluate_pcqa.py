"""
PcQA Evaluation Script with ROUGE and BERT Score

Evaluates the Extended Type-KoPL pipeline on PcQA dataset
using text generation metrics (matching official KGT evaluation).
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ROUGE Score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Run: pip install rouge-score")

# BERT Score
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: bert-score not installed. Run: pip install bert-score")


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    reference: str
    generated: str
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bert_f1: float = 0.0


def compute_rouge(references: List[str], generated: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores"""
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, gen in zip(references, generated):
        result = scorer.score(ref, gen)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def compute_bert_score(references: List[str], generated: List[str]) -> float:
    """Compute BERT Score (F1)"""
    if not BERT_AVAILABLE:
        return 0.0

    P, R, F1 = bert_score(generated, references, lang="en", verbose=False)
    return F1.mean().item()


def evaluate_pcqa(
    n_samples: int = 20,
    data_path: str = "/app/data/pcqa/PcQA.json",
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate Extended Type-KoPL pipeline on PcQA dataset

    Args:
        n_samples: Number of samples to evaluate
        data_path: Path to PcQA.json
        verbose: Print detailed results

    Returns:
        Tuple of (metrics dict, list of individual results)
    """
    from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:n_samples]

    # Initialize pipeline
    pipeline = ExtendedTypeKoPLPipeline(kg_type="pcqa")

    # Run evaluation
    results = []
    references = []
    generated = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        reference = sample["answer"]

        # Run pipeline
        result = pipeline.run(question)
        gen_answer = result.natural_answer

        references.append(reference)
        generated.append(gen_answer)

        eval_result = EvaluationResult(
            question=question,
            reference=reference,
            generated=gen_answer,
        )
        results.append(eval_result)

        if verbose:
            print(f"{i+1}. Q: {question[:60]}...")
            print(f"   Ref: {reference[:80]}...")
            print(f"   Gen: {gen_answer[:80]}...")
            print()

    # Compute metrics
    rouge_scores = compute_rouge(references, generated)
    bert_f1 = compute_bert_score(references, generated)

    # Update individual results with scores
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i, (ref, gen) in enumerate(zip(references, generated)):
            r = scorer.score(ref, gen)
            results[i].rouge1 = r["rouge1"].fmeasure
            results[i].rouge2 = r["rouge2"].fmeasure
            results[i].rougeL = r["rougeL"].fmeasure

    if BERT_AVAILABLE:
        P, R, F1 = bert_score(generated, references, lang="en", verbose=False)
        for i, f1 in enumerate(F1.tolist()):
            results[i].bert_f1 = f1

    metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bert_f1": bert_f1,
        "n_samples": len(samples),
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PcQA with ROUGE/BERT Score")
    parser.add_argument("--n", type=int, default=20, help="Number of samples")
    parser.add_argument("--data", type=str, default="/app/data/pcqa/PcQA.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("PcQA Evaluation (ROUGE / BERT Score)")
    print("=" * 60)
    print()

    metrics, results = evaluate_pcqa(
        n_samples=args.n,
        data_path=args.data,
        verbose=not args.quiet,
    )

    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']}")
    print()
    print("ROUGE Scores:")
    print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print()
    print(f"BERT Score (F1): {metrics['bert_f1']:.4f}")
    print()


if __name__ == "__main__":
    main()
