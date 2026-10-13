"""
Summarization quality metrics used to score each model's predictions
against reference summaries.
"""

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def calculate_redundancy(summaries: list) -> float:
    """
    Fraction of repeated tokens across a set of summaries.
    Higher = more repetitive / less diverse language. This is a "lower is
    better" criterion when fed into TOPSIS.
    """
    total_tokens = sum(len(summary.split()) for summary in summaries)
    unique_tokens = len({token for summary in summaries for token in summary.split()})
    return 1 - (unique_tokens / total_tokens)


def calc_metrics(actual_summaries: list, pred_summaries: list) -> dict:
    """
    Compute BLEU, BERTScore (F1), ROUGE-1/2/L (F1), and redundancy for one
    model's predictions against ground-truth summaries.
    """
    actual_tokenized = [[ref.split()] for ref in actual_summaries]
    pred_tokenized = [pred.split() for pred in pred_summaries]
    bleu = corpus_bleu(actual_tokenized, pred_tokenized)

    _, _, f1 = bert_score_fn(pred_summaries, actual_summaries, lang="en", verbose=False)
    bert = f1.mean().item()

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
    for pred, actual in zip(pred_summaries, actual_summaries):
        scores = rouge.score(pred, actual)
        rouge_1_scores.append(scores["rouge1"].fmeasure)
        rouge_2_scores.append(scores["rouge2"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)

    redundancy = calculate_redundancy(pred_summaries)

    return {
        "bleu": bleu,
        "bert": bert,
        "rouge_1": sum(rouge_1_scores) / len(rouge_1_scores),
        "rouge_2": sum(rouge_2_scores) / len(rouge_2_scores),
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores),
        "redundancy": redundancy,
    }
