from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple
import itertools
import statistics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def generate_samples(model: Callable[..., str], prompt: str, k: int, *, temperature: float = 0.9, **kwargs) -> List[str]:
    """
    Generate K different responses to the same prompt using stochastic sampling.

    Parameters
    ----------
    model:
        A callable that produces text: model(prompt, temperature=..., **kwargs) -> str
        In practice this could wrap an LLM API or a local model.
    prompt:
        The user prompt.
    k:
        Number of samples.
    temperature:
        Sampling temperature (higher => more diverse generations).
    kwargs:
        Any extra arguments passed to the model.

    Returns
    -------
    List[str]
        K generated responses.
    """
    responses: List[str] = []
    for _ in range(k):
        # The model is expected to use `temperature` to add randomness.
        responses.append(model(prompt, temperature=temperature, **kwargs))
    return responses


@dataclass
class SemanticConsistencyResult:
    avg_pairwise_similarity: float
    min_pairwise_similarity: float
    num_clusters: int
    cluster_labels: List[int]
    semantic_entropy_score: float  # higher => less consistent (more uncertain)


def calculate_semantic_consistency(responses: List[str], *, distance_threshold: float = 0.55) -> SemanticConsistencyResult:
   
    if len(responses) < 2:
        raise ValueError("Need at least 2 responses to measure consistency.")

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(responses)

    sim = cosine_similarity(X)
    # collect pairwise similarities (upper triangle)
    pair_sims = []
    for i, j in itertools.combinations(range(len(responses)), 2):
        pair_sims.append(float(sim[i, j]))

    avg_sim = statistics.fmean(pair_sims)
    min_sim = min(pair_sims)

    # Cosine distance matrix for clustering: dist = 1 - sim
    dist = 1.0 - sim

    # Agglomerative clustering with a distance threshold.
    # If the threshold is low, it will split more aggressively.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(dist).tolist()
    num_clusters = len(set(labels))

    # A simple "semantic entropy-like" score:
    # - higher when similarity is low and clusters are many
    # This is NOT a true Shannon entropy; it's a practical alert metric.
    semantic_entropy_score = (1.0 - avg_sim) * (num_clusters / len(responses))

    return SemanticConsistencyResult(
        avg_pairwise_similarity=avg_sim,
        min_pairwise_similarity=min_sim,
        num_clusters=num_clusters,
        cluster_labels=labels,
        semantic_entropy_score=semantic_entropy_score,
    )


# ---- Example usage (optional) ----
if __name__ == "__main__":
    # Dummy stochastic model for demonstration only.
    import random

    def toy_model(prompt: str, temperature: float = 0.9, **kwargs) -> str:
        variants = [
            "We don't have enough evidence; propose safe questions and observe their responses.",
            "Their dreams might be social simulations; we should avoid assuming human sleep cycles.",
            "Dreaming could be a metaphor for predictive planning; treat this as uncertain.",
            "They may not dream at all; some species remain conscious while resting.",
            "Any claim is speculation; gather data before interpreting internal experiences.",
        ]
        # temperature isn't used here, but a real model would.
        return random.choice(variants) + f" (prompt: {prompt[:40]}...)"

    samples = generate_samples(toy_model, "What do the Keplerians dream about?", 5)
    result = calculate_semantic_consistency(samples)
    print("Samples:")
    for s in samples:
        print("-", s)
    print("\nResult:", result)
