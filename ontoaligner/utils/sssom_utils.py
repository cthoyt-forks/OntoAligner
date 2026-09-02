"""Utilities for working with SSSOM, to be shared across SSSOM-Py and SSSOM-Pydantic."""

from __future__ import annotations

from typing import TypeAlias, Dict, NamedTuple, TYPE_CHECKING, Callable, Any

import curies
from curies import vocabulary as v

if TYPE_CHECKING:
    from ontoaligner.base.model import BaseOMModel

__all__ = [
    "FUZZY_ALIGNERS",
    "GRAPH_ALIGNERS",
    "get_justification",
    "get_score",
    "SimilarityScoreTuple",
    "ConfidenceTuple",
    "Matching",
    "Postprocessor",
]

Matching: TypeAlias = Dict
Postprocessor: TypeAlias = Callable

#: A list of fuzzy aligners from the
#: :mod:`ontoaligner.aligner.lightweight.models` module
FUZZY_ALIGNERS = {
    "SimpleFuzzySMLightweight": "RapidFuzz fuzz.ratio",
    "WeightedFuzzySMLightweight": "RapidFuzz fuzz.WRatio",
    "TokenSetFuzzySMLightweight": "RapidFuzz fuzz.token_set_ratio",
}

#: A list of aligner class names from the
#: :mod:`ontoaligner.aligner.graph.models` module
GRAPH_ALIGNERS = {
    "ConvEAligner",
    "TransDAligner",
    "TransEAligner",
    "TransFAligner",
    "TransHAligner",
    "TransRAligner",
    "DistMultAligner",
    "ComplExAligner",
    "HolEAligner",
    "RotatEAligner",
    "SimplEAligner",
    "CrossEAligner",
    "BoxEAligner",
    "CompGCNAligner",
    "MuREAligner",
    "QuatEAligner",
    "SEAligner",
}


def get_justification(
    *,
    postprocessor: str | Postprocessor | None,
    postprocessor_params: dict[str, Any] | None = None,
    aligner: BaseOMModel,
) -> curies.NamedReference:
    """Get the justification."""
    aligner_name = aligner.__class__.__name__
    postprocessor_name = _get_postprocessor_name(postprocessor)

    has_retrieval_threshold = (
        postprocessor_name == "retriever_postprocessor"
        and postprocessor_params is not None
        and "threshold" in postprocessor_params
    )

    if postprocessor_name in {
        "rag_heuristic_postprocessor",
        "rag_hybrid_postprocessor",
    }:
        return v.composite_matching_process

    elif aligner_name in FUZZY_ALIGNERS:
        return v.lexical_similarity_threshold_based_matching_process

    elif (
        aligner_name in {"TFIDFRetrieval", "BM25Retrieval"} and has_retrieval_threshold
    ):
        return v.lexical_similarity_threshold_based_matching_process

    elif aligner_name in {"SBERTRetrieval", "AdaRetrieval"} and has_retrieval_threshold:
        return v.semantic_similarity

    elif aligner_name in GRAPH_ALIGNERS:
        return v.structural_matching

    elif aligner_name == "PropMatchAligner":
        disable_domain_range = getattr(
            aligner,
            "disable_domain_range",
            getattr(aligner, "kwargs", {}).get("disable_domain_range", False),
        )
        return (
            v.lexical_similarity_threshold_based_matching_process
            if disable_domain_range
            else v.composite_matching_process
        )

    elif aligner_name == "OLaLaHighPrecisionMatcher":
        return v.lexical_matching_process

    elif aligner_name in {"FLORAAligner", "EnsembleLearningAligner"}:
        return v.composite_matching_process

    return v.unspecified_matching_process


class SimilarityScoreTuple(NamedTuple):
    """A named tuple containing a similarity score and measure label."""

    score: float
    measure: str


class ConfidenceTuple(NamedTuple):
    """A named tuple containing a confidence value on [0.0, 1.0]."""

    confidence: float


def get_score(
    matching: Matching, aligner: BaseOMModel
) -> SimilarityScoreTuple | ConfidenceTuple | None:
    score = matching.get("score")
    if score is None:
        return None

    aligner_name = aligner.__class__.__name__
    score = float(score)

    if aligner_name in FUZZY_ALIGNERS:
        return SimilarityScoreTuple(score, FUZZY_ALIGNERS[aligner_name])

    elif aligner_name == "TFIDFRetrieval":
        return SimilarityScoreTuple(score, "TF-IDF cosine similarity")

    elif aligner_name == "SBERTRetrieval" and 0 <= score <= 1:
        return SimilarityScoreTuple(
            score, "cosine similarity over SentenceTransformer embeddings"
        )

    elif aligner_name == "AdaRetrieval" and 0 <= score <= 1:
        return SimilarityScoreTuple(score, "cosine similarity over OpenAI embeddings")

    elif aligner_name in GRAPH_ALIGNERS and 0 <= score <= 1:
        return SimilarityScoreTuple(score, "cosine similarity over graph embeddings")

    elif (
        aligner_name
        in {
            "OLaLaHighPrecisionMatcher",
            "OLaLaLLMAligner",
            "OLaLaAligner",
        }
        and 0 <= score <= 1
    ):
        return ConfidenceTuple(score)

    # FIXME implement remaining or raise an exception
    return None


def _get_postprocessor_name(postprocessor: str | Postprocessor | None) -> str | None:
    if postprocessor is None:
        return None
    elif isinstance(postprocessor, str):
        return postprocessor
    elif callable(postprocessor):
        return postprocessor.__name__
    else:
        raise TypeError(f"invalid postprocessor: {postprocessor}")
