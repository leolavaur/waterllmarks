"""RAGAS-compatible metrics for WaterLLMarks."""

import pickle
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import nltk
import numpy as np
import pandas as pd
import ragas.evaluation
import ragas.metrics
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from nltk import word_tokenize
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings, HuggingfaceEmbeddings
from ragas.evaluation import EvaluationResult
from ragas.llms.base import BaseRagasLLM
from ragas.metrics.base import (
    Metric,
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.run_config import RunConfig


@dataclass
class MetaMeteor(SingleTurnMetric):
    """METEOR score metric.

    TODO: Rewrite the docstring with more appropriate information.

    This metric computes a Meta-METEOR score between a reference and the predition
    provided by a model. METEOR is a metric that is based on the harmonic mean of the
    precision and recall of the n-grams between the reference and the prediction.
    Therefore, it is most relevant on a sentence, where it can capture the quality of
    the generated text, regardless of the order of the words. However, its relevance
    decreases with the length of the text.

    Meta-METEOR deals with the issue of multiple references by computing the METEOR
    score of each sentence of the prediction for all the sentences of the references.
    Based on the assumption that the higher score corresponds to the most related
    sentence, the final score for a prediction is the average of the best METEOR score
    for each of its sentences, with respect to the references' sentences.

    Parameters
    ----------
    alpha : float, optional
        Relative weights of precision and recall. Defaults to 0.9.
    beta : float, optional
        Shape of the penalty as a function of fragmentation. Defaults to 0.5.
    gamma : float, optional
        Relative weight assigned to the fragmentation penalty. Defaults to 0.5.
    """

    name: str = "meta_meteor_score"
    _required_columns: dict[Metric, set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    language: str = "english"

    kwargs: dict[str, Any] = field(default_factory=dict)

    def init(self, run_config: RunConfig):  # noqa: D102 -> docstring is inherited
        pass

    # async def _single_turn_ascore(
    #     self, sample: SingleTurnSample, callbacks: Callbacks
    # ) -> float:  # noqa: D102 / docstring is inherited
    #     reference, response = sample.reference, sample.response
    #     assert isinstance(reference, str), "METEOR expects a valid reference string"
    #     assert isinstance(response, str), "METEOR expects a valid response string"

    #     return single_meteor_score(
    #         word_tokenize(reference),
    #         word_tokenize(response),
    #         **self.kwargs,
    #     )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: dict, callbacks: Callbacks) -> float:
        reference, response = row["reference"], row["response"]

        assert isinstance(reference, str), "METEOR expects a valid reference string"
        assert isinstance(response, str), "METEOR expects a valid response string"

        return single_meteor_score(
            word_tokenize(reference),
            word_tokenize(response),
            **self.kwargs,
        )


@dataclass
class RetrievedContextSimilarity(ragas.metrics.SemanticSimilarity):
    """Scores the distance between the retrieved context and the provided query.

    The code is based on the `SemanticSimilarity` metric from RAGAS, and the `_ascore`
    method has been modified to calculate the similarity between the query and the
    different retrieved contexts.

    Attributes
    ----------
    model_name : str
        The model to be used for calculating semantic similarity. Defaults to
        "open-ai-embeddings". Select a cross-encoder model for best results.
        https://huggingface.co/spaces/mteb/leaderboard

    See Also
    --------
    ragas.metrics.SemanticSimilarity
    """

    name: str = "retrieved_context_similarity"
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    is_cross_encoder: bool = False
    threshold: Optional[float] = None

    async def _ascore(self, row: dict, callbacks: Callbacks) -> float:
        assert self.embeddings is not None, "embeddings must be set"

        if self.is_cross_encoder and isinstance(self.embeddings, HuggingfaceEmbeddings):
            raise NotImplementedError(
                "async score [ascore()] not implemented for HuggingFace embeddings"
            )

        # Handle embeddings for empty strings using a space
        query = cast(str, row["user_input"]) or " "
        contexts = row["retrieved_contexts"]

        scores = []
        for context in contexts:
            context = cast(str, context) or " "

            embedding_1 = np.array(await self.embeddings.embed_text(query))
            embedding_2 = np.array(await self.embeddings.embed_text(context))

            # Normalization factors of the above embeddings
            norms_1 = np.linalg.norm(embedding_1, keepdims=True)
            norms_2 = np.linalg.norm(embedding_2, keepdims=True)
            embedding_1_normalized = embedding_1 / norms_1
            embedding_2_normalized = embedding_2 / norms_2
            similarity = embedding_1_normalized @ embedding_2_normalized.T
            score = similarity.flatten()

            assert isinstance(score, np.ndarray), "Expects ndarray"
            if self.threshold:
                raise NotImplementedError("This metric cannot map to binary.")

            scores.append(score.tolist()[0])

        return np.mean(scores)

    def init(self, run_config: RunConfig):  # noqa: D102 -> docstring is inherited
        pass


@dataclass
class ContextOverlap(SingleTurnMetric):
    """Computes the Jaccard similarity between the retrieved and reference contexts."""

    name: str = "context_overlap"
    _required_columns: dict[Metric, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )

    async def _ascore(self, row: dict, callbacks: Callbacks) -> float:
        retrieved_contexts = set(row["retrieved_contexts"])
        reference_contexts = set(row["reference_contexts"])

        intersection = retrieved_contexts.intersection(reference_contexts)
        union = retrieved_contexts.union(reference_contexts)

        return len(intersection) / len(union)

    def init(self, run_config: RunConfig):  # noqa: D102 -> docstring is inherited
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)


class PipelineResult(TypedDict):
    """The result of a pipeline execution.

    Attributes
    ----------
    id : str
        The ID of the sample.
    user_input : str
        Reference user input, e.g. a question.
    reference : str
        Reference answer.
    reference_context_ids : list[str]
        IDs of the reference contexts.
    reference_contexts : list[str]
        Contents of the reference contexts.
    retrieved_contexts : list[langchain_core.documents.Document]
        Retrieved contexts as `Document` objects.
    context : str
        The generated context from the retrieved documents.
    pipeline_input : str
        The `user_input`, after being processed by the pipeline.
    """

    id: str
    user_input: str
    reference: str
    reference_context_ids: list[str]
    reference_contexts: list[str]
    retrieved_contexts: list[Document]
    context: str
    pipeline_input: str


class WLLMKResult:
    """The result of evaluating a or multiple pipelines.

    Attributes
    ----------
    metric_names : list[str]
        The names of the metrics used for evaluation.
    synthesis : dict[str, float] | pd.DataFrame
        The synthesis of the evaluation results. If the results represent a single
        evalutation, this will be a dictionary with the metric names as keys and the
        corresponding scores as values. If the results represent multiple evaluations,
        this will be a pandas DataFrame with the metric names as columns and the
        different pipeline results as rows.
    details : pd.DataFrame | dict[str, pd.DataFrame]
        The detailed evaluation results. If the results represent a single evalutation,
        this will be a pandas DataFrame with the different metrics as columns and the
        pipeline results as rows. If the results represent multiple evaluations, this
        will be a dictionary with the metric names as keys and the corresponding
        detailed results as values.
    """

    metric_names: set[str]
    synthesis: dict[str, float] | pd.DataFrame
    details: dict[str, pd.DataFrame] | pd.DataFrame

    def __init__(
        self, result: Optional[EvaluationResult] = None, **results: EvaluationResult
    ):
        self.synthesis = {}
        self.details = {}

        if result is not None:
            if not isinstance(result, EvaluationResult):
                raise ValueError("The result must be a RAGAS EvaluationResult object.")
            results = {0: result}

        if len(results) == 0:
            raise ValueError("At least one result must be provided.")

        if len(results) == 1:
            _, res = results.popitem()
            self.metric_names = set(res._repr_dict.keys())
            self.synthesis = res._repr_dict
            self.details = res.to_pandas()

        else:
            self.metric_names = set()
            for name, res in results.items():
                self.synthesis[name] = res._repr_dict
                self.details[name] = res.to_pandas()
                self.metric_names |= res._repr_dict.keys()

            self.synthesis = pd.DataFrame(
                list(self.synthesis.values()), index=self.synthesis.keys()
            )

    def save(self, path: str | Path) -> None:
        """Save the evaluation results to a file using pickle.

        Parameters
        ----------
        path : str | Path
            The path to save the evaluation results to. If a string is provided, it will
            be converted to a `Path` object.
        """
        if isinstance(path, str):
            path = Path(path)

        with path.open("wb") as file:
            pickle.dump(self, file)


def evaluate(
    pipeline_results: list["PipelineResult"],
    metrics: list[Metric],
    llm: BaseRagasLLM,
    embeddings: BaseRagasEmbeddings,
    runconfig: RunConfig | None = None,
) -> WLLMKResult:
    """Evaluate a pipeline using the given metrics.

    Parameters
    ----------
    pipeline_results : list[PipelineResult]
        The pipeline_results to evaluate.
    metrics : list[Metric]
        The metrics to use for evaluation.
    llm : BaseRagasLLM
        The LLM to use for evaluation, wrapped in `LangchainLLMWrapper`.
    embeddings : BaseRagasEmbeddings
        The embeddings to use for evaluation, wrapped in `LangchainEmbeddingsWrapper`.

    Returns
    -------
    Result
        The evaluation result.
    """
    if llm is None and any(isinstance(metric, MetricWithLLM) for metric in metrics):
        raise ValueError("One or more metrics require an LLM, but none was provided.")

    if embeddings is None and any(
        isinstance(metric, MetricWithEmbeddings) for metric in metrics
    ):
        raise ValueError(
            "One or more metrics require embeddings, but none was provided."
        )

    evaluation_samples = []
    for sample in pipeline_results:
        new_sample = {
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            "response": sample["response"],
            "pipeline_input": sample["pipeline_input"],
            "reference_contexts": sample["reference_contexts"],
            "reference_context_ids": sample["reference_context_ids"],
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
        }

        for doc in sample["retrieved_contexts"]:
            new_sample["retrieved_contexts"].append(doc.page_content)
            new_sample["retrieved_context_ids"].append(doc.id)

        ragas_sample = SingleTurnSample(**new_sample)
        # ragas_sample.__dict__.update(new_sample)

        evaluation_samples.append(ragas_sample)

    ragas_dataset = EvaluationDataset(samples=evaluation_samples)

    return ragas.evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=runconfig,
    )


# Module initialization
# ---------------------

DEFAULT_NLP_METRICS: list[Metric] = [
    ragas.metrics.BleuScore(),
    ragas.metrics.RougeScore(),
    MetaMeteor(),
    ragas.metrics.NonLLMStringSimilarity(),
]

DEFAULT_LLM_METRICS: list[Metric] = [
    ragas.metrics.SemanticSimilarity(),
    ragas.metrics.FactualCorrectness(),
]

DEFAULT_RAG_METRICS: list[Metric] = [
    ragas.metrics.LLMContextPrecisionWithoutReference(),
    ragas.metrics.LLMContextRecall(),
    ragas.metrics.Faithfulness(),
    #    ragas.metrics.NoiseSensitivity(),
    ContextOverlap(),
    RetrievedContextSimilarity(),
]

DEFAULT_ALL_METRICS: list[Metric] = (
    DEFAULT_NLP_METRICS + DEFAULT_LLM_METRICS + DEFAULT_RAG_METRICS
)


def _module_setup() -> None:
    _data_dir = "./nltk_data"
    nltk.data.path = [_data_dir]
    nltk.download("punkt_tab", download_dir=_data_dir)
    nltk.download("wordnet", download_dir=_data_dir)


_module_setup()
