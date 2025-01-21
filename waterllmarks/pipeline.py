"""Implement LLM and text-watermarking pipelines."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from textwrap import dedent
from typing import Iterator, TypedDict

from langchain.llms.base import LLM as BaseLLM
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings as BaseEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    BleuScore,
    DistanceMeasure,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    Metric,
    NoiseSensitivity,
    NonLLMStringSimilarity,
    ResponseRelevancy,
    RougeScore,
    SemanticSimilarity,
)
from tqdm import tqdm

from .watermaks import TextWatermark

logger = logging.getLogger(__name__)


class Artifact(TypedDict, total=False):
    """A dictionary to represent the artifacts passed between pipeline steps."""

    query: str
    context_texts: list[str]
    context_ids: list[str]
    answer: str
    reference: str


class ExecError(Exception):
    """Raise when an error occurs during pipeline execution."""


class ConfigError(Exception):
    """Raise when an error occurs when configuring pipelines or steps."""


@dataclass
class ExecConfig:
    """A class to represent the execution configuration of a pipeline."""

    llm: BaseLLM = None
    vectordb: BaseRetriever = None
    watermark: TextWatermark = None
    embedding: BaseEmbeddings = None

    log: bool = True


class Pipeline:
    """A class to represent a pipeline of steps.

    Data is passed from step to step as "artifacts", Python dictionaries for which the
    keys are standardized. See the documentation of the Step class to learn more.

    Parameters
    ----------
    steps : list[Step | Pipeline], optional
        A list of pipeline steps, by default []. The steps can be either instances of
        the Step class or of the Pipeline class, but Steps are preferred. If a Pipeline
        is given, its steps will be appended to the current pipeline.
    config : ExecConfig, optional
        The execution configuration for the pipeline, by default None. If provided,
        it will be used to configure each step in the pipeline.
    """

    def __init__(
        self,
        steps: list["Step | Pipeline"] = [],
        config: ExecConfig = None,
        name: str = None,
    ):
        self.steps = steps
        self.history = []

        if config is not None:
            self.config = config

        if name is None:
            name = self.__class__.__name__
        self.name = name

    def __call__(self, artifacts: Artifact) -> Artifact:
        """Run the pipeline on the given input."""
        logger.info(f"Starting {self.name}.")
        self.history.append(artifacts)

        try:
            for step in self.steps:
                logger.info(f"Running step {step.name}")
                artifacts = step(artifacts)
                self.history.append(artifacts)
        except Exception as e:
            raise e
            artifacts = {"error": str(e)}
        return artifacts

    def __or__(self, value) -> "Pipeline":
        """Combine pipeline steps using the | operator."""
        if isinstance(value, Step):
            self.steps.append(value)
        elif isinstance(value, Pipeline):
            self.steps.extend(value.steps)
        else:
            raise ValueError("Invalid value for pipeline step.")
        return self

    def configure(self, config: ExecConfig = None, recurse: bool = True):
        """Configure the pipeline with the given execution configuration."""
        if config is not None:
            self.config = config

        if self.config is None:
            # No configuration was provided, and none was found in the class.
            raise ConfigError("No configuration available for pipeline.")

        if recurse:
            for step in self.steps:
                step.configure(self.config)

    def apply(
        self,
        dataset: list[Artifact],
        max_workers: int = None,
        show_progress: bool = True,
    ) -> list[Artifact]:
        """Process a list of dictionaries through the pipeline.

        Parameters
        ----------
        dataset : list[Artifact]
            A list of dictionaries to process.
        max_workers : int, optional
            The number of workers to use for parallel processing, by default None.
        show_progress : bool, optional
            Whether to show a progress bar, by default True.
        """
        if not max_workers:  # Sequential processing
            iterator = tqdm(dataset) if show_progress else dataset
            return [self(item) for item in iterator]

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = []
            # for i in range(0, len(dataset), batch_size):
            #     batch = dataset[i : i + batch_size]
            #     futures = [executor.submit(self, item) for item in batch]
            #     batch_results = [f.result() for f in futures]
            #     results.extend(batch_results)
            # return results

            futures = [executor.submit(self, item) for item in dataset]
            iterator = tqdm(futures) if show_progress else futures
            return [f.result() for f in iterator]

    def apply_iter(
        self, dataset: list[Artifact] | Iterator[Artifact], batch_size: int = None
    ) -> Iterator[Artifact]:
        """Process a dataset through the pipeline, yielding results as they complete.

        Useful for large datasets that shouldn't be held in memory all at once.

        Parameters
        ----------
        dataset : list[Artifact] | Iterator[Artifact]
            A list or iterator of dictionaries to process.
        batch_size : int, optional
            The number of items to process in each batch, by default None.
        """
        if batch_size:
            batch = []
            for item in dataset:
                batch.append(item)
                if len(batch) == batch_size:
                    for processed in self.apply(batch):
                        yield processed
                    batch = []
            if batch:  # Process remaining items
                for processed in self.apply(batch):
                    yield processed
        else:
            for item in dataset:
                yield self(item)


def requires(fields: list[str]):
    """Ensure the input dictionary contains the required fields."""

    def decorator(call_func):
        def wrapper(self: Step, artifacts: Artifact) -> Artifact:
            for field in fields:
                if field not in artifacts:
                    raise ValueError(
                        f"Field '{field}' is required but not found in input."
                    )
            return call_func(self, artifacts)

        return wrapper

    return decorator


def configured(fields: list[str] = []):
    """Ensure the step is configured with an execution configuration."""

    def decorator(call_func):
        def wrapper(self: Step, artifacts: Artifact) -> Artifact:
            if self.config is None:
                raise ValueError(
                    f"Step '{self.__class__.__name__}' cannot run without configuration."
                )
            for field in fields:
                if getattr(self.config, field, None) is None:
                    raise ValueError(
                        f"Configuration field '{field}'"
                        " is required to execute step '{self.__class__.__name__}'."
                    )
            return call_func(self, artifacts)

        return wrapper

    return decorator


class Step:
    """A class to represent a pipeline step.

    Steps are the building blocks of a pipeline. They consume and produce dictionaries
    according to the following schema:
    ```python
    {
        "query": str,
        "context_texts": list[str],
        "context_ids": list[str],
        "answer": str,
        "reference": str,
    }
    ```

    Although all the fields are technically optional, each Step can require specific
    fields using the `@requires` decorator on the __call__ magic method.
    """

    def __init__(self, config: ExecConfig = None, name: str = None):
        self.config = config
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def __or__(self, value) -> Pipeline:
        """Combine step with another step or pipeline using the | operator."""
        if isinstance(value, Step):
            return Pipeline([self, value])

    def configure(self, config: ExecConfig):
        """Configure the step with the given execution configuration."""
        self.config = config


class LLM(Step):
    """A Step to generate text using a LLM.

    Consumes: "query"
    Produces: "answer"
    """

    default_prompt = PromptTemplate(
        input_variables=["query"],
        template=dedent("""You are a helpful assistant.
        
        Question: {query}
        
        Answer: """),
    )

    def __init__(self, *args, prompt: PromptTemplate = None, **kwargs):
        super().__init__(*args, **kwargs)

        if prompt is None:
            prompt = self.default_prompt

        self.prompt = prompt

    @requires(["query"])
    @configured(["llm"])
    def __call__(self, artifacts: Artifact) -> Artifact:
        """Generate text using the LLM."""
        chain = self.prompt | self.config.llm
        reply: AIMessage = chain.invoke(artifacts)
        artifacts["answer"] = reply.content

        return artifacts


class RAGLLM(LLM):
    """A Step to generate text using a LLM and a context.

    Consumes: "query", "context"
    Produces: "answer"
    """

    default_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=dedent("""Your are an helpful assistant.
                        
        Context: {context}
         
        Question: {query}
        
        Answer: """),
    )

    @requires(["query", "context"])
    def __call__(self, artifacts):
        """Generate text using the LLM."""
        return super().__call__(artifacts)


class VectorDB(Step):
    """A Step to retrieve context from a vector store.

    Consumes: "query"
    Produces: "context", "chunks"
    """

    @requires(["query"])
    @configured(["vectordb"])
    def __call__(self, artifacts):
        """Query the DB to return the k most relevant entries."""
        db = self.config.vectordb
        docs = db.invoke(artifacts["query"])
        artifacts["chunks"] = [doc.page_content for doc in docs]
        artifacts["chunk_ids"] = [doc.id for doc in docs]
        artifacts["context"] = "\n\n".join(artifacts["chunks"])
        return artifacts


class Watermark(Step):
    """A Step to watermark text with an invisible message."""

    @requires(["query"])
    @configured(["watermark"])
    def __call__(self, artifacts):
        """Watermark the text with the invisible message."""
        marker = self.config.watermark
        artifacts["query"] = marker.apply(artifacts["query"])

        return artifacts


class Evaluate(Step):
    """A Step to evaluate the quality of the generated text."""

    @requires(["query", "answer", "reference", "context"])
    @configured(["llm", "embedding"])
    def __call__(self, artifacts):
        """Evaluate the quality of the generated text."""
        data = SingleTurnSample(
            user_input=artifacts["query"],
            retrieved_contexts=artifacts["chunks"],
            response=artifacts["answer"],
            reference=artifacts["reference"],
        )
        llm = LangchainLLMWrapper(self.config.llm)
        embeddings = LangchainEmbeddingsWrapper(self.config.embedding)

        async def compute_metrics():
            metrics = {
                # Non-LLM metrics
                "blue": BleuScore(),
                "rouge": RougeScore(),
                "levenstein_sim": NonLLMStringSimilarity(),
                # LLM-based metrics
                "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
                "context_recall": LLMContextRecall(llm=llm),
                "noise_sensitivity": NoiseSensitivity(llm=llm),
                "response_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
                "faithfulness": Faithfulness(llm=llm),
                "factual_correctness": FactualCorrectness(llm=llm),
                "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
            }

            async with asyncio.TaskGroup() as tg:
                tasks = {
                    name: tg.create_task(metric.single_turn_ascore(data))
                    for name, metric in metrics.items()
                }

            results = {}
            for name, task in tasks.items():
                results[name] = task.result()

            # Calculate F1 score after gathering results
            results["context_f1"] = (
                2
                * results["context_precision"]
                * results["context_recall"]
                / (results["context_precision"] + results["context_recall"])
            )

            return results

        artifacts["metrics"] = asyncio.run(compute_metrics())
        return artifacts


class QueryAugmentation(Step):
    """A Step to augment the query size to fit a watermark.

    Parameters
    ----------
    size : int, optional
        The aimed size for the query.
    """

    def __init__(self, *args, size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    @requires(["query"])
    @configured(["llm"])
    def __call__(self, artifacts):
        """Augment the query size."""
        default_prompt = PromptTemplate(
            input_variables=["query"],
            template=dedent(f"""
            You are a helpful assistant, specialized in query augmentation. 
            Your goal is to make the query longer, while keeping the same meaning.
            
            Minimal query size: {self.size}

            Query: {artifacts["query"]}
            """),
        )

        chain = default_prompt | self.config.llm
        reply: AIMessage = chain.invoke(artifacts)
        artifacts["query"] = reply.content

        return artifacts
