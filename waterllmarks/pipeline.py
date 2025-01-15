"""Implement LLM and text-watermarking pipelines."""

import logging
from dataclasses import dataclass
from textwrap import dedent

from langchain.llms.base import LLM as BaseLLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class ExecError(Exception):
    """Raise when an error occurs during pipeline execution."""

    pass


@dataclass
class ExecConfig:
    """A class to represent the execution configuration of a pipeline."""

    llm: BaseLLM
    vectordb: VectorStore


class Pipeline:
    """A class to represent a pipeline of steps.

    Parameters
    ----------
    steps : list[Step | Pipeline], optional
        A list of pipeline steps, by default []. The steps can be either instances of
        the Step class or of the Pipeline class, but Steps are preferred. If a Pipeline
        is given, its steps will be appended to the current pipeline.
    """

    def __init__(
        self, steps: list["Step" | "Pipeline"] = [], config: ExecConfig = None
    ):
        self.steps = steps

        if config is not None:
            self.config = config

    def __or__(self, value) -> "Pipeline":
        """Combine pipeline steps using the | operator."""
        if isinstance(value, Step):
            self.steps.append(value)
        elif isinstance(value, Pipeline):
            self.steps.extend(value.steps)
        else:
            raise ValueError("Invalid value for pipeline step.")
        return self

    def __call__(self, input: dict) -> dict:
        """Run the pipeline on the given input."""
        for step in self.steps:
            logger.info(f"Running step {step.__class__.__name__}")
            input = step(input)
        return input


def requires(fields: list[str]):
    """Ensure the input dictionary contains the required fields."""

    def decorator(func):
        def wrapper(self: Step, input: dict, *args, **kwargs):
            for field in fields:
                if field not in input:
                    raise ValueError(
                        f"Field '{field}' is required but not found in input."
                    )
            return func(self, input, *args, **kwargs)

        return wrapper

    return decorator


def configured(fields: list[str] = []):
    """Ensure the step is configured with an execution configuration."""

    def decorator(func):
        def wrapper(self: Step, input: dict, *args, **kwargs):
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
            return func(self, input, *args, **kwargs)

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
    fields using the `@requires` decorator.
    """

    def __init__(self, config: ExecConfig = None):
        self.config = config

    def __or__(self, value) -> Pipeline:
        """Combine step with another step or pipeline using the | operator."""
        if isinstance(value, Step):
            return Pipeline([self, value])
        elif isinstance(value, Pipeline):
            return Pipeline([self] + value.steps)
        raise ValueError("Invalid value for pipeline step.")

    def __call__(self, *args, **kwargs) -> dict:
        """Run the step on the given input."""
        raise NotImplementedError("The parent class Step cannot be called directly.")

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
        
        Answer:"""),
    )

    def __init__(self, *args, prompt: PromptTemplate = None, **kwargs):
        super().__init__(*args, **kwargs)

        if prompt is None:
            prompt = self.default_prompt

        self.prompt = prompt

    @requires(["query"])
    @configured(["llm"])
    def __call__(self, input: dict) -> dict:
        """Generate text using the LLM."""
        chain = self.prompt | self.config.llm
        input["answer"] = chain.invoke(input["query"])

        return input


class RAGLLM(LLM):
    """A Step to retrieve context using a RAG model.

    Consumes: "query", "context"
    Produces: "answer"
    """

    default_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=dedent("""Your are an helpful assistant.
                        
        Context: {context}
         
        Question: {query}
        
        Answer"""),
    )

    @requires(["query", "context"])
    def __call__(self, input):
        """Generate text using the LLM."""
        chain = self.prompt | self.config.llm
        input["answer"] = chain.invoke(input)
