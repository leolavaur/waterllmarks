"""Tests for the evaluation pieline."""

import pytest
from pytest import fixture, raises

from waterllmarks.pipeline import (
    LLM,
    RAGLLM,
    ConfigError,
    ExecConfig,
    ExecError,
    Pipeline,
    Step,
    VectorDB,
    Watermark,
    configured,
    requires,
)


class MockLLM:
    """A mock LLM class for testing."""

    def invoke(self, artifacts):
        """Mock of the LLM invoke in Langchain."""
        return "mocked answer"


class MockRetriever:
    """A mock Retriever class for testing."""

    def invoke(self, artifacts):
        """Mock of the Retriever invoke in Langchain."""
        return [{"page_content": "mocked content", "id": "1"}]


class MockWatermark:
    """A mock Watermark class for testing."""

    def apply(self, text):
        """Mock of the Watermark apply in waterLLMarks."""
        return f"[watermarked] {text}"


@fixture
def mock_config():
    """Return a mocked configuration."""
    return ExecConfig(
        llm=MockLLM(),
        vectordb=MockRetriever(),
        watermark=MockWatermark(),
    )


def test_pipeline_step_or_operator(mock_config):
    """Test the Or (|) operator for combining steps and pipelines."""
    step1 = Step()
    step2 = Step()
    pipeline1 = step1 | step2
    assert isinstance(pipeline1, Pipeline)
    assert len(pipeline1.steps) == 2

    pipeline2 = pipeline1 | Step()
    assert isinstance(pipeline2, Pipeline)
    assert len(pipeline2.steps) == 3

    pipeline3 = Step() | pipeline2
    assert isinstance(pipeline3, Pipeline)
    assert len(pipeline3.steps) == 4

    pipeline4 = Pipeline([Step(), Step()])
    pipeline5 = pipeline3 | pipeline4
    assert isinstance(pipeline5, Pipeline)
    assert len(pipeline5.steps) == 6
