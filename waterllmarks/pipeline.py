"""Langchain runnables for WaterLLMarks."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from uuid import UUID

from joblib import Memory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import (
    RunnableConfig,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_core.runnables.base import Runnable, RunnableSerializable
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# Custom runnable classes
# -----------------------


class WLLMKRunnable(RunnableSerializable, BaseModel):
    """Base class for WaterLLMarks's runnables."""

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class DictParser(WLLMKRunnable):
    """A runnable that parses the input for a downstream step.

    Parameters
    ----------
    required_fields : list[str]
        The list of required fields in the input dictionary.
    """

    required_fields: str | list[str] = Field()

    def __init__(self, required_fields: str | list[str], **kwargs) -> None:
        """Overwrite Basemodel's `__init__` to allow positional arguments."""
        super().__init__(required_fields=required_fields, **kwargs)

    def invoke(
        self, input: dict[str, Any], config: RunnableConfig = None
    ) -> Any | dict[str, Any]:
        """Parse the input dictionary."""
        if isinstance(self.required_fields, str):
            return input[self.required_fields]

        ouptut = {
            field: (input[field] if field in input else None)
            for field in self.required_fields
            if field in input
        }

        return ouptut


class DictWrapper(WLLMKRunnable):
    """A runnable that wraps the input for a downstream step.

    Parameters
    ----------
    name : str
        The name of the field to wrap the input in.
    """

    name: str = Field()

    def invoke(self, input: Any, config: RunnableConfig = None) -> dict[str, Any]:
        """Wrap the input dictionary."""
        return {self.name: input}


class RunnableTryFix(WLLMKRunnable):
    """A runnable that attempts multiple fixes if a primary step fails.

    Parameters
    ----------
    primary_step : Runnable
        The primary step to run.
    fix_step : Runnable
        The fix step to run if the primary step fails.
    max_retries : int, optional
        The maximum number of retries, by default 3.
    log_failures : bool, optional
        Whether to log failures in the output dictionary, by default False. Raises an
        error if the input is not a dictionary.
    """

    primary_step: Runnable = Field()
    fix_step: Runnable = Field()
    max_retries: int = Field(default=3, gt=0)
    log_failures: bool = Field(default=False)

    def invoke(self, input: Any, config: RunnableConfig = None) -> Any:
        """Invoke the primary step and retry with the fix step if it fails."""
        if self.log_failures and not isinstance(input, dict):
            raise ValueError("Input must be a dictionary to log failures.")

        counter = 0
        current_input = input
        while counter < self.max_retries:
            try:
                # Try the primary step
                if self.log_failures:
                    return self.primary_step.invoke(current_input, config) | {
                        "failures": counter,
                        "last_input": current_input,
                    }
                return self.primary_step.invoke(current_input, config)
            except Exception as e:
                if counter == self.max_retries - 1:
                    # If max retries reached, re-raise the last exception
                    raise

                # Apply fix step
                current_input = self.fix_step.invoke(current_input, config)
                counter += 1

        # This should never be reached due to max_retries logic
        raise RuntimeError("Unexpected error in multi-retry pipeline")


class ProgressBarCallback(BaseCallbackHandler):
    """A callback handler that updates a progress bar after each chain run.

    See Also
    --------
    https://gist.github.com/BrandonStudio/638a629911e47fee29175ca5c0b7430c
    """

    def __init__(self, total: int):
        super().__init__()
        self.progress_bar = tqdm(total=total)  # define a progress bar

    def on_chain_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.progress_bar.update(1)
        return response

    def __enter__(self):
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__(exc_type, exc_value, exc_traceback)

    def __del__(self):
        self.progress_bar.__del__()


class ThreadedSequence(RunnableSequence):
    """A RunnableSequence that runs each input in a separate thread.

    ThreadedSequence overrides the `batch` method to run each input in a separate
    thread, instead of step-by-step parallism as done in RunnableSequence.

    See Also
    --------
    langchain_core.runnables.RunnableSequence
    """

    def batch(self, input: list[Any], config: RunnableConfig = None) -> list[Any]:
        """Run each input in a separate thread."""
        max_workers = config.get("max_workers", None) if config else None

        with ProgressBarCallback(total=len(input)) as pg:
            config = config or {}
            if "callbacks" in config:
                if any(
                    isinstance(cb, ProgressBarCallback) for cb in config["callbacks"]
                ):
                    pass
                else:
                    config["callbacks"].append(pg)
            else:
                config["callbacks"] = [pg]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                outputs = list(
                    executor.map(
                        lambda x: self.invoke(x, config),
                        input,
                    )
                )

        return outputs
