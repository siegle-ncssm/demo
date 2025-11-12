"""
Base execution step abstraction for pipeline workflows.

This module provides a flexible framework for creating modular, reusable
execution steps that can be composed into ML pipelines.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any


class StepStatus(Enum):
    """Execution step status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep(ABC):
    """
    Abstract base class for execution steps in a pipeline.

    Each execution step represents a discrete unit of work in a larger pipeline.
    Steps can have dependencies, validate inputs, execute logic, and track metrics.

    Attributes:
        name: Human-readable name for the step
        description: Detailed description of what the step does
        dependencies: List of step names that must complete before this step
        status: Current execution status
        metadata: Dictionary for storing step-specific metadata
        start_time: When step execution started
        end_time: When step execution completed
        error: Error message if step failed
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize an execution step.

        Args:
            name: Unique name for this step
            description: Human-readable description
            dependencies: List of step names this step depends on
            config: Configuration dictionary for the step
        """
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.config = config or {}
        self.status = StepStatus.PENDING
        self.metadata: dict[str, Any] = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error: str | None = None
        self.logger = logging.getLogger(f"step.{name}")

    def can_execute(self, completed_steps: list[str]) -> bool:
        """
        Check if this step can execute based on dependencies.

        Args:
            completed_steps: List of step names that have completed

        Returns:
            True if all dependencies are satisfied
        """
        return all(dep in completed_steps for dep in self.dependencies)

    def validate(self) -> bool:
        """
        Validate that the step is ready to execute.

        Override this method to add custom validation logic.

        Returns:
            True if validation passes, False otherwise
        """
        return True

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the step's main logic.

        This is the core method that must be implemented by subclasses.

        Args:
            context: Shared context dictionary with data from previous steps

        Returns:
            Dictionary with results to be added to the context

        Raises:
            Exception: If step execution fails
        """
        pass

    def pre_execute(self, _context: dict[str, Any]) -> None:
        """
        Hook called before execute(). Override for custom setup logic.

        Args:
            _context: Shared pipeline context
        """
        # Default implementation does nothing
        return

    def post_execute(self, _context: dict[str, Any], _result: dict[str, Any]) -> None:
        """
        Hook called after successful execute(). Override for custom cleanup logic.

        Args:
            _context: Shared pipeline context
            _result: Results returned from execute()
        """
        # Default implementation does nothing
        return

    def on_failure(self, _context: dict[str, Any], error: Exception) -> None:
        """
        Hook called when execute() raises an exception.

        Args:
            _context: Shared pipeline context (unused in base implementation)
            error: The exception that was raised
        """
        self.logger.error(f"Step {self.name} failed: {str(error)}")

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the full step lifecycle.

        This method orchestrates the complete execution flow:
        1. Validation
        2. Pre-execution hooks
        3. Main execution
        4. Post-execution hooks
        5. Error handling

        Args:
            context: Shared pipeline context

        Returns:
            Dictionary with step results

        Raises:
            RuntimeError: If validation fails
            Exception: If execution fails and retry logic is exhausted
        """
        self.logger.info(f"Starting step: {self.name}")

        # Validate
        if not self.validate():
            raise RuntimeError(f"Validation failed for step: {self.name}")

        self.status = StepStatus.RUNNING
        self.start_time = datetime.now()

        try:
            # Pre-execution hook
            self.pre_execute(context)

            # Main execution
            result = self.execute(context)

            # Post-execution hook
            self.post_execute(context, result)

            self.status = StepStatus.COMPLETED
            self.end_time = datetime.now()
            self.logger.info(
                f"Step {self.name} completed in "
                f"{(self.end_time - self.start_time).total_seconds():.2f}s"
            )

            return result

        except Exception as e:
            self.status = StepStatus.FAILED
            self.end_time = datetime.now()
            self.error = str(e)
            self.on_failure(context, e)
            raise

    def skip(self, reason: str = "") -> None:
        """
        Mark this step as skipped.

        Args:
            reason: Optional reason for skipping
        """
        self.status = StepStatus.SKIPPED
        self.metadata["skip_reason"] = reason
        self.logger.info(f"Step {self.name} skipped: {reason}")

    def get_execution_time(self) -> float | None:
        """
        Get the execution time in seconds.

        Returns:
            Execution time in seconds, or None if not yet executed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the step execution.

        Returns:
            Dictionary with step summary information
        """
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "execution_time": self.get_execution_time(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation of the step."""
        return (
            f"<ExecutionStep(name='{self.name}', "
            f"status={self.status.value}, "
            f"dependencies={self.dependencies})>"
        )
