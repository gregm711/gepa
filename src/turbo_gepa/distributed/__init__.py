"""
Utilities for running TurboGEPA islands across multiple processes.
"""

from .local_multiworker import run_local_multiworker
from .runner import run_worker_from_factory
