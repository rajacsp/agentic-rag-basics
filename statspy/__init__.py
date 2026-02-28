"""
Canadian Statistics Package
Placeholder module for Canadian data and information
"""

__version__ = "1.0.0"
__author__ = "Canada Stats Team"

from .basics import CanadianProvinces, CanadianCities
from .tests import run_tests

__all__ = ["CanadianProvinces", "CanadianCities", "run_tests"]
