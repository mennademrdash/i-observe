"""Pytest bootstrap.

Caps BLAS thread counts before numpy is imported so collection cannot die with
"OpenBLAS error: Memory allocation still failed after 10 retries" on low-RAM machines.
"""
import config  # noqa: F401
