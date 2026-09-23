"""Evaluation harness: `run_eval(model, protocol, generator) -> RunRecord`.

A protocol declares everything that makes two numbers comparable; its identity hashes those
declared fields and the data they select, never source files. Runs are cached by (model
identity, protocol identity), so repeating a cell is a no-op. The generator is the seam:
production uses `vqar.inference.VLLMGenerator`, tests use a scripted fake.
"""

from vqar.eval.protocol import Protocol, protocol_identity
from vqar.eval.record import ModelRef, RunRecord
from vqar.eval.runner import Cell, find_cell, run_eval

__all__ = ["Cell", "ModelRef", "Protocol", "RunRecord", "find_cell", "protocol_identity",
           "run_eval"]
