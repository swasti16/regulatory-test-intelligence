"""
Project-wide pytest fixtures and setup.

Docling's layout model needs TORCHDYNAMO_DISABLE / TORCH_COMPILE_DISABLE
set on CPU-only Windows to avoid requiring an MSVC cl.exe compiler for
torch.compile — a compiler that isn't needed for inference at all, just
triggered by torch's default compile path. Setting this here (rather
than relying on `setx` at the OS level) makes tests portable across any
machine/CI runner without manual environment setup.
Must be set BEFORE docling/torch are imported anywhere in the test run.
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
