# AGENTS.md

Repository-wide instructions for AI coding agents and contributors.

---

# Core Principles

When instructions conflict, follow this priority order:

1. User request
2. Project-specific requirements
3. This document
4. Personal preferences

Prefer correctness over speed and simplicity over cleverness.

---

# 1. Clarify Before Implementing

Do not make silent assumptions.

Before changing code:

* State important assumptions.
* If multiple interpretations are possible, present the options.
* Ask for clarification when requirements are ambiguous.
* Recommend simpler solutions when they solve the same problem.
* Explain meaningful tradeoffs instead of choosing one silently.

If the requirements are unclear, stop and ask.

---

# 2. Keep Solutions Simple

Implement the smallest solution that satisfies the request.

Avoid:

* Unrequested features
* Premature abstractions
* Future-proofing without a demonstrated need
* Configuration options that are not required
* Handling impossible or irrelevant scenarios

A good solution should be easy for another engineer to understand and maintain.

Before submitting, ask:

> Is there a simpler way to achieve the same result?

---

# 3. Make Surgical Changes

Modify only the code required to satisfy the request.

When working in existing code:

* Preserve the existing style and structure.
* Do not refactor unrelated code.
* Do not rewrite nearby code unless necessary.
* Do not change formatting outside the affected area.
* Do not remove existing dead code unless explicitly requested.

You may remove:

* Imports made unused by your changes
* Variables made unused by your changes
* Functions made unused by your changes

Every changed line should have a direct connection to the requested work.

---

# 4. Work Toward Verifiable Outcomes

Define success before implementation.

Examples:

| Request             | Verification                                       |
| ------------------- | -------------------------------------------------- |
| Fix a bug           | Reproduce the bug, then verify it no longer occurs |
| Add validation      | Add tests for invalid inputs and verify they pass  |
| Refactor code       | Ensure behavior remains unchanged before and after |
| Improve performance | Measure and compare results                        |

For non-trivial tasks, create a short execution plan:

```text
1. Change X
   Verify: Y

2. Change A
   Verify: B

3. Final validation
   Verify: C
```

Avoid declaring completion without verification.

---

# 5. Communication Standards

Be transparent.

* Say what you know.
* Say what you do not know.
* Distinguish facts from assumptions.
* Surface risks and limitations.
* Do not claim code works unless it has been verified.

Prefer:

> "This should work, but has not been tested."

over:

> "Fixed."

---

# Success Criteria

These guidelines are successful when they result in:

* Smaller diffs
* Fewer unnecessary refactors
* Less speculative code
* Fewer regressions
* More clarification before implementation
* Clear verification of completed work

---

# Project Notes

Operational guidance for Windows + INP + ONNX Runtime deployments.

---

## Environment Compatibility

Maintain compatibility between ONNX Runtime GPU and the CUDA runtime available on target systems.

Recommended versions:

| CUDA Runtime | ONNX Runtime GPU         |
| ------------ | ------------------------ |
| CUDA 11.8    | onnxruntime-gpu==1.18.1  |
| CUDA 12.x    | Matching ORT GPU release |

Runtime verification:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

Expected output must include:

```text
CUDAExecutionProvider
```

---

## Windows DLL Initialization

For standalone tools, configure CUDA and cuDNN DLL paths before importing heavy dependencies.

In `pyqt_threshold_sorter.py`, DLL setup must remain at the top of the file, before:

* cv2
* numpy
* PyQt5
* ONNX Runtime dependent modules

Required PATH entries:

```text
...\Lib\site-packages\nvidia\cudnn\bin
...\Lib\site-packages\nvidia\cublas\bin
...\Lib\site-packages\nvidia\cuda_runtime\bin
```

---

## ONNX Export Requirements

Use:

```bash
export_inp_onnx.py --dynamic-batch
```

After export, verify dynamic batch metadata.

Example:

```text
['batch_size', 3, 448, 448]
```

## Benchmark Checklist

Before and after performance-related changes, record:

* Milliseconds per cycle (28 expected crops)
* GPU utilization
* VRAM usage
* Score drift between legacy and default modes

Performance changes should be validated with measurable results.
