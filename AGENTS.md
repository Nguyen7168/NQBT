# AGENTS.md

Operational notes for future agents working on this repo (Windows + INP/ONNX runtime).

## 1) Environment compatibility (critical)
- Keep ONNX Runtime GPU aligned with CUDA runtime available on target machines.
- For CUDA 11.8 deployments, prefer `onnxruntime-gpu==1.18.1`.
- For CUDA 12.x deployments, use a matching ORT GPU build.
- Verify at runtime:
  - `import onnxruntime as ort`
  - `ort.get_available_providers()` includes `CUDAExecutionProvider`.

## 2) Windows DLL loading order
- CUDA/cuDNN DLL directories must be configured before heavy imports in standalone tools.
- In `pyqt_threshold_sorter.py`, keep DLL setup at the top of file (before `cv2`, `numpy`, `PyQt5`, ORT-dependent paths).
- If import errors occur, verify `PATH` contains:
  - `...\\Lib\\site-packages\\nvidia\\cudnn\\bin`
  - `...\\Lib\\site-packages\\nvidia\\cublas\\bin`
  - `...\\Lib\\site-packages\\nvidia\\cuda_runtime\\bin`

## 3) ONNX export policy for INP
- Use `export_inp_onnx.py` with `--dynamic-batch`.
- After export, verify input metadata shows dynamic batch:
  - e.g. `['batch_size', 3, 448, 448]`.

## 4) INP mode parity
- If `models.inp.inp_mode=legacy_script`, strict/fast paths must preserve legacy preprocessing/scoring behavior.

## 5) Main app (`run.py`) performance note
- `run.py` gains from batch > 1 only if INP detector computes ONNX in true batch mode.
- Keep `app/models/anomaly.py` `_InpOnnxDetector._infer_default` batched.

## 6) Benchmark checklist (before/after changes)
- ms/cycle over expected 28 crops.
- GPU utilization.
- VRAM usage.
- Score drift between legacy/default paths.

