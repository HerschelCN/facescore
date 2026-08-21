# Third-Party Notices

This repository is an independent CPU inference adaptation. It is not an
official release of, or endorsed by, the FPEM authors or their institutions.
Names and trademarks belong to their respective owners.

## Source code included in this repository

### FPEM

- Project: FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos with Face Retouching
- Upstream: https://github.com/Estella-LH/FPEM
- Source revision used: `c2965425247d7bf8b764d27e4483a06fc7a061e5`
- Files adapted here: `fpem_official/Clips.py`, `fpem_official/VQA.py`,
  `fpem_official/VIT.py`, and `fpem_official/SwinFace_arch.py`
- Copyright: Copyright (c) 2025 Estella-LH
- License: MIT License, reproduced in `fpem_official/LICENSE`

Changes made for this repository include fixing execution to CPU, loading the
checkpoint with `map_location="cpu"`, disabling CUDA AMP use, preserving the
batch dimension for batch size one, constructing the CLIP ViT-B/16 architecture
without downloading redundant pretrained weights, replacing three small timm
helper imports with local equivalents, and adding current-Python compatibility.
The detailed behavioral changes are also listed in `README.md`.

### pytorch-image-models (timm)

The upstream FPEM `nets/VIT.py` expressly identifies Ross Wightman's
`pytorch-image-models` Vision Transformer implementation as its original code
source. Small timm-style helper implementations are also present in
`fpem_official/SwinFace_arch.py`.

- Upstream: https://github.com/huggingface/pytorch-image-models
- Copyright notice in the upstream license: Copyright 2019 Ross Wightman
- License: Apache License 2.0, reproduced in
  `third_party/pytorch-image-models-LICENSE.txt`
- Files modified in this distribution: `fpem_official/VIT.py` and
  `fpem_official/SwinFace_arch.py`

### Swin Transformer

`fpem_official/SwinFace_arch.py`, obtained through FPEM, implements the Swin
Transformer architecture. The official Microsoft implementation is identified
here for attribution and license visibility.

- Upstream: https://github.com/microsoft/Swin-Transformer
- Copyright: Copyright (c) Microsoft Corporation
- License: MIT License, reproduced in `third_party/Swin-Transformer-LICENSE.txt`

## Runtime dependencies not copied into this repository

Packages in `requirements_cpu.txt` are installed separately and remain subject
to the license terms in their own distributions. In particular:

- OpenAI CLIP: https://github.com/openai/CLIP — MIT License,
  Copyright (c) 2021 OpenAI
- facenet-pytorch: https://github.com/timesler/facenet-pytorch — MIT License,
  Copyright (c) 2019 Timothy Esler
- PyTorch and torchvision: https://github.com/pytorch/pytorch and
  https://github.com/pytorch/vision
- NumPy: https://github.com/numpy/numpy
- Pillow: https://github.com/python-pillow/Pillow
- einops: https://github.com/arogozhnikov/einops
- packaging: https://github.com/pypa/packaging

No source code from these separately installed packages is redistributed here
except for the provenance described in the bundled-source sections above.

## Model checkpoint and datasets

The FPEM checkpoint and all datasets are intentionally excluded from this Git
repository. The README links to the checkpoint hosted by the FPEM project, but
linking to or downloading an artifact does not itself grant redistribution or
commercial-use rights. Users should confirm the applicable terms with the
artifact or dataset owner before redistribution, deployment, or commercial use.

This notice documents provenance and is not legal advice.
