# FPEM 单图人脸评分（Windows / CPU）

这是论文 **FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos with Face Retouching**（ICCV 2025）的最小单图命令行版本。它使用官方网络结构、官方预处理流程和官方检查点，只在 CPU 上推理，不调用 CUDA，也不需要 LiveBeauty 数据集。

## 快速开始

克隆仓库并进入项目目录：

```powershell
git clone https://github.com/HerschelCN/facescore.git
cd facescore
```

### 安装依赖

支持 Python 3.10–3.14。`requirements_cpu.txt` 会按 Python 版本选择兼容的纯 CPU PyTorch：

使用当前系统 Python：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements_cpu.txt
```

或创建独立 Conda 环境：

```powershell
conda create -n fpem-cpu python=3.10 -y
conda activate fpem-cpu
python -m pip install --upgrade pip
pip install -r requirements_cpu.txt
```

`requirements_cpu.txt` 明确安装 PyTorch CPU 轮子，不包含 CUDA 运行库。Python 3.13/3.14 使用 PyTorch 2.13；Python 3.10–3.12 使用已验证的 PyTorch 2.5.1。脚本内含 OpenAI CLIP 对新版 setuptools 移除 `pkg_resources` 的兼容处理。

### 下载官方权重

模型权重约 556 MiB，超过 GitHub 普通文件大小限制，因此不包含在本仓库中。请把官方检查点下载到：

```text
pretrained/fpem_srcc_0.9243.pth
```

它来自 [FPEM 官方仓库](https://github.com/Estella-LH/FPEM) 的同名 Git LFS 对象：

- 文件大小：`582528073` 字节
- SHA-256：`5e18fb2a807366bd1f4bf8498493437059c60d4862a2f5b54ab664cc92af9615`

如果文件缺失，可在项目根目录执行：

```powershell
New-Item -ItemType Directory -Path ".\pretrained" -Force | Out-Null
Invoke-WebRequest -Uri "https://media.githubusercontent.com/media/Estella-LH/FPEM/c2965425247d7bf8b764d27e4483a06fc7a061e5/pretrained/fpem_srcc_0.9243.pth" -OutFile ".\pretrained\fpem_srcc_0.9243.pth"
Get-FileHash ".\pretrained\fpem_srcc_0.9243.pth" -Algorithm SHA256
```

不要使用随机初始化模型生成分数。也可以显式指定另一份官方检查点：

```powershell
python score.py ".\face.png" --checkpoint "D:\models\fpem_srcc_0.9243.pth"
```

### 单张图片评分

```powershell
python score.py ".\face.png"
```

### 批量评分与网页报告

把图片放入 `face` 文件夹后运行：

```powershell
python batch_score.py
```

脚本只加载一次模型，逐张评分，并生成：

- `face_scores.json`：文件名与对应分数；
- `face_scores.html`：由 JSON 结果生成的照片评分页面，可直接双击打开，无需启动本地服务器。

每次重新运行 `batch_score.py` 都会原子更新这两个文件。

输出格式：

```text
FPEM score: 1.6579 / 5.0000
```

支持 `.png`、`.jpg`、`.jpeg`（扩展名大小写均可）。带透明通道的 PNG 会先转换为普通 RGB。

输入图片应当：

- 只包含一张完整、清晰、已由用户手动裁剪的人脸；
- 尽量保留整张脸，不要只截取局部五官；
- 不要求人脸框、landmark 或额外 CSV；脚本不会自动检测或裁脸。

脚本按官方 `FaceDataset` 流程把同一张图转换为 `[0,1]` 张量，并生成 224、112、160 三种尺寸：保持比例缩放，再在右侧和下侧补黑，不做额外 normalize。模型把五个吸引力等级的 softmax 概率按 1–5 加权，直接输出论文模型的原始量表分数。

第一次运行需要加载约 556 MiB 检查点并构建约 2.07 亿参数的模型，可能较慢。已在 Python 3.12.13 / PyTorch 2.5.1+cpu 与 Python 3.14 / PyTorch 2.13.0+cpu 上验证；前者单张官方示例人脸首次完整运行（含模型构建和权重加载）约 6.8 秒。

该输出是模型对这张照片的 **1–5 分预测**，不是客观颜值真值，不应据此对个人作价值判断。光线、姿态、表情、妆容、滤镜和裁剪都会影响结果。

## 实现来源与 CPU 改动

必要网络模块来自官方仓库提交 `c2965425247d7bf8b764d27e4483a06fc7a061e5`，保留在 `fpem_official/`，许可证见 `fpem_official/LICENSE`。CPU 适配仅包括：

- 将模型设备固定为 CPU，并以 `map_location="cpu"` 加载检查点；
- 禁用官方 Swin 模块中的 CUDA AMP 上下文；
- 修正官方代码在 batch size 为 1 时误删 batch 维的 `squeeze`；
- 直接构建官方 OpenAI CLIP ViT-B/16 结构，随后由 FPEM 检查点严格加载其权重，避免重复下载一份 CLIP 权重；
- 内置官方代码仅使用的三个小型 `timm` helper，避免安装模型中心等无关依赖。

检查点的 `FPEM_add.` 外层前缀会被明确移除，之后使用 `strict=True` 加载全部 1242 个 state-dict 项；不会忽略缺失键或多余键。

论文：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FPEM_Face_Prior_Enhanced_Facial_Attractiveness_Prediction_for_Live_Videos_ICCV_2025_paper.html)

## 版权、许可与引用

本仓库是为 CPU 单图推理制作的非官方适配，并非 FPEM 作者或其所属机构的官方发行版或背书产品。

- `fpem_official/` 中的实质性代码修改自 [FPEM](https://github.com/Estella-LH/FPEM)；上游版权归原作者所有，按 MIT License 使用，完整文本见 [`fpem_official/LICENSE`](fpem_official/LICENSE)。
- `fpem_official/VIT.py` 的上游文件明确注明源自 Ross Wightman 的 `pytorch-image-models`（timm）；相关代码按 Apache License 2.0 使用，完整文本见 [`third_party/pytorch-image-models-LICENSE.txt`](third_party/pytorch-image-models-LICENSE.txt)。
- Swin Transformer、OpenAI CLIP、facenet-pytorch 及其他运行时依赖仍归各自权利人所有；具体归属、许可链接以及本仓库的修改清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
- 模型检查点和训练数据不包含在本仓库中。源码许可证不应被自动理解为对模型权重、数据集、论文内容、名称或商标授予相同权利；再分发或商用前应另行核对上游条款。
- 除第三方文件自身许可证明确授予的权利外，本仓库目前没有为其余原创包装代码声明统一的开源许可证。

学术或研究使用请引用原论文：

```bibtex
@inproceedings{li2025fpem,
  title     = {FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos with Face Retouching},
  author    = {Li, Hui and Ren, Xiaoyu and Yu, Hongjiu and Chen, Ying and Li, Kai and Wang, L. and Min, Xiongkuo and Duan, Huiyu and Zhai, Guangtao and Liu, Xu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025}
}
```
