# DVCF-Net: Difference-aware Verification Cross Fusion for Microscopy Image Reconstruction

DVCF-Net is a dual-input microscopy reconstruction network that combines a low-resolution wide-field (WF) image with a corresponding structured illumination microscopy (SIM) reconstruction. It aims to preserve reliable structures observed in WF imaging while selectively incorporating high-frequency SIM information and suppressing inconsistent or potentially artifactual details.

This repository contains a standalone PyTorch implementation for training and inference, pretrained weights for the included microtubule examples, and two paired WF/SIM test images.

> **Repository scope:** The original training dataset is not distributed with this repository. The files under `testdata/` are inference examples and do not include independent ground truth.

## Highlights

- Dual-input reconstruction from WF and SIM images.
- Difference-aware comparison using absolute feature differences and element-wise products.
- Four explicit paths for common, WF-specific, SIM-specific, and conflicting information.
- Multi-level fusion using the 5th, 10th, 15th, and 20th WF RRDB features.
- Final high-resolution fusion after both branches have been upsampled.
- Joint pixel, perceptual, and relativistic adversarial training objectives.
- End-to-end processing of single-channel microscopy images.

## Method Overview

Given a low-resolution WF image and a SIM image at the target resolution, DVCF-Net extracts four intermediate WF features. The SIM feature is resized to the WF feature resolution and progressively fused with these representations through four cascaded DVCF modules. The two branches are then independently upsampled and combined by a fifth DVCF module at the target resolution.

```text
WF (1 x H x W)
  |
  +-- 3x3 Conv -- 23 RRDB ----------------------------------+
          |          |          |          |                 |
        RRDB-5    RRDB-10    RRDB-15    RRDB-20             |
          |          |          |          |                 |
          +----------+----------+----------+                 |
                                            |                |
SIM (1 x 2H x 2W) -- 3x3 Conv -- Resize ----+                |
                             |                               |
                       4 cascaded DVCF                        |
                             |                               |
                    residual refinement                      |
                             |                               |
                       2x upsampling                 2x upsampling
                             |                               |
                             +------ final DVCF -------------+
                                        |
                                    3x3 Conv
                                        |
                              SR (1 x 2H x 2W)
```

The default model uses 64 base feature channels, 32 dense growth channels, and 23 RRDB blocks in the WF backbone. Each of the five DVCF modules contains an additional refinement RRDB, giving 28 RRDB blocks in the complete generator.

### Difference-aware Verification Cross Fusion

For WF and SIM features, \(F_{wf}\) and \(F_{sim}\), a DVCF module computes projected features

\[
Q_{wf}=P_{wf}(F_{wf}), \qquad Q_{sim}=P_{sim}(F_{sim}),
\]

and constructs an absolute difference and element-wise product:

\[
D=|Q_{wf}-Q_{sim}|, \qquad P=Q_{wf}\odot Q_{sim}.
\]

The comparison feature is obtained from the concatenation of \(Q_{wf}\), \(Q_{sim}\), \(D\), and \(P\). Four independent `3x3 Conv + Sigmoid` mappings generate:

- a consistency map, \(G_c\);
- a conflict-reliability map, \(G_r\);
- a WF selection map, \(G_{wf}\);
- a SIM selection map, \(G_{sim}\).

All mappings operate on 64-channel features and do not share parameters. They control the common, WF-specific, SIM-specific, and conflict paths. The four paths are concatenated, projected back to 64 channels, added to a `1x1` base projection, and refined by one RRDB.

The implementation is in [`models/modules/dvcf_architecture.py`](models/modules/dvcf_architecture.py).

## Network Configuration

| Component | Default setting |
|---|---:|
| Input/output channels | 1 / 1 |
| Upscaling factor | 2 |
| Base feature channels | 64 |
| Dense growth channels | 32 |
| WF backbone RRDBs | 23 |
| DVCF modules | 5 |
| Refinement RRDBs inside DVCF | 5 |
| Total generator RRDBs | 28 |
| Intermediate WF features | RRDB 5, 10, 15, and 20 |

Each RRDB contains three residual dense blocks. Each residual dense block contains five `3x3` convolutions and uses residual scaling of 0.2.

## Complexity and Inference Efficiency

The theoretical complexity below uses a `128x128` WF patch, a `256x256` SIM patch, batch size 1, and counts convolutional and fully connected operations. FLOPs use the convention `1 MAC = 2 FLOPs`.

| Metric | Value |
|---|---:|
| Generator parameters | 24.209 M |
| Discriminator parameters | 16.959 M |
| Generator MACs | 481.021 G |
| Generator FLOPs | 962.041 G |
| Approximate FP32 generator weight size | 92.35 MiB |

Measured generator-only inference performance:

| Hardware | WF / SIM input | Precision | Mean latency | Throughput | Peak allocated memory |
|---|---:|---:|---:|---:|---:|
| NVIDIA RTX 3090 | `128x128 / 256x256` | FP32 | 104.52 ms | 9.57 patches/s | 557.25 MiB |
| NVIDIA RTX 3090 | `512x512 / 1024x1024` | FP32 | 1501.29 ms | 0.67 images/s | 7521.93 MiB |

Measurements used PyTorch 1.13.1, CUDA 11.6, batch size 1, `torch.inference_mode()`, and CUDA-event timing. They exclude image loading, host-to-device transfer, and result saving. Performance can vary with hardware and software versions.

## Perceptual Feature Extractor

The feature reconstruction loss uses a fixed ImageNet-pretrained VGG-19 network:

\[
\Phi(\cdot)=\operatorname{VGG19}_{\mathrm{conv5\_4}}(\cdot).
\]

The extractor is truncated at the output of `conv5_4`, before its following ReLU, and all its parameters are frozen. A single-channel image is replicated into three identical channels and normalized with ImageNet statistics:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

The feature loss is the mean L1 distance between the reconstructed and ground-truth `conv5_4` features.

## Repository Structure

```text
DVCF_clean/
|-- data/
|-- models/
|   `-- modules/
|       |-- architecture.py
|       |-- block.py
|       |-- dvcf_architecture.py
|       `-- loss.py
|-- options/
|   |-- train/train_dvcf.json
|   `-- test/test_dvcf.json
|-- testdata/
|   |-- WF/
|   `-- SIM/
|-- weights/
|   `-- MTs.pth
|-- utils/
|-- train.py
|-- test.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.8 or later
- PyTorch and torchvision
- NumPy
- OpenCV
- LMDB
- A CUDA-capable GPU is recommended

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your CUDA environment before installing the remaining packages. Dependency versions are currently not pinned; record the Python, PyTorch, torchvision, CUDA, and cuDNN versions when reproducing experiments.

## Quick Inference

The default configuration points to the included microtubule examples and pretrained generator:

```bash
python test.py -opt options/test/test_dvcf.json
```

The relevant paths are:

```json
{
  "dataroot_LR": "testdata/WF",
  "dataroot_SIM": "testdata/SIM",
  "pretrain_model_G": "weights/MTs.pth"
}
```

Reconstructed TIFF images are written to:

```text
results/DVCF_test/MTs_test/
```

| Sample | WF input | SIM input | Output |
|---|---:|---:|---:|
| `1.tif` | `512x512` | `1024x1024` | `1024x1024` |
| `2.tif` | `512x512` | `1024x1024` | `1024x1024` |

WF and SIM files are paired by sorted filename order. The script saves the final reconstruction only; it does not export intermediate DVCF maps or features.

### CPU Inference

The default configuration uses GPU 0. For CPU inference, change the following entry in [`options/test/test_dvcf.json`](options/test/test_dvcf.json):

```json
"gpu_ids": []
```

CPU inference is supported but can be slow because of the large number of RRDB operations.

## Preparing a Training Dataset

Organize WF, SIM, and ground-truth images in separate directories:

```text
datasets/
|-- train/
|   `-- CCPs/
|       |-- WF/
|       |-- SIM/
|       `-- GT/
`-- test/
    `-- CCPs/
        |-- WF/
        |-- SIM/
        `-- GT/
```

For example:

```text
datasets/train/CCPs/WF/0001.tif
datasets/train/CCPs/SIM/0001.tif
datasets/train/CCPs/GT/0001.tif
```

Images are paired by sorted filename order. Ensure all three directories contain the same number of files and use a consistent naming convention.

Default training crops:

| Image | Patch size |
|---|---:|
| WF | `128x128` |
| SIM | `256x256` when SIM matches the GT resolution |
| Ground truth | `256x256` |

The loader converts grayscale images to single-channel floating-point tensors in `[0, 1]`. Augmentation includes random horizontal flips, vertical flips, and 90-degree rotations.

## Training

Update the dataset paths in [`options/train/train_dvcf.json`](options/train/train_dvcf.json), then run:

```bash
python train.py -opt options/train/train_dvcf.json
```

Default training configuration:

| Setting | Value |
|---|---:|
| Total iterations | 500,000 |
| Batch size | 2 |
| Generator/discriminator learning rate | `1e-4 / 1e-4` |
| Adam betas | `(0.9, 0.999)` |
| Learning-rate milestones | 50k, 100k, 200k, 300k |
| Learning-rate decay factor | 0.5 |
| Pixel loss weight | 5.0 |
| Feature loss weight | 0.5 |
| Adversarial loss weight | 0.5 |
| Validation frequency | 2,000 iterations |
| Checkpoint frequency | 2,000 iterations |
| Random seed | 9 |

The generator objective is

\[
\mathcal{L}_G = 5\mathcal{L}_{pix}+0.5\mathcal{L}_{fea}+0.5\mathcal{L}_{adv}.
\]

- \(\mathcal{L}_{pix}\): L1 reconstruction loss.
- \(\mathcal{L}_{fea}\): L1 loss between fixed VGG-19 `conv5_4` features.
- \(\mathcal{L}_{adv}\): relativistic average adversarial loss implemented with `BCEWithLogitsLoss`.

Training outputs are saved under:

```text
experiments/DVCF_train/
|-- models/
|-- training_state/
|-- val_images/
`-- train_*.log
```

To resume training, set `resume_state` in the training JSON. Generator and discriminator checkpoints can also be initialized separately through `pretrain_model_G` and `pretrain_model_D`.

The first run may require torchvision to download the ImageNet-pretrained VGG-19 weights. The implementation uses FP32 training and does not currently enable automatic mixed precision.

## Evaluation Notes

The example configuration uses `testdata/SIM` for both the SIM input and the nominal HR path because independent ground truth is not included. The examples are suitable for demonstrating inference and measuring runtime, but they must not be used to report independent PSNR or SSIM results.

For quantitative evaluation, prepare a separate test set with WF, SIM, and independent ground-truth images. Besides PSNR and SSIM, microscopy experiments may benefit from evaluating structural continuity, artifact suppression, and temporal consistency in live-cell sequences.

## Known Limitations

- The original training dataset and historical training-time logs are not included.
- The example data do not contain independent ground truth.
- The test script does not automatically calculate PSNR or SSIM.
- The discriminator expects `256x256` targets because its fully connected input size is fixed.
- Automatic mixed precision is not enabled.
- Tiled inference and overlap-aware stitching are not implemented.
- Dependency versions are not pinned.
- A license file is not currently included.

## Citation

If this repository accompanies a published paper, replace the placeholder below with the final bibliographic information:

```bibtex
@article{dvcf,
  title   = {DVCF-Net: Difference-aware Verification Cross Fusion for Microscopy Image Reconstruction},
  author  = {Author List},
  journal = {Journal or Conference},
  year    = {Year}
}
```

## License

No license is currently included. Before public release, add a license compatible with the source code, pretrained weights, and example microscopy images, and document the terms for each artifact where necessary.

## Acknowledgements

This implementation uses RRDB/ESRGAN-style residual dense blocks, torchvision's ImageNet-pretrained VGG-19 features, and PyTorch. We thank the authors and maintainers of the related open-source projects and microscopy reconstruction research.
