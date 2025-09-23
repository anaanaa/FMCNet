  <h1 align="center">Frequency-enhanced Multi-granularity Context
Network for Efficient Vertebrae Segmentation</h1>


  <p align="center">
    <img src="assets/Logo.png" alt="Description of the image" width="400" height="395">
  <p align="center">



  <p align="center">
    <a href="https://github.com/goupup-ai/miccai25" rel="external nofollow noopener" target="_blank"><strong>Jian Shi</strong></a>
    ·
    <a href="https://github.com/goupup-ai/miccai25" rel="external nofollow noopener" target="_blank"><strong>Tianqi You</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=MfbIbuEAAAAJ&hl=zh-CN" rel="external nofollow noopener" target="_blank"><strong>Pingping Zhang*</strong></a>
    ·
    <a target="_blank"><strong>Hongli Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=rgtOrCMAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Rui Xu</strong></a>
    ·
    <a target="_blank"><strong>Haojie Li</strong></a>
  </p>
<p align="center">
    <a href="https://arxiv.org/pdf/2506.23086" rel="external nofollow noopener" target="_blank">MICCAI 2025 Paper</a>
  <p align="center">
    <img src="assets/Challenging examples.png" alt="Description of the image" style="width:100%;">
  <p align="center">
<p align="center" style="font-size: 18px; color: gray;">
    Figure 1: Challenging examples in vertebrae segmentation.
</p>
<p align="center">
    <img src="assets/Overall.png" alt="Description of the image" style="width:100%;">
</p>
<p align="center" style="font-size: 18px; color: gray;">
    Figure 2: Overall Framework of FMC-Net.
</p>


 **FMC-Net** is a novel vertebrae segmentation framework. Specifically, we first apply wavelet transform for lossless downsampling to reduce the feature distortion in blurred images. The decomposed high and low-frequency components are then processed separately. For the high-frequency components, we apply a High-frequency Feature Refinement (HFR) to amplify the prominence of key features and filter out noises, restoring fine-grained details in blurred images. For the low frequency components, we use a Multi-granularity State Space Model (MG-SSM) to aggregate feature representations with different receptive fields, extracting spatially-varying contexts while capturing long-range dependencies with linear complexity. The utilization of multi-granularity contexts is essential for distinguishing similar vertebrae and improving segmentation accuracy. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches on both CT and MRI vertebrae segmentation datasets.

---

## News 📢

- We released the **FMC-Net** codebase and paper! [Paper](https://arxiv.org/pdf/2506.23086)
- Great news! Our paper has been accepted to **MICCAI 2025**! 🏆

---

## Table of Contents 📑

- [Introduction](#introduction)
- [Contributions](#contributions)
- [Experimental Results](#experimental-results)
- [Visualizations](#visualizations)
- [Reproduction](#reproduction)
- [Citation](#citation)

---

## **Introduction** 🌟

Automated and accurate segmentation of individual vertebra in 3D CT and MRI images is essential for various clinical applications. Due to the limitations of current imaging techniques and the complexity of spinal structures, existing methods still struggle with reducing the impact of image blurring and distinguishing similar vertebrae.

To alleviate these issues, we introduce a Frequency-enhanced Multi-granularity Context Network (FMC-Net) to improve the accuracy of vertebrae segmentation.

---

## **Contributions** 

**1.** We propose FMC-Net that leverages wavelet transform to avoid information loss and enhances high and low-frequency features to address the challenges of image blurring and the high similarity of vertebral structures.

**2.** For the high-frequency components, we introduce the HFR to restore fine grained details. For the low-frequency components, we employ the MG-SSM to capture spatially-varying long-range dependencies.

**3.** Experiments on VERSE2019 and LUMBAR datasets demonstrate the superior performance of FMC-Net compared to other state-of-the-art approaches.

## **Results**

### Segmentation[VERSE2019, LUMBAR]

<p align="center">
    <img src="assets/visualization.png" alt="seg Results" style="width:100%;">
</p>

---
### Ablation Studies [LUMBAR]

<p align="center">
    <img src="assets/Ablation.png" alt="Ablation Results" style="width:100%;">
</p>

---
## **Visualizations**

### Qualitative results with different methods

<p align="center">
    <img src="assets/seg.png" alt="Qualitative results with different methods" style="width:100%;">
</p>

---

## **Reproduction**

### Datasets

- [VerSe'19: Large Scale Vertebrae Segmentation Challenge](https://verse2019.grand-challenge.org/)
- **Lumbar**: [Lumbar vertebral body and intervertebral disc segmentation in multi-scanner and multi-modal MRI](https://osf.io/qx5rt/) 

---

### Training

```bash
conda create -n UMverse python=3.10
conda activate UMverse
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
cd (your_path)
pip install -r requirements.txt
cd (your_path/umverse)
pip install -e .
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerUMverse
```

---

## **Citation**

If you find **FMC-Net** helpful in your research, please consider citing:

```bibtex
@article{shi2025frequency,
  title={Frequency-enhanced Multi-granularity Context Network for Efficient Vertebrae Segmentation},
  author={Shi, Jian and You, Tianqi and Zhang, Pingping and Zhang, Hongli and Xu, Rui and Li, Haojie},
  journal={arXiv preprint arXiv:2506.23086},
  year={2025}
}
```
## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects:
  - [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
  - [U-Mamba](https://github.com/bowang-lab/U-Mamba)
