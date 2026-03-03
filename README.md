# Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation (ICLR2026)

<div align="center">

**Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation**

[[🌐 Website](https://embodied-r1.github.io)] [[📄 Paper](http://arxiv.org/abs/2508.13998)] [[🏆 ICLR2026 Version](https://openreview.net/forum?id=i5wlozMFsQ)] [[🤗 Model](https://huggingface.co/IffYuan/Embodied-R1-3B-v1)] [[🎯 Dataset](https://huggingface.co/datasets/IffYuan/Embodied-R1-Dataset)]

</div>

---

## 🔥 Updates


- **[2026-03-03]** 📦 **Dataset Open-Sourced!** The Embodied-R1 dataset is now publicly available at [Hugging Face Datasets](https://huggingface.co/datasets/IffYuan/Embodied-R1-Dataset).

- **[2026-01-27]** 🏆 **Accepted by (ICLR2026)!** Embodied-R1 are accepted to **The Fourteenth International Conference on Learning Representations (ICLR2026)**.

- **[2025-08-22]** 🤗 **Model Released!** The Embodied-R1 3B v1 checkpoint is now available at [Hugging Face Model Hub](https://huggingface.co/IffYuan/Embodied-R1-3B-v1).

- **[2025-08-21]** 🚀 **Inference Scripts Released!** We have released our inference prompts and scripts for embodied pointing abilities.

- **[2025-08-20]** 📚 **Models and Datasets Released!** We have released our pre-trained models, training datasets, and comprehensive evaluation benchmarks. Check out our [HuggingFace collection](https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081) for all available resources.

---

## 📖 Overview

**Embodied-R1** is a 3B vision-language model (VLM) designed for general robotic manipulation. Through an innovative **"Pointing"** mechanism and **Reinforced Fine-tuning (RFT)** training methodology, it effectively bridges the "seeing-to-doing" gap in robotics, achieving remarkable zero-shot generalization capabilities.

![Embodied-R1 Framework](assets/r1_framework_readme.jpg)
*Figure 1: Embodied-R1 framework overview, comprehensive performance evaluation, and zero-shot robotic manipulation demonstrations.*

---

## 🛠️ Setup

1.  **Clone the repository**:
    
    ```bash
    git clone https://github.com/pickxiguapi/Embodied-R1.git
    cd Embodied-R1
    ```
    
2.  **Create and activate Conda environment**:
    ```bash
    conda create -n embodied_r1 python=3.11 -y
    conda activate embodied_r1
    ```

3.  **Install dependencies for inference**:
    ```bash
    pip install transformers==4.51.3 accelerate
    pip install qwen-vl-utils[decord]
    ```

3.  **Install dependencies for training (optional)**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Inference

**Run the example code:**

~~~python
cd Embodied-R1/
python inference_example.py
~~~

### VTG Example

Task instruction: put the red block on top of the yellow block

**Before prediction (original image):**

<img src="assets/put the red block on top of the yellow block.png" width="400" alt="Original input image">

**After prediction (visualization result):**

<img src="assets/put the red block on top of the yellow block_visualized.png" width="400" alt="Visualization result with predicted points">



### RRG Example

Task instruction: put pepper in pan

**Before prediction (original image):**

<img src="assets/put pepper in pan.png" width="400" alt="Original input image">

**After prediction (visualization result):**

<img src="assets/put pepper in pan_visualized.png" width="400" alt="Visualization result with predicted points">



### REG Example

Task instruction: bring me the camel model

**Before prediction (original image):**

<img src="assets/roborefit_18992.png" width="400" alt="Original input image">

**After prediction (visualization result):**

<img src="assets/roborefit_18992_visualized.png" width="400" alt="Visualization result with predicted points">



### OFG Example

Task instruction: loosening stuck bolts

**Before prediction (original image):**

<img src="assets/handal_090002.png" width="400" alt="Original input image">

**After prediction (visualization result):**

<img src="assets/handal_090002_visualized.png" width="400" alt="Visualization result with predicted points">





---

## 📊 Evaluation

```bash
cd eval
python hf_inference_where2place.py
python hf_inference_vabench_point.py
...
```

## 🧠 Training

Training scripts are available in [`scripts/`](scripts):

```bash
# Stage 1 training
bash scripts/stage_1_embodied_r1.sh

# Stage 2 training (set your stage-1 checkpoint path first)
bash scripts/stage_2_embodied_r1.sh
```

Key training files:
- `scripts/config_stage1.yaml`
- `scripts/config_stage2.yaml`
- `scripts/stage_1_embodied_r1.sh`
- `scripts/stage_2_embodied_r1.sh`
- `scripts/model_merger.py` (for checkpoint merging and HF export)

## 📜 Citation

If you use our work in your research, please cite:

```bibtex
@article{yuan2026embodied,
  title={Embodied-r1: Reinforced embodied reasoning for general robotic manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Huang, Yaoting and Chen, Yibin and Ni, Fei and Dong, Zibin and Li, Pengyi and Zheng, Yan and Tang, Hongyao and Hao, Jianye},
  journal={The Fourteenth International Conference on Learning Representations},
  year={2026}
}

@article{yuan2026seeing,
  title={From seeing to doing: Bridging reasoning and decision for robotic manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Chen, Yibin and Dong, Zibin and Ni, Fei and Kou, Longxin and Liu, Jinyi and Li, Pengyi and Zheng, Yan and Hao, Jianye},
  journal={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
</div>
