# Geometry-Aware Spherical Sampling (GASS) for Diverse Text-to-Image Generation

Ye Zhu (*LIX, CNRS, École Polytechnique & CS, Princeton*), Kaleb Newman (*CS, Princeton*), Johannes F. Lutzeyer (*LIX, CNRS, École Polytechnique*), Adriana Romero-Soriano (*FAIR at Meta - Montreal & McGill University & Mila & Canada CIFAR AI chair*), Michal Drozdzal (*FAIR at Meta - Montreal*), Olga Russakovsky (*CS, Princeton*)

This is the official Pytorch implementation of the ICML 2026 paper **[GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](https://arxiv.org/abs/2602.17200)**.

Below we show non-cherry-picked qualitative results of our proposed **GASS** sampling method compared to the vanilla CFG (classifier-free guidance) sampling baseline and other more recent diversity enhancement methods.



<p align="center">
	<img src="assets/teaser.png", width="800">



## 1. Motivation and problem


We consider the task of amplifying the sample diversity for text-to-image generative models given a fixed prompt. Instead of framing this as an entropy enhancement problem like most prior work does, we formulate this as a geometrical challenge, with the goal of increasing the geometrical spread covered by a batch of generated images within a hypersphere (CLIP space in our case). 


<p align="center">
	<img src="assets/sphere.png", width="600">



## 2. Environment setup

In our experiments, we used Python 3.15.5 and PyTorch 2.8.0+cu128. In principle, our method is rather framework-agnostic. As long as your local environment supports the installation and execution of the base models (e.g., SD2.1 and SD3), it should work fine.



## 3. Dataset preparation

Our pre-processed prompt files of ImageNet and Drawbench can be found in the folder ```./datasets```. For ImageNet, we used a template ```A photo of [class name]``` for each prompt. 


## 4. Base CFG and GASS Sampling 

Our main experiments were conducted using two base models: SD2.1 (based on a U-Net diffusion architecture) and SD3-M (which uses a DiT backbone with Rectified Flow).

Due to the compliance requirements of the 2026 EU AI Act, the official release of SD2.1 has been deprecated. Consequently, we are only releasing our implementation based on SD3-M. However, the implementation of GASS is framework-agnostic and highly consistent across different architectures. If necessary, our SD3-M code can be easily adapted and transferred to other open-source text-to-image (T2I) models.

Using DrawBench as an example benchmark, you can run the following command to perform vanilla sampling with CFG: ```python sd3_run_drawbench.py```. To run our GASS sampling method on the same benchmark, use: ```python sd3_gass_sampling.py```.


You can easily run experiments on ImageNet by manually modifying the prompt file path inside the scripts.


## 5. Citation

If you find our work interesting and useful, please consider citing it.

```
@inproceedings{zhu2026gass,
  title={GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation},
  author={Zhu, Ye and Newman, Kaleb S. and Lutzeyer, Johannes F. and Romero-Soriano, Adriana and Drozdzal, Michal and Russakovsky, Olga},
  booktitle={International Conference on Machine Learning},
  year={2026},
}
```


### Acknowledgements

This project is primarily supported through the research grant from Meta Inc..
