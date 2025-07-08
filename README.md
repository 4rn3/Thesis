# DIFFUSION MODELS FOR ENERGY TIME SERIES GENERATION

This repository contains the implementation, evaluation, and thesis manuscript for the master's thesis titled:

> **DIFFUSION MODELS FOR ENERGY TIME SERIES GENERATION**  
> By Arne  
> Master’s Thesis in Artificial Intelligence

The project explores the use of conditional denoising diffusion models for generating synthetic energy time series data and evaluates how well these synthetic samples capture the statistical and predictive structure of real smart meter data.

---

## 🧠 Project Overview

The goal of this thesis is to generate realistic synthetic electricity consumption data using diffusion-based generative models. This supports data availability for energy research, while preserving privacy and enabling robust downstream tasks such as load forecasting and grid optimization.

Key contributions:
- A conditional 1D diffusion model architecture tailored for time series.
- Comparison of multiple conditioning strategies (time features, weather, statistics).
- Systematic evaluation of realism via statistical distance metrics, Discriminative evaluation and TSTR (Train on Synthetic, Test on Real) performance.

---

## 📄 Thesis Document

Key sections include:
- Background on generative modeling and smart meter data
- Diffusion model architecture and conditioning design
- Experimental setup and dataset preprocessing
- Quantitative and qualitative evaluation of generation fidelity
- 
---

## 📝 License

This code is intended for academic use only.  
See `LICENSE` for more information
