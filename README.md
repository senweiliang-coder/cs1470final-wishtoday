## FRFD: Feature-Refined and Disentangled Representations for Robust Multimodal Ophthalmic Disease Grading

post:8193b27d97713d3459c6b45bf962bd1.jpg
report:csci1470final (3).pdf
reflection:midreflection (1).pdf

This project studies multimodal ophthalmic disease grading with paired `fundus` and `OCT` inputs. The current runnable codebase uses a fusion pipeline built around:

- `KFR`: key-feature regularization for proxy-guided representation learning on each modality
- `PoE`: product-of-experts fusion over modality-wise latent distributions
- `FDE`: feature disentanglement and enhancement for shared and modality-specific fusion

In the current implementation:

- `KFR` is the module name used in code for the proxy-guided feature regularization block
- `FDE` is the module name used in code for the disentangled fusion block
- the default 2D/3D encoders are configurable through `--fundus_encoder` and `--oct_encoder`

### 1. Dataset
Harvard 30K / FairVision30K  
https://yutianyt.com/projects/fairvision30k/

### 2. Method Overview
The main model is `MedFusion`, defined in [fusion_net.py](./fusion_net.py).

Its forward pipeline is:

1. Extract 2D fundus tokens and 3D OCT tokens with configurable encoders.
2. Use `KFR_fundus` and `KFR_oct` to produce proxy-guided latent features for each modality.
3. Fuse the two modality distributions with `PoE` to obtain a shared global representation.
4. Use `FDE` to combine shared features with modality-specific guided features.
5. Predict the final disease grade from the fused representation.

### 3. Run
Train the end-to-end framework:

`./Run_fusion`

Test a checkpoint:

`./Run_test`

### 4. Environment
See `requirement.txt`.
