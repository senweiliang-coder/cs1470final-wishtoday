# MedFusion Dataflow And Tensor Shapes

This note summarizes the runnable root-path implementation:

- [data_fairvision.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/data_fairvision.py:43)
- [fusion_net.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/fusion_net.py:767)
- [fusion_train.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/fusion_train.py:193)
- [Models/fundus_swin_network.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/Models/fundus_swin_network.py:7)
- [Models/unetr.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/Models/unetr.py:21)

Assumptions in this diagram:

- `model_base=transformer`
- fundus input size: `384 x 384`
- OCT input size: `96 x 96 x 96`
- `B = batch size`
- `C = num_classes`, where `C=4` for AMD and `C=2` for DR/Glaucoma

## 1. Main Forward Graph

```mermaid
flowchart TD
    A["FairVision .npz sample
    slo_fundus
    oct_bscans
    label"] --> B["FairVisionDataset
    train: returns low/high views
    val/test: low view is used"]

    B --> C0["Fundus tensor
    X[0]: [B, 3, 384, 384]"]
    B --> C1["OCT tensor
    X[1]: [B, 1, 96, 96, 96]"]

    C0 --> D0["FundusSwinBackbone
    tokens x: [B, 144, 1024]
    pooled: [B, 1024]"]
    C1 --> D1["UNETR_base_3DNet
    tokens x1: [B, 216, 768]
    pooled: [B, 768]"]

    D0 --> E0["KFR_fundus
    encoder(z): [B, 144, 256]
    mu_topk: [B, C, 256]
    sigma_topk: [B, C, 256]
    proxy_loss_fundus"]
    D1 --> E1["KFR_oct
    encoder(z): [B, 216, 256]
    mu_topk: [B, C, 256]
    sigma_topk: [B, C, 256]
    proxy_loss_oct"]

    E0 --> F0["fundus_guided
    mu + rand * sigma
    [B, C, 256]"]
    E1 --> F1["oct_guided
    mu + rand * sigma
    [B, C, 256]"]

    E0 --> G["PoE
    inputs: 2 x ([B, C, 256], [B, C, 256])
    output: [B, 1, C, 256]"]
    E1 --> G

    G --> H["mean over dim=1
    poe_embed: [B, C, 256]"]
    H --> I["flatten
    [B, C * 256]"]
    I --> J["fc_fundus
    global_fusion: [B, 1024]"]

    D0 --> K["FDE input 1
    x: [B, 144, 1024]"]
    D1 --> L["FDE input 2
    x1: [B, 216, 768]"]
    J --> M["FDE input 3
    shared_features: [B, 1024]"]
    F0 --> N["FDE input 4
    funds_guided: [B, C, 256]"]
    F1 --> O["FDE input 5
    octs_guided: [B, C, 256]"]

    K --> P["projector1: 1024 -> 2048
    y1: [B, 144, 2048]"]
    L --> Q["projector2: 768 -> 2048
    y2: [B, 216, 2048]"]

    P --> R["split into 1024 + 1024
    unique/common token halves"]
    Q --> S["split into 1024 + 1024
    unique/common token halves"]
    N --> T["guided_features_projector1
    256 -> 1024
    [B, C, 1024]"]
    O --> U["guided_features_projector2
    256 -> 1024
    [B, C, 1024]"]
    M --> V["shared_features_projector
    1024 -> 1024
    [B, 1, 1024]"]

    R --> W["self_attn1
    query: [B, C, 1024]
    key/value: [B, 144, 1024]
    output -> mean: y1_uni [B, 1024]"]
    S --> X["self_attn2
    query: [B, C, 1024]
    key/value: [B, 216, 1024]
    output -> mean: y2_uni [B, 1024]"]
    T --> W
    U --> X
    R --> Y["cross_attn1
    query: [B, 1, 1024]
    key/value: [B, 144, 1024]
    output: y1_common [B, 1024]"]
    S --> Z["cross_attn2
    query: [B, 1, 1024]
    key/value: [B, 216, 1024]
    output: y2_common [B, 1024]"]
    V --> Y
    V --> Z

    W --> AA["concat per modality
    y1: [B, 2048]"]
    Y --> AA
    X --> AB["concat per modality
    y2: [B, 2048]"]
    Z --> AB

    AA --> AC["BatchNorm1d
    y1: [B, 2048]"]
    AB --> AD["BatchNorm1d
    y2: [B, 2048]"]

    AC --> AE["combined_features
    cat(y1 unique, y1_common+y2_common, y2 unique)
    [B, 3072]"]
    AD --> AE

    AE --> AF["classifier fc
    3072 -> 64 -> C
    pred: [B, C]"]
    AF --> AG["loss
    label-smoothing CE
    + proxy KL
    + proxy losses
    + 0.001 * FDE loss"]
```

## 2. Training-Time Two-View Path

```mermaid
flowchart LR
    A["One batch from FairVisionDataset"] --> B["data_low
    fundus: [B, 3, 384, 384]
    OCT: [B, 1, 96, 96, 96]"]
    A --> C["data_high
    same shapes
    optionally noisy"]

    B --> D["MedFusion forward
    pred1, loss_model, combined_features1
    combined_features1: [B, 3072]"]
    C --> E["MedFusion forward
    pred2, _, combined_features2
    combined_features2: [B, 3072]"]

    D --> F["MK_MMD
    align combined_features1 and combined_features2"]
    E --> F
    D --> G["final train loss
    loss_model + loss_MMD"]
    F --> G
```

## 3. Shape Cheat Sheet

| Module | Input | Output |
| --- | --- | --- |
| `FairVisionDataset` | `.npz` | `X[0]: [B,3,384,384]`, `X[1]: [B,1,96,96,96]` |
| `FundusSwinBackbone` | `[B,3,384,384]` | tokens `[B,144,1024]`, pooled `[B,1024]` |
| `UNETR_base_3DNet` | `[B,1,96,96,96]` | tokens `[B,216,768]`, pooled `[B,768]` |
| `KFR_fundus` | `[B,144,1024]` | `z=[B,144,256]`, `mu/sigma=[B,C,256]` |
| `KFR_oct` | `[B,216,768]` | `z=[B,216,256]`, `mu/sigma=[B,C,256]` |
| `PoE` | two modalities of `[B,C,256]` | `[B,1,C,256]` |
| `fc_fundus` | `[B,C*256]` | `global_fusion=[B,1024]` |
| `FDE` | tokens + guided features + shared feature | `combined_features=[B,3072]` |
| `classifier fc` | `[B,3072]` | logits `[B,C]` |
| `MK_MMD` | two x `[B,3072]` | scalar alignment loss |

## 4. Notes For Reading The Code

- The current 3D backbone in [Models/unetr.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/Models/unetr.py:21) is a lightweight 3D Conv adapter that emits `216` tokens. It is not a full canonical UNETR implementation.
- In [fusion_net.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/fusion_net.py:60), `KFR` currently returns batch-repeated class proxy tensors `[B, C, 256]`; it is closer to proxy-guided regularization than a true sample-specific top-k selector.
- In [fusion_net.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/fusion_net.py:13), `PoE` still contains stale sampling code, but the actual returned tensor is `unsqueeze(mu) + unsqueeze(var)`, not a sampled tensor.
- In [fusion_train.py](/home/liang/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation/fusion_train.py:193), training uses both low/high views and adds `MK_MMD`; validation and test only use the low/clean view.

## 5. Minimal ASCII View

```text
NPZ
 |- slo_fundus ------------------> [B, 3, 384, 384] --Swin--> [B, 144, 1024] --KFR--> [B, C, 256]
 |- oct_bscans ------------------> [B, 1, 96, 96, 96] --3DNet-> [B, 216, 768] --KFR--> [B, C, 256]

two [B, C, 256] --PoE--> [B, 1, C, 256] --mean/flatten/fc--> global_fusion [B, 1024]

([B, 144, 1024], [B, 216, 768], global_fusion [B, 1024], two guided [B, C, 256])
    --FDE--> combined_features [B, 3072]
    --fc--> logits [B, C]

train only:
    low view  -> combined_features1 [B, 3072]
    high view -> combined_features2 [B, 3072]
    MK_MMD(combined_features1, combined_features2)
```
