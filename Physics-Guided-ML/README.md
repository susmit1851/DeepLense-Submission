# Gravitational Lensing Classification — Physics-Informed ResNet-50 (PINN)

Classifies gravitational lensing images into 3 classes — **No Lensing (`no`)**, **Sphere (`sphere`)**, and **Vortex (`vort`)** — using a Physics-Informed Neural Network (PINN) built on a pretrained ResNet-50.

---

## What Makes This a PINN

A learnable **Gravitational Lensing Layer** is prepended to the backbone. It computes three physics-derived channels from the Singular Isothermal Sphere (SIS) deflection angle formula:

$$\vec{\alpha}(\vec{\theta}) = \theta_E^2 \frac{\vec{\theta}}{|\vec{\theta}|^2}$$

These channels — deflection magnitude $|\vec{\alpha}|$, $\alpha_x$, and $\alpha_y$ — are concatenated with the raw image to form a **4-channel input**. The Einstein radius $\theta_E$ is a learnable parameter.

The loss function combines classification and physics terms:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{phys} \cdot \mathcal{L}_{phys}$$

where $\lambda_{phys} = 0.5$ and $\mathcal{L}_{phys}$ penalises the gap between the predicted and layer-estimated $\theta_E$.

---

## Dataset

| Split      | Size   |
|------------|--------|
| Train      | 27,000 |
| Test       | 3,000  |
| Validation | 7,500  |

Classes: `no` → 0, `sphere` → 1, `vort` → 2

---

## Training Configuration

| Hyperparameter   | Value                          |
|------------------|--------------------------------|
| Backbone         | ResNet-50 (pretrained)         |
| Input channels   | 4 (image + 3 physics maps)     |
| Epochs           | 10                             |
| Batch size       | 16                             |
| Learning rate    | 1e-4                           |
| Weight decay     | 1e-4                           |
| Optimizer        | Adam + gradient clipping (1.0) |
| Scheduler        | ReduceLROnPlateau              |
| λ physics        | 0.5                            |
| Saved model      | `resnet_50_pinn_best.pth`      |

---

## Results

**Best Test Accuracy: 93.03%**  
**Validation Accuracy: 93.03%**  
**Test Physics Loss: 0.0002** *(near-zero — lensing layer well-calibrated)*

### Classification Report (Validation Set)

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| no      | 0.89      | 0.99   | 0.94     | 2500    |
| sphere  | 0.94      | 0.90   | 0.92     | 2500    |
| vort    | 0.97      | 0.90   | 0.94     | 2500    |
| **avg** | **0.93**  | **0.93** | **0.93** | **7500** |

### AUC-ROC Scores

| Class  | AUC    |
|--------|--------|
| no     | 0.9898 |
| sphere | 0.9818 |
| vort   | 0.9912 |

---

## AUC-ROC Curve



### Model Weights:
https://drive.google.com/file/d/1qtxCkJgBqewqytrnmJYI-2qDAUauHhVx/view?usp=sharing
