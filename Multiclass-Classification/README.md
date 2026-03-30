# Multiclass Classification 

Classifies grayscale flow pattern images into 3 classes: **No Flow**, **Sphere**, and **Vortex**, using a pretrained ResNet-50 fine-tuned for single-channel input.

---

## Dataset

| Split      | Size   |
|------------|--------|
| Train      | 27,000 |
| Test       | 3,000  |
| Validation | 7,500  |

Classes: `no` → 0, `sphere` → 1, `vort` → 2

---

## Training

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 (pretrained) |
| Input channels | 1 (grayscale) |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | Adam (weight decay 1e-4) |
| Scheduler | ReduceLROnPlateau |
| Loss | CrossEntropyLoss |

---

## Results

**Best Test Accuracy: 92.67%**  
**Validation Accuracy: 92.47%**

### Classification Report (Validation Set)

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| no      | 0.89      | 0.98   | 0.94     | 2500    |
| sphere  | 0.94      | 0.87   | 0.90     | 2500    |
| vort    | 0.95      | 0.92   | 0.93     | 2500    |
| **avg** | **0.93**  | **0.92** | **0.92** | **7500** |

### AUC-ROC Scores

| Class  | AUC    |
|--------|--------|
| no     | 0.9899 |
| sphere | 0.9778 |
| vort   | 0.9893 |

---

## AUC-ROC Curve

![ROC Curve](https://drive.google.com/file/d/1tW3vd4lwGjYWMqsnXnXrGba3JsFliwSd/view?usp=sharing)

### Model Weights
https://drive.google.com/file/d/1_vx9CVTbSq5qbLU5xY3zltEuiBEKkoep/view?usp=sharing