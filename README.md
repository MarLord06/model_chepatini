# Severstal Steel Defect Detection 🔬

Sistema de detección de defectos en acero usando Deep Learning con PyTorch.

## 📁 Estructura del Proyecto

```
model_chepatini-main/
├── predict.py              # Script rápido para predicciones
├── train.py                # Script rápido para entrenamiento
├── requirements.txt        # Dependencias del proyecto
├── submission.csv          # Última submission generada
│
├── configs/                # Configuraciones
│   ├── config.py          # Configuración principal
│   └── test_config.json   # Config para tests
│
├── scripts/               # Scripts CLI
│   └── main.py            # Punto de entrada principal
│
├── src/                   # Código fuente
│   ├── models/            # Arquitecturas de redes
│   │   ├── classifier.py  # Clasificador binario (EfficientNet)
│   │   ├── segmentation.py# Modelos de segmentación (UNet, UNet++, etc.)
│   │   ├── ensemble.py    # Ensemble de modelos
│   │   └── losses.py      # Funciones de pérdida (Dice, Focal, etc.)
│   │
│   ├── data/              # Datasets y utilidades de datos
│   │   └── dataset.py     # Datasets de clasificación y segmentación
│   │
│   ├── training/          # Pipelines de entrenamiento
│   │   └── trainer.py     # Trainer unificado
│   │
│   ├── inference/         # Predicción y post-procesamiento
│   │   ├── predictor.py   # Pipeline de predicción + TTA
│   │   └── visualizer.py  # Visualización de predicciones
│   │
│   ├── analysis/          # Análisis y EDA
│   │   └── analyzer.py    # Análisis de dataset y entrenamiento
│   │
│   └── utils/             # Utilidades
│       └── helpers.py     # Funciones auxiliares (RLE, etc.)
│
├── data/                  # Datos
│   ├── train.csv          # CSV con etiquetas de entrenamiento
│   ├── sample_submission.csv
│   ├── train_images/      # Imágenes de entrenamiento
│   └── test_images/       # Imágenes de test
│
├── checkpoints/           # Modelos entrenados
│   ├── classifier_*/      # Checkpoints del clasificador
│   └── segmentation_*/    # Checkpoints de segmentación
│
├── visualizations/        # Visualizaciones generadas
│
└── deprecated/            # Código antiguo (TensorFlow)
    ├── predict_tensorflow.py
    ├── train_tensorflow.py
    ├── model_chepatini_tensorflow.py
    └── dataset_basic.py
```

## 🏗️ Arquitectura

El sistema usa un enfoque de **dos etapas**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: BINARY CLASSIFIER                          │
│              (EfficientNet-B4 Backbone)                          │
│              → Predicts: Has defect? (Yes/No)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
         Has Defect                      No Defect
              │                               │
              ▼                               ▼
┌─────────────────────────────┐    ┌─────────────────────────┐
│  STAGE 2: SEGMENTATION      │    │   Return Empty Mask     │
│  ENSEMBLE                   │    │   (No RLE encoding)     │
│  ┌────────────────────────┐ │    └─────────────────────────┘
│  │ U-Net + EfficientNet-B4│ │
│  ├────────────────────────┤ │
│  │ U-Net++ + SE-ResNeXt50 │ │
│  └────────────────────────┘ │
│       ↓ Weighted Average    │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSOR                                │
│  • Threshold binarization                                        │
│  • Remove small connected components (per-class min area)        │
│  • Morphological operations (close, open)                        │
│  • Resize to original dimensions (256 × 1600)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Clases de defectos
- 🔴 **Clase 1**: Defectos tipo 1
- 🟢 **Clase 2**: Defectos tipo 2
- 🔵 **Clase 3**: Defectos tipo 3
- 🟡 **Clase 4**: Defectos tipo 4

## 🚀 Uso Rápido

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Entrenar modelos
```bash
# Entrenar clasificador
python scripts/main.py train-classifier --csv_path data/train.csv --image_dir data/train_images

# Entrenar segmentación
python scripts/main.py train-segmentation --csv_path data/train.csv --image_dir data/train_images

# Entrenar ensemble completo
python scripts/main.py train-ensemble --csv_path data/train.csv --image_dir data/train_images
```

### Generar predicciones
```bash
python predict.py submit \
    --classifier checkpoints/classifier_efficientnet_b4_*/best_model.pth \
    --segmentation checkpoints/segmentation_unet_*/best_model.pth checkpoints/segmentation_unetplusplus_*/best_model.pth \
    --test_dir data/test_images \
    --output submission.csv
```

### Visualizar predicciones
```bash
python predict.py visualize \
    --submission submission.csv \
    --test_dir data/test_images \
    --num_images 10 \
    --save_dir visualizations
```

## 📊 Resultados

- **Modelos entrenados**: Clasificador + 2 modelos de segmentación
- **Device**: CUDA (RTX 3070 - 8GB)
- **Imágenes de test**: 5,495
- **Predicciones con defectos**: ~20.6%

## 📦 Dependencias principales

- PyTorch 2.7+ (CUDA 11.8)
- segmentation-models-pytorch
- albumentations
- opencv-python
- pandas, numpy, matplotlib

## 📝 Notas

- El código TensorFlow antiguo está en `deprecated/` para referencia
- Los checkpoints se guardan automáticamente durante el entrenamiento
- TTA (Test-Time Augmentation) está habilitado por defecto para mejor precisión

## 👤 Autor

Proyecto para la competencia [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) de Kaggle.
