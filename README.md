# PBR Material Classifier — ML-Assisted Texture Pipeline Tool for Maya

A Maya pipeline tool that analyzes shader networks and uses a lightweight machine learning classifier to assist in identifying material types from albedo textures (wood, rock, metal, ground, fabric).

The system integrates directly into Maya scenes to tag shader nodes with material metadata and confidence scores, and supports automated organization of texture assets into structured, publish-ready directory layouts for improved asset management and pipeline consistency.

---

## Features

### Shader & Scene Analysis
- Traverses Maya shading networks (shading engine → shader → file texture nodes)
- Extracts **albedo textures** from connected materials
- Supports full scene or selection-based processing
- Recursively resolves nested hierarchy selections

### ML-Assisted Classification
- Predicts one of five material categories: wood, rock, metal, ground, fabric
- Outputs:
  - predicted label
  - confidence score
  - full class probability breakdown
- Lightweight inference using a ResNet-18 model trained on curated PBR datasets

### Scene Integration
- Tags shaders with:
  - `materialType` (string attribute)
  - `mlConfidence` (float attribute)
- Stores metadata directly in Maya scene files for persistence

### Texture Pipeline & Publishing
- Organizes textures into structured category folders:

```
textures/
  ├── wood/
  ├── rock/
  ├── metal/
  ├── ground/
  └── fabric/
```
- Deduplicates shared texture usage across multiple objects
- Updates Maya file nodes after relocation to preserve scene integrity
- Supports custom output directory for asset publishing workflows


---


## Requirements

- **Autodesk Maya** 2020 or later (tested on 2026)
- **PyTorch, TorchVision, Pillow** — must be installed into Maya's Python environment (see setup guide below)
- **Python 3.x** — Maya's built-in interpreter

---

## How It Works

Maya's embedded Python interpreter cannot import PyTorch directly. The ML layer and the Maya tool are kept separate:

- **Google Colab** handles training and exports the model as `pbr_classifier.pth`
- **`classifier.py`** loads the checkpoint and runs inference locally — the model loads lazily on the first prediction call and stays in memory for the session
- **`pbr_tools.py`** handles all Maya-side logic: mesh collection, texture extraction, metadata writing, and file organization
- **`tool_window.py`** is the PySide6 tool window parented to Maya's main window


---

## Dataset

The model was trained on a curated dataset of PBR textures sourced from AmbientCG and Poly Haven with CC0 liscense.

| Category | Images |
|----------|--------|
| fabric   | 187    |
| ground   | 265    |
| metal    | 136    |
| rock     | 243    |
| wood     | 204    |
| **Total**| **1,035** |

Images were collected at mixed resolutions (1K–2K) to help the model generalize across the resolution variety found in real Maya scenes. The dataset was split 70/15/15 into train, validation, and test sets.

**Download the dataset on Kaggle:**
[https://www.kaggle.com/datasets/alanadubie/pbr-dataset-kaggle](https://www.kaggle.com/datasets/alanadubie/pbr-dataset-kaggle)

---

## Training

The classifier is a pretrained ResNet-18 fine-tuned via transfer learning using PyTorch and TorchVision. The final fully connected layer was replaced to output probabilities across the 5 material classes. Training uses cross-entropy loss with class weights to handle the metal category imbalance, the Adam optimizer, and a learning rate scheduler.

Training runs on Google Colab (free T4 GPU runtime) and exports a `.pth` checkpoint.

**Open the training notebook on Colab:**
[https://colab.research.google.com/drive/1f6ShO1LfgN9ZRgPZ6Mw8SKys3Lrp3RiX?usp=sharing](https://colab.research.google.com/drive/1f6ShO1LfgN9ZRgPZ6Mw8SKys3Lrp3RiX?usp=sharing)

---

## Installation

1. Clone or download this repository into your Maya scripts folder:

    ```
    C:\Users\<YourUsername>\Documents\maya\<MayaVersion>\scripts\pbr-ml-classifier\
    ```

2. Open Maya, go to the **Script Editor**, and run this in the **MEL tab** to enable the command port:

    ```mel
    commandPort -name "localhost:7001" -sourceType "mel" -echoOutput;
    ```

3. In the **Python tab**, run `launch.py` to open the tool. You can also paste its contents into a **shelf button** for one-click access.

---

## Installing Python Libraries (PyTorch, TorchVision & Pillow)

These must be installed into Maya's own Python environment, otherwise the tool will fail with a `ModuleNotFoundError`.

**Step 1 — Find your mayapy.exe**

It lives in the same folder as maya.exe:

```
C:\Program Files\Autodesk\Maya<version>\bin\mayapy.exe
```

For example, for Maya 2026:
```
C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe
```

**Step 2 — Open Command Prompt as Administrator**

Search for `cmd` in the Start menu, right-click, and choose **Run as administrator**.

**Step 3 — Install the libraries**

```
"C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe" -m pip install torch torchvision pillow
```

This may take a few minutes to download.

**Step 4 — Verify the install**

```
"C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe" -c "import torch; print(torch.__version__)"
```

If it prints a version number like `2.x.x+cpu` you're good to go. If it throws an error, re-run Step 3 and confirm you used the correct mayapy path.

> The tool runs on CPU — no GPU required. Training happens in Colab; Maya only runs inference on one image at a time.

---

## License

    Copyright 2026 Alana Dubie

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.