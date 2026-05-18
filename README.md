# PBR Material Classifier — Automatic Texture Tagger for Maya

A Maya tool that looks at the textures on objects in your scene and uses a machine learning model to identify what type of material each one is (wood, rock, metal, ground, or fabric). It tags each material with their type and then automatically sorts texture files into organized category folders on disk.

---

## Features

- Looks at the colour texture on each object and predicts which of **5 material types** it is: wood, rock, metal, ground, or fabric
- Tags each shader the scene with predicted material types and a confidence score
- Moves texture files into named category folders (`textures/wood/`, `textures/metal/`, etc.) and updates all the file paths inside Maya automatically
- Option to scan every object in the scene at once or objects selected
- View mesh's texture path and confidence level across all five categories
- Type in or browse to any folder on your machine as the destination for the sorted textures

---

## Requirements

- **Autodesk Maya** 2020 or later (tested on 2026)
- **PyTorch, TorchVision, Pillow** — must be installed and accessible outside Maya (inference runs via subprocess)
- **Python 3.x** — Maya's built-in interpreter

---

## How It Works

Maya's embedded Python interpreter cannot import PyTorch directly. The ML layer and the Maya tool are kept deliberately separate:

- **Google Colab** handles training and exports the model as `pbr_classifier.pth`
- **`classifier.py`** loads the checkpoint and runs inference locally (model is loaded lazily on the first prediction call and stays in memory for the session)
- **`SceneScanner.py`** handles all Maya-side logic: mesh collection, texture extraction, metadata writing, and file organization
- **`SceneScannerUI.py`** is the PySide6 tool window parented to Maya's main window

NOTE: Only albedo maps are used as classifier input. 

---

## Dataset

The training dataset was collected from CGTrader and PolyHaven's PBR texture libraries using their public APIs.

| Category | Images |
|----------|--------|
| fabric   | 187    |
| ground   | 265    |
| metal    | 136    |
| rock     | 243    |
| wood     | 204    |
| **Total**| **1,035** |

Images were collected at mixed resolutions (1K–2K) to help the model generalize across the resolution variety found in real Maya scenes. The dataset was split 70/15/15 into train, validation, and test sets.

**Check out the Dataset on Kaggle:**
[https://www.kaggle.com/datasets/alanadubie/pbr-dataset-kaggle](https://www.kaggle.com/datasets/alanadubie/pbr-dataset-kaggle)

---

## Training

The classifier is a pretrained ResNet-18 fine-tuned via transfer learning using PyTorch and TorchVision. The final fully connected layer was replaced to output probabilities across the 5 material classes. Training uses cross-entropy loss with class weights to handle the metal category imbalance, the Adam optimizer, and a learning rate scheduler.

Training runs on Google Colab (free T4 GPU runtime) and exports a `.pth` checkpoint.

**Check out the training notebook on Colab:**
[https://colab.research.google.com/drive/1f6ShO1LfgN9ZRgPZ6Mw8SKys3Lrp3RiX?usp=sharing](https://colab.research.google.com/drive/1f6ShO1LfgN9ZRgPZ6Mw8SKys3Lrp3RiX?usp=sharing)

---

## Installation

1. Clone or download this repository into your Maya scripts folder:

    ```
    C:\Users\<YourUsername>\Documents\maya\<MayaVersion>\scripts\pbr-ml-classifier\
    ```

2. Open Maya, go to the **Script Editor**, and run the MEL command below in the **MEL tab** to enable the command port:

    ```mel
    commandPort -name "localhost:7001" -sourceType "mel" -echoOutput;
    ```

3. In the **Python tab**, run the ```launch_scene_scanner.py``` script:

---

## Installing Python Libraries (PyTorch, Pillow & Torchvision) for Maya

 These must be installed, otherwise the tool will fail with a `ModuleNotFoundError`.

**Step 1 — Find your mayapy.exe**

It lives in the same folder as maya.exe. The default path is:

```
C:\Program Files\Autodesk\Maya<version>\bin\mayapy.exe
```

For example, for Maya 2026:
```
C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe
```

**Step 2 — Open Command Prompt as Administrator**

Search for `cmd` in the Windows Start menu, right-click it, and choose **Run as administrator**. This is required to install packages into the Program Files directory.

**Step 3 — Install libraries using mayapy**

Run the following command, replacing the path with your actual Maya version:

```
"C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe" -m pip install torch torchvision pillow
```

This installs PyTorch, TorchVision, and Pillow (for image loading) directly into Maya's Python environment. It may take a few minutes to download.

**Step 4 — Verify the install**

Run this to confirm PyTorch is accessible from mayapy:

```
"C:\Program Files\Autodesk\Maya2026\bin\mayapy.exe" -c "import torch; print(torch.__version__)"
```

If it prints a version number like `2.x.x+cpu`, you're good to go. If it throws an error, re-run Step 3 and make sure you used the correct mayapy path.
