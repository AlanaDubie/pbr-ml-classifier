# PBR Machine-Learning CNN Texture Classifier & File Orginization

**PBR ML Classifier** is a Maya Python tool that uses a model to classify textures in a scene automatically assigns tags to the meshes and orginizes the textures. 
---

### Features


---

### ⚙️ Installation & Setup

#### Requirements

- **Autodesk Maya**: 2020 or later (tested on 2026)
- **Python**: Maya’s built-in Python (3.x recommended)
- **Pytorch**: ML python library

---

#### Setup Instructions

1. **Clone or download** this repository and save it to your Maya scripts folder:

    ```
    C:\Users\<YourUsername>\Documents\maya\<YourMayaVersion>\scripts
    ```

2. **Launch Maya**, open the **Script Editor**, and switch to the **Python** tab.

3. Run the following MEL command in the **MEL tab** of the **Script Editor**:

    ```
    commandPort -name "localhost:7001" -sourceType "mel" -echoOutput;
    ```

4. **Import and run the tool** 
