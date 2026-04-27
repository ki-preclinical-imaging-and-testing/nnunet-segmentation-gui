# Segmentation GUI

A desktop application for running segmentation models on 3D medical images.

What is it?

This GUI lets you load 3D medical images (NIFTI format), run pre-trained nnUNet models or YOLO models to generate segmentations, and save the results. It's designed to be straightforward: load an image, select a model, run prediction, view the results.

<img width="763" height="676" alt="Screenshot from 2026-04-27 16-37-10" src="https://github.com/user-attachments/assets/ec3db989-bdfe-4864-ada5-ff3888a0571c" />


## Installation
1. Requirements

    Python 3.10+
    8GB RAM minimum
    NVIDIA GPU with CUDA (optional, can use CPU)

2. Create Virtual Environment

### Linux/macOS:

Terminal:
```
python3.10 -m venv venv
source venv/bin/activate
```

### Windows:

Powershell:
```
python -m venv venv
venv\Scripts\activate`
```

3. Install Dependencies

Terminal:
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
4. Run

Terminal:
```
python main.py
```

## How to Use
### Loading Images

Click "Load Image" or drag-and-drop a .nii.gz file onto the viewer. Use the slice slider at the bottom to scroll through slices. Click "Axis: Axial" to switch between axial, coronal, and sagittal views.
### Adding Models

Click "Add Model..." and select your checkpoint_best.pth file. Give it a name and description. Models are saved to config.yaml and persist between sessions.
Running Predictions

Select a model from the dropdown, click "Load Model", then click "Run Prediction". A progress dialog will show until the prediction completes. Previous predictions are detected automatically.
### Viewing Results

The segmentation overlays on the image. Use the opacity slider to adjust transparency. Click "Toggle Overlay" to show/hide it. Click "Save Segmentation" to export the result as a NIFTI file.
