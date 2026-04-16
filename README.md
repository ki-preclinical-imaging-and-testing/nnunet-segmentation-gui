# 1. Create conda environment
conda create -n nnunet-gui python=3.10 -y
conda activate nnunet-gui

# 2. Install dependencies
pip install -r requirements_simple.txt

# 3. Setup nnUNet paths (one-time)
mkdir -p ~/nnUNet_raw ~/nnUNet_preprocessed ~/nnUNet_results

# 4. Link your models directory
# Option A: Copy models
cp -r /path/to/your/models ~/nnUNet_models

# Option B: Symbolic link (recommended for shared drives)
ln -s /path/to/shared/models ~/nnUNet_models

# 5. Or set environment variable
export NNUNET_MODELS="/path/to/your/models"

# 6. Run the GUI
python main.py


1. LOAD IMAGE
   - Click "Load Image"
   - Select a .nii or .nii.gz file
   - Use slider to navigate slices
   - Click "Axis" button to change view

2. LOAD MODEL
   - Select model from dropdown
   - Click "Load Model"
   - Models are auto-discovered from model_directory

3. RUN PREDICTION
   - Click "Run Prediction"
   - Results saved to "predictions" folder next to image
   - Prediction appears as overlay

4. EDIT SEGMENTATION
   - Click "Enable Edit"
   - Choose Paint or Erase tool
   - Adjust brush size with slider
   - Select label number to paint
   - Click and drag to edit
   - Use Undo/Redo as needed

5. SAVE RESULTS
   - Click "Save Segmentation"
   - Choose save location
   - Original spacing and affine preserved
