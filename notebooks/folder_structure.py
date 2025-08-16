import os

# Define the main project folders (excluding 'data' which you already have)
folders = [
    "notebooks",         # Jupyter notebooks for experiments
    "src",               # Source code
    "src/preprocessing", # Preprocessing scripts
    "src/model",         # Model building and training scripts
    "src/utils",         # Helper functions
    "outputs",           # Model checkpoints, logs, plots
    "outputs/checkpoints",
    "outputs/plots",
    "app",               # Streamlit or Flask app files
]

# Create the folders if they don't exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Project folder structure created successfully!")
