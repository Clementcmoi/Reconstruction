@echo off
REM Path to your Miniconda installation
CALL C:\Users\Bioran\anaconda3\Scripts\activate.bat

REM Activate the napari-env environment
CALL conda activate napari-env

REM Go to the plugin folder
cd /d "C:\path\to\your\plugin" 

REM Update the plugin from Git
git pull origin main

REM Return to the initial location if needed (optional)
cd /d "C:\Users\Bioran"

REM Launch Napari
napari