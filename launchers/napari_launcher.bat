@echo off
REM Path to your Miniconda installation (adapt if necessary)
CALL %USERPROFILE%\anaconda3\Scripts\activate.bat

REM Activate the napari-env environment
CALL conda activate napari-env

REM Launch napari
napari
