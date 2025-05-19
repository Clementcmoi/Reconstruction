# reco-plugin: Reconstruction Napari Plugin

This project is a Napari plugin for scientific image or data reconstruction workflows.

## Requirements

- Python 3.12+
- [Anaconda](https://www.anaconda.com/)
- [Napari](https://napari.org/)
- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Astra Toolbox](https://www.astra-toolbox.com/)
- [Cupy](https://cupy.dev/)
- Other dependencies listed in `requirements.txt` or `pyproject.toml`

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Clementcmoi/Reconstruction.git
cd Reconstruction
pip install -r requirements.txt
```

To install the plugin in editable/development mode:

```bash
pip install -e .
```

## Usage

Start Napari:

```bash
napari
```

Then, access the plugin via the Napari plugins menu (`Plugins > reco-plugin > Reconstruction` or `Multi Paganin`).

## Project Structure

- Plugin source code: `src/reco_plugin/`
- Plugin manifest: `src/reco_plugin/napari.yaml`
- Data files: `.npy`, `.tif` (ignored by Git)
- Notebooks: `.ipynb` (ignored by Git)

## License

This project is licensed under the MIT License.