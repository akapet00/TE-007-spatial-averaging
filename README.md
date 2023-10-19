# TE-007-spatial-averaging

Spatial averaging scheme for reference level limits on whole body ambient exposure to RF EMFs at the 0.1-6 GHz range.

# Use

To run the notebooks, easiest way is to create a local environment by using `conda` as
```bash
conda create --name <environment name> python=3.9
```
activating the environment
```bash
conda activate <environment name>
```
and run the following command
```bash
pip install -r requirements.txt
```
to install all dependencies listed in `requirements.txt`.

In addition, `polatory` should be builed manually.
Separate instructions are available for:
- [Windows](https://github.com/polatory/polatory/blob/main/docs/build-windows.md)
- [macOS](https://github.com/polatory/polatory/blob/main/docs/build-macos.md)
- [Ubuntu](https://github.com/polatory/polatory/blob/main/docs/build-ubuntu.md)

After building, `polatory` can be installed by running
```bash
python -m pip install .
```
inside the cloned `polatory` repository within the newly created `conda` environemnt.

# Run

Start the notebooks by simply running
```bash
jupyter lab
```
inside the `notebooks` directory.

# License

[MIT](https://github.com/akapet00/TE-007-spatial-averaging/blob/main/LICENSE)

# Author

Ante Kapetanović
