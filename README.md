# Installation
```bash
conda env create -f environment.yml
conda activate [insert title here]
```
<!--- Follow instructions for https://github.com/Pointcept/SegmentAnything3D installation, then place SegmentAnything3D in [env name]/lib/python3.8/ with a blank __init__.py file in it --->

Follow instructions for [SegmentAnything3D](https://github.com/Pointcept/SegmentAnything3D) installation of pointops, then place SegmentAnything3D in [env name]/lib/python3.8/ with a blank \_\_init\_\_.py file in it.
On line 23 of sam3d.py, replace
```python
from util import *
```
with 
```python
from .util import *
```



If the scenes being evaluated do not have .aggregation.json files, on line 140 of SegmentAnything3D/scannet-preprocess/preprocess_scannet.py, replace
```python
    if split_name != "test":
```
with
```python
    if False:
```

The list of folders that need to be populated manually are:
```
data/frames_square
data/sqa3dTest
```
The list of folders that are automatically populated are:
```
data/save2dmask
data/savepcd
data/scannet_3d
```

open a terminal in data/ScanNet and run
```bash
ln ../sqa3dTest/ scans
```
Some of the SegmentAnything3D preprocessing expects that file structure.


# Usage

run
```
pipeline.py
```

To convert room scans to captions.

Then run
```
PromptGenerator.py
```
to pair questions and captions.

ChatCaptioner and the finetuned BLIP-2 model unfortunately need mutually exclusive versions of transformers, so you'll need to switch between them manually.

BLIP-2:
```bash
pip install --upgrade transformers==4.26.1
```
ChatCaptioner:
```bash
pip install --upgrade transformers==4.27.0
```




# Evaluation on SQA3D

run 
```
sqa3dtest.py
```

This code was originally designed for use with local LLMs, so it should be easily modifiable if you want to do that.


<!--- TODO for README --->
<!--- Need SegmentAnything3D in env folder manually --->
<!--- Need to install pointops manually --->
<!--- Need ScanNet downloaded --->
<!--- Figure out if huggingface 'just works' with conda env for pipeline --->
<!--- Hopefully so, otherwise need to include separate directions for setting up pipeline and testing model. --->
<!--- Do I need to include sqa3dTestScenes? --->

<!--- test --->
