# Specifying Object Attributes and Relations in Interactive Scene Generation
A PyTorch implementation of [Specifying Object Attributes and Relations in Interactive Scene Generation](https://arxiv.org/abs/)
<p align="center"><img src='images/scene_generation.png' width='650px'></p>

## Paper
[Specifying Object Attributes and Relations in Interactive Scene Generation](https://arxiv.org/abs/)
<br/>
[Oron Ashual](https://)<sup>1</sup>, [Lior Wolf](https://www.cs.tau.ac.il/~wolf/)<sup>1,2</sup><br/>
<sup>1 </sup> Tel-Aviv University, <sup>2 </sup> Facebook AI Research <br/>
IEEE International Conference on Computer Vision ([ICCV](http://iccv2019.thecvf.com/)), 2019, (<b>Oral</b>)

## Network Architechture
<p align='center'><img src='images/arch.png' width='1000px'></p>

## Youtube
[![Specifying Object Attributes and Relations in Interactive Scene Generation](https://img.youtube.com/vi/V2v0qEPsjr0/0.jpg)](https://www.youtube.com/watch?v=V2v0qEPsjr0 "Specifying Object Attributes and Relations in Interactive Scene Generation")

## Usage

### 1. Creating virtual environment (optional)
All code was developed and tested on Ubuntu 18.04 with Python 3.6 (Anaconda) and PyTorch 1.0.

```bash
$ conda create -n scene_generation python=3.6
$ conda activate scene_generation
```

### 2. Install COCO API
```bash
$ cd ~
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI/
$ python setup.py install
$ cd ..
```

### 3. Cloning the repository
```bash
$ git clone git@github.com:ashual/scene_generation.git
$ cd scene_generation
```

### 4. Installing dependencies
```bash
$ conda install -r requirements.txt
```

### 5. Training
```bash
$ python train.py
```

### 6. Downloading trained models
TBD

### 7. GUI
The GUI was built as POC. For using it run:
```bash
python scripts/gui/simple-server.py --checkpoint YOUR_MODEL_CHECKPOINT
```

## Citation

If you find this code useful in your research then please cite
```
@inproceedings{ashual2019scenegeneration,
  title={Specifying Object Attributes and Relations in Interactive Scene Generation},
  author={Ashual, Oron and Wolf, Lior},
  booktitle={ICCV},
  year={2019}
}
```

## Acknowledgement 
Our project borrows some source files from [sg2im](https://github.com/google/sg2im). We thank the authors.