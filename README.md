## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)


## Installation

```bash
git clone https://github.com/Quinnan-Gill/pytorch-fcn
cd pytorch-fcn
pip install .
```

If there are issues installing with setup tools run:
```bash
pip install setuptools==64
pip install .
```

## Training

### Download Dataset

```bash
cd train/
./download_dataset.sh
```

### Train Siren FCN model

In `pytorch-fcn/train/` is the file `train_sire.py` the usage is:
```bash
usage: train_siren.py [-h] -g GPU [--resume RESUME] [--fcn {fcn32s,fcn16s,fcn8s}] [--height HEIGHT] [--width WIDTH]
                      [--siren_plus] [--filters FILTERS] [--epochs EPOCHS] [--lr LR] [--weight-decay WEIGHT_DECAY]
                      [--momentum MOMENTUM] [-o OUT]

options:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     gpu id (default: None)
  --resume RESUME       checkpoint path (default: None)
  --fcn {fcn32s,fcn16s,fcn8s}
  --height HEIGHT
  --width WIDTH
  --siren_plus
  --filters FILTERS     comma seperated list of all the ReLUs to convert to SIRENs (default: None)
  --epochs EPOCHS       # of epochs to train (default: 5)
  --lr LR               learning rate (default: 1e-12)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 0.0005)
  --momentum MOMENTUM   momentum (default: 0.99)
  -o OUT, --out OUT
```

Train for:
* FCN32s
```bash 
cd train/
python train_sire.py -g 0 --fcn fcn32s
```

* FCN16s
```bash 
cd train/
python train_sire.py -g 0 --fcn fcn16s
```

* FCN8s
```bash 
cd train/
python train_sire.py -g 0 --fcn fcn8s
```

* For replacing the first ReLUs with SIREN
```bash 
cd train/
python train_sire.py -g 0 --filters "relu1_1,relu1_2"
```

* For replaceing the last two ReLUs with SIREN
```bash 
cd train/
python train_sire.py -g 0 --filters "relu6,relu7"
```


## Citation

This project was extended from:

```bibtex
@misc{pytorch-fcn2017,
  author =       {Ketaro Wada},
  title =        {{pytorch-fcn: PyTorch Implementation of Fully Convolutional Networks}},
  howpublished = {\url{https://github.com/wkentaro/pytorch-fcn}},
  year =         {2017}
}
```
```bibtex
@misc{sitzmann2020implicit,
    title   = {Implicit Neural Representations with Periodic Activation Functions},
    author  = {Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year    = {2020},
    eprint  = {2006.09661},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{mehta2021modulated,
    title   = {Modulated Periodic Activations for Generalizable Local Functional Representations}, 
    author  = {Ishit Mehta and Michaël Gharbi and Connelly Barnes and Eli Shechtman and Ravi Ramamoorthi and Manmohan Chandraker},
    year    = {2021},
    eprint  = {2104.03960},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
