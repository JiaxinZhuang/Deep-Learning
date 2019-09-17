

## Books

1. [Deep Learning](https://github.com/janishar/mit-deep-learning-book-pdf)

2. [PRML](./Books/PRML-2006.pdf)
3. [Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

## Common Datasets for deep learning

Last update on 17 th Sep, 2019, 

###  Few-Shot Tasks

- CUB200-2011, [[Official]](http://www.vision.caltech.edu/visipedia/CUB-200.html)

- mini-ImageNet,  [[Google Drive]](https://drive.google.com/uc?export=download&confirm=qgVQ&id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk)  

- OfficeHome, [[Official]](http://hemanthdv.org/OfficeHome-Dataset) [[Google Drive]](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) 

### Image Recognition

* STL10, [[Official]](https://cs.stanford.edu/~acoates/stl10/) 
* Tiny-ImageNet,  [[Official]](https://tiny-imagenet.herokuapp.com)

### Skin Lesion

* Skin-7, [[Official]](https://challenge2018.isic-archive.com/participate/)
* SD-198,  TODO

### Chest-Ray

* Pneumonia, [[Official]](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)



## Load Dataset

You can find read file under the Dataset directory, containing main function to test data.

| # Dataset    | # Supported | # Train | # Val  | Mean ,  STD                                        |
| ------------ | ----------- | ------- | ------ | -------------------------------------------------- |
| CUB200-2011  | Y           | 5,994   | 5794   | [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]       |
| Pneumonia    | Y           | 21,345  | 5,339  | [0.4833, 0.4833, 0.4833], [0.2480, 0.2480, 0.2480] |
| Skin7        | Y           | 8,010   | 2,005  | [0.7626, 0.5453, 0.5714], [0.1404, 0.1519, 0.1685] |
| SD198        | Y           | 5,206   | 1,376  | [0.592, 0.479, 0.451], [0.265, 0.245, 0.247]       |
| TinyImageNet | Y           | 100,000 | 10,000 | [0.480, 0.448, 0.398], [0.230, 0.227, 0.226]       |

