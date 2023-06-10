# unsupervised-image-segmentation
In this study, it is aimed to establish a model that uses the advantages of U-Net to extract detailed features and a simple CNN architecture that will initially guide the model through extracting coarse features.

## Installation
- Install CUDA
- Install PyTorch
- Install dependencies
```console
pip install -r requirements.txt
```

## Run
```console
usage: main.py [options] [target]
options:
        [--nEpochs E] [--nChannel N] [--lr LR] [--maxIter T] [--minLabels minL]
        [--nConv M] [--trainInput] [--testInpıt] [--model NM]
        [--combine CM] [--equalize EQ]

Train the unsupervised and/or UNET model on target directory/image 

optional arguments:
  --nEpochs E,      Number of epochs
  --lr LR,          Learning rate
  --nChannel N,     Number of channels
  --maxIter T,      maksimum iteration
  --minLabels minL, minimum number of labels 
  --nConv M,        number of convolutional layers
  --trainInput,     train data <image>
  --testInput,      test data <image> or </<path>/directory/>
  --model NM,       the name of the model to be selected
  --combine CM,     combine models True or False
  --equalize EQ,    apply Histogram Equalization or not
```
Example usage:
```console
python3 src/main.py --trainInput '/<path>/src/demo_test_data/2092.jpg' --testInput '/<path>/src/demo_test_data/2092.jpg' --resultPath /<path>/demo_results/ --combine True --model mynet --nChannel 50 --lr 0.1 --maxIter 80
```
