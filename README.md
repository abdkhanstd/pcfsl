# Enhancing Few-Shot 3D Point Cloud Classification with Soft Interaction Module and Self-Attention Residual Feedforward


## Environment Requirements
Before you get started, ensure you have the following dependencies installed:
- Python3
- PyTorch
- json
- h5py
- tensorboard

## Getting Started

### Data Preprocessing
Before running experiments, it's essential to preprocess the data to enhance training speed. Follow these steps:
1. Execute `python ./data/dataset.py`. Please note that this process may take over two hours to complete for the entire dataset, as the farthest point sample method is time-consuming.

**WARNING:** The file path may vary on different systems. If you encounter a "file not found" error, modify the 'pwd' variable in the main function of './data/dataset.py' accordingly.

## Train Your Model
To start training your model, run the following command:
```shell
python ./train.py --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]
```
For example, you can run `python ./train.py --model pointnet --method protonet --weightdecay 0.01`. If you wish to implement different methods with varying hyperparameters, please refer to 'io_utils.py'.

## Save Features
After training, it's advisable to save features first, as this can significantly speed up repeated experiments during testing. For instance, you can run the following command:
```shell
python ./save_features.py --model pointnet --method protonet
```
For more details, consult 'io_utils.py'. You can locate your stored feature files in the './features' directory.

## Test Your Model
To test your trained model, run a command like this:
```shell
python ./test.py --model pointnet --method protonet
```
For further details, refer to 'io_utils.py'.

**Reminder:** You can run the test command directly without executing 'train.py' and 'save_features.py' since we have provided a pre-trained model for PointNet+ProtoNet. You can check your experiment results in `./record/results.txt`.

## References
This code is adapted from:
https://github.com/PeiZhou26/Few-Shot-3D-Point-Cloud-Classification/edit/main/README.md
We thank the author for providing code and guidance.
