# Enhancing Few-Shot 3D Point Cloud Classification with Soft Interaction Module and Self-Attention Residual Feedforward

## Environment Requirements
Before you get started, ensure you have the following dependencies installed:
- Python3, PyTorch, json, h5py

## Getting Started
### Dataset Download and Split for ModelNet40
1. Navigate to the `../dataset/` directory. (Please note that the 'dataset' folder should be at the same level as the project folder. If it doesn't exist, create the directory.)
2. Download the dataset by clicking on this [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) link, and unzip it into the `../dataset/modelnet40_normal_resampled` folder.
3. Dataset Split: You can split the dataset manually or use our predefined split by copying the './data/base.json' and './data/novel.json' files to `../dataset/modelnet40`.

### Data Preprocessing (ModelNet40, ShapeNetCore, ScanObjectNN, etc.)
Set all the file paths according to config.py and refer to the DataProcess folder to organize and build JSON files for the datasets.
## Train Your Model
To start training your model, run the following command:
```shell
python ./train.py --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]
```
For example, you can run `python ./train.py --model pointnet --method protonet --weightdecay 0.01`. If you wish to implement different methods with varying hyperparameters, please refer to 'io_utils.py'.

## Save Features
After training, saving features first is advisable, as this can significantly speed up repeated experiments during testing. For instance, you can run the following command:
```shell
python ./save_features.py --model pointnet --method protonet
```
If you would like more details, you can consult 'io_utils.py'. You can locate your stored feature files in the './features' directory.

## Test Your Model
To test your trained model, run a command like this:
```shell
python ./test.py --model pointnet --method protonet
```
For further details, refer to 'io_utils.py'.

**Reminder:** You can run the test command directly without executing 'train.py' and 'save_features.py' since we have provided a pre-trained model for PointNet+ProtoNet. You can check your experiment results in `./record/results.txt`.

## Download Pre-trained Weights/Checkpoints
[Download Features and Checkpoints](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EnGjUXZ07hZLrS7nT8YCfmMBxRRxa6Yk9i_9lfqZJTUP-w?e=JfTE6P)



## References
This code is adapted from:
https://github.com/PeiZhou26/Few-Shot-3D-Point-Cloud-Classification.

We thank the author for providing code and guidance.
