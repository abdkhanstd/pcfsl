import os
import glob
import argparse
import numpy as np
import backbone

# Define the model dictionary
model_dict = dict(
    pointnet=backbone.pointnet,
    pointnet2=backbone.pointnet2,
    dgcnn=backbone.dgcnn
)

# Define the function to parse command-line arguments
def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % script)
    
    # Common arguments
    parser.add_argument('--dataset', default='ShapeNetCore', help='ModelNet40, ShapeNetCore, ScanObjectL1, ScanObjectL2, ScanObjectL3')
    parser.add_argument('--model', default='pointnet', help='pointnet/pointnet2/dgcnn')  # 3 feature networks are used in the paper
    parser.add_argument('--opti', default='Adam', help='Adam, SGD')
    parser.add_argument('--weightdecay', default=0, type=float, help='0/0.01/0.001')
    parser.add_argument('--method', default='protonet', help='baseline/baseline++/protonet/relationnet_softmax')  # 5 few shot methods
    parser.add_argument('--train_n_way', default=5, type=int, help='class num to classify for training')  # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing (validation)')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=10, type=int, help='number of labeled data in each class, same as n_support')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug', default=True, type=bool, help='perform data augmentation or not during training')  # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--classbalance', default=True, type=bool, help='use two kinds of dataloader')
    parser.add_argument('--num_episode', default=400, type=int, help='Number of episodes per epoch i.e., 400, 100')
    parser.add_argument('--iter_num_test', default=100, type=int, help='Number of test iterations for test loop during training i.e., 100, 600, 700')
    parser.add_argument('--embedding_length', default=512, type=int, help='1024, 512, 256')
    parser.add_argument('--patience', default=800, type=int, help='Wait for a certain number of epochs if there is no improvement in validation accuracy. A number bigger than stop_epoch means no early stopping')

    if script == 'train':
        # Training-specific arguments
        parser.add_argument('--num_classes', default=100, type=int, help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in the base class
        parser.add_argument('--save_freq', default=900, type=int, help='Bigger number means to save the best only. Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int, help='Total number of epochs')  # for meta-learning methods, each epoch contains --num_episode episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', action='store_true', help='continue from the previous trained model with the largest epoch')
        parser.add_argument('--warmup', action='store_true', help='continue from baseline, neglected if resume is true')  # never used in the paper
    elif script == 'save_features':
        # Save features script arguments
        parser.add_argument('--split', default='novel', help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='save features from the model trained in x epochs, use the best model if x is -1')
    elif script == 'test':
        # Test script arguments
        parser.add_argument('--split', default='novel', help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='saved features from the model trained in x epochs, use the best model if x is -1')
        parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    elif script == 'visualize':
        # Visualize script arguments
        parser.add_argument('--split', default='novel', help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='saved features from the model trained in x epochs, use the best model if x is -1')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()

# Define utility functions
def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
