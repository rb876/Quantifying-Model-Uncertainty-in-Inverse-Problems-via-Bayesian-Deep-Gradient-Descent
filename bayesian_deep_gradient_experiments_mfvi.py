import sys
sys.path.append('./mean_field_VI')

import os
import argparse
import time
import numpy as np
import pickle
from torch.utils.data import DataLoader
from distutils.util import strtobool

# torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# forward model & dataset
import odl
from odl.contrib import torch as odl_torch
from dataset_constructor import DatasetConstructor
from forward_model import SimpleCT, ForwardModel
from dataset import DataSet

# metrics
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# utils
from utils import TrainVisualiser, next_step_update, get_stats, save_net
import layer_utils as layer_utils

def main():

    parser = argparse.ArgumentParser()
    # general & dataset & training settings
    parser.add_argument('--k_max', type=int, default=5,
                        help='Max reconstruction iterations')
    parser.add_argument('--save_figs', type = lambda x:bool(strtobool(x)), default=True,
                        help='save pics in reconstruction')
    parser.add_argument('--img_mode', type=str, default='SimpleCT',
                        help=' image-modality reconstruction: SimpleCT')
    parser.add_argument('--train_size', type=int, default=4000,
                        help='dataset size')
    parser.add_argument('--pseudo_inverse_init', type = lambda x:bool(strtobool(x)), default=True,
                        help='initialise with pseudoinverse')
    parser.add_argument('--brain', type = lambda x:bool(strtobool(x)), default=False,
                        help='test set of brain images')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--initial_lr', type=float, default=1e-3,
                        help='initial_lr')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')

    # forward models setting
    parser.add_argument('--size', type=int, default=128,
                        help='image size')
    parser.add_argument('--beam_num_angle', type=int, default=30,
                        help='number of angles / projections')
    # options
    parser.add_argument('--no_cuda', type = lambda x:bool(strtobool(x)), default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')

    args = parser.parse_args()
    layer_utils.set_gpu_mode(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.img_mode is not None:
        forward_model = ForwardModel()
        half_size = args.size / 2
        space =  odl.uniform_discr([-half_size, -half_size],
                                   [half_size, half_size],
                                   [args.size, args.size], dtype='float32')
        forward_model.space = space
        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=args.beam_num_angle)
        forward_model.geometry = geometry
        operator = odl.tomo.RayTransform(space, geometry)
        opnorm = odl.power_method_opnorm(operator)
        forward_model.operator = odl_torch.OperatorModule( (1 / opnorm) * operator )
        forward_model.adjoint = odl_torch.OperatorModule(operator.adjoint)
        pseudoinverse = odl.tomo.fbp_op(operator)
        pseudoinverse = odl_torch.OperatorModule( pseudoinverse * opnorm )
        forward_model.pseudoinverse = pseudoinverse

        geometry_specs = 'full_view_sparse_' + str(args.beam_num_angle)
        dataset_name = 'dataset' + '_' + args.img_mode + '_' + str(args.size) \
        + '_' + str(args.train_size) + '_' + geometry_specs + '_' \
        + 'brain' + '_' + str(args.brain)


    if args.img_mode == SimpleCT.__name__:
        img_mode = SimpleCT(forward_model)
        data_constructor = DatasetConstructor(img_mode, train_size=args.train_size, brain=args.brain, dataset_name=dataset_name)
        data = data_constructor.data()
    else:
        raise NotImplementedError
    dataset = DataSet(data, img_mode, args.pseudo_inverse_init)

    optim_parms = {'epochs':args.epochs, 'initial_lr':  args.initial_lr, 'batch_size': args.batch_size}
    from hybrid_model import HybridModel as NeuralLearner

    # results directory
    path = os.path.dirname(__file__)
    dir_path = os.path.join(path, 'results', args.img_mode, 'MFVI', str(args.train_size), geometry_specs, str(args.seed))
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # all config
    print('===========================\n', flush=True)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val), flush=True)
    print('===========================\n', flush=True)

    blocks_history = {'model': [], 'optimizer': []}
    arch_args = {'arch': {'up':  [ [1, 16, 3, 1, 1],  [16, 32, 3, 1, 1]],
                          'low': [ [1, 16, 3, 1, 1],  [16, 32, 3, 1, 1]],
                          'cm':  [ [64, 32, 3, 1, 1], [32, 16, 3, 1, 1]] }}

    # savings training procedures
    filename = 'train_phase'
    filepath = os.path.join(dir_path, filename)
    vis = TrainVisualiser(filepath)

    start_time = time.time()
    # looping through architecture-blocs
    for idx in range(1, args.k_max + 1):

        print('============== training block number: {} ============= \n'.format(idx), flush=True)

        train_tensor =  dataset.construct(flag='train')
        val_tensor = dataset.construct(flag='validation')

        train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_tensor, batch_size=args.val_batch_size, shuffle=True)

        model = NeuralLearner(arch_args)
        model = model.to(device)
        model_path = os.path.join(dir_path, str(idx) + '.pt')
        if os.path.exists(model_path):
            model_loaded = True
            model.load_state_dict(torch.load(model_path))
            print('idx: {} model loaded!\npath to model:\n{}'.format(idx, model_path), flush=True)
        else:
            model_loaded = False
            model.optimise(train_loader, **optim_parms)
            save_net(model, os.path.join(dir_path, str(idx) + '.pt'))
            print('idx: {} optimisation finished!'.format(idx), flush=True)

        start = time.time()
        info = next_step_update(dataset, train_tensor, model, device, flag='train')
        end = time.time()
        print('============= {} {:.4f} ============= \n'.format('training reconstruction', end-start), flush=True)
        for key in info.keys():
            print('{}: {} \n'.format(key, info[key]), flush=True)

        start = time.time()
        info = next_step_update(dataset, val_tensor, model, device, flag='validation')
        end = time.time()
        print('============= {} {:.4f} ============= \n'.format('validation reconstruction', end-start), flush=True)
        for key in info.keys():
            print('{}: {} \n'.format(key, info[key]), flush=True)

        vis.update(dataset, flag='validation')
        blocks_history['model'].append(model)

        # reconstruction
        resonstruction_dir_path = os.path.join(dir_path, str(idx))
        if model_loaded:
            resonstruction_dir_path = os.path.join(dir_path, str(idx), 're-loaded')

        if not os.path.isdir(resonstruction_dir_path):
            os.makedirs(resonstruction_dir_path)
        get_stats(dataset, blocks_history, device, resonstruction_dir_path)

    print('--- training time: %s seconds ---' % (time.time() - start_time), flush=True)
    vis.generate()

if __name__ == '__main__':
    main()
