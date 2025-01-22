import argparse
import ast
from Processor import *
import numpy as np
import random
import torch
import yaml
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_parser():
    parser = argparse.ArgumentParser(description='GATraj')
    parser.add_argument(
        '--ifvalid',default=True,type=ast.literal_eval,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    
    parser.add_argument(
        '--input_offset',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_position',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_mix',default=True,type=ast.literal_eval)
    
    parser.add_argument("--eta_min",
                        default=1e-6,
                        type=float)

    parser.add_argument(
        '--min_obs',default=8,type=int)
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of hidden state')
    parser.add_argument('--x_encoder_layers', type=int, default=3, help='Number of transformer block layers for x_encoder')
    parser.add_argument('--x_encoder_head', type=int, default=8, help='Head number of x_encoder')

    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set',default=0,type=int,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir',default='./savedata/',
        help='Directory for saving caches and models.')

    parser.add_argument(
        '--train_phase',default='train_adaptor', help='choose from [gen_memory, train_adaptor, train_diffusion')
    parser.add_argument('--key_points',default=[0,5,8,11], help='key future stamps')
    parser.add_argument('--memory_filter',default=1)
    
    parser.add_argument(
        '--phase', default='test',
        help='Set this value to \'train\' or \'test\'')
    parser.add_argument(
        '--GT',default=True)
    parser.add_argument(
        '--train_model', default='MDTraj',
        help='Your model name')
        
    parser.add_argument(
        '--load_model', default=1000,type=int,
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--load_memory', default=500,type=int,
        help="load memory")
    ######################################
    parser.add_argument(
        '--dataset',default='eth5')
    parser.add_argument(
        '--num_epochs',default=1000,type=int)
    
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')
    
    # Perprocess
    parser.add_argument(
        '--seq_length',default=20,type=int)
    parser.add_argument(
        '--obs_length',default=8,type=int)
    parser.add_argument(
        '--pred_length',default=12,type=int)
    parser.add_argument(
        '--batch_size',default=32,type=int)
    parser.add_argument(
        '--show_step',default=100,type=int)
    parser.add_argument(
        '--step_ratio',default=0.5,type=int)
    parser.add_argument(
        '--lr_step',default=20,type=int)
    parser.add_argument(
        '--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--randomRotate',default=True,type=ast.literal_eval,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred',default=10,type=int)
    parser.add_argument(
        '--learning_rate',default=1e-3,type=float)
    parser.add_argument(
        '--clip',default=1,type=int)
    return parser
def load_arg(p):
    # save arg
    if  os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # default_arg = yaml.load(f,Loader=yaml.FullLoader)
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False
def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)
def prepare_seed(rand_seed):
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
	
if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    prepare_seed(42)
    p.save_dir = p.save_base_dir+str(p.test_set)+'/' # ./savedata/1'
    p.model_dir = p.save_base_dir+str(p.test_set)+'/'+p.train_model+'/' # ./savedata/1/TFTraj/'
    args = p
    if args.test_set == 9:
        args.input_size = 3
    else:
        args.input_size = 2

    if not os.path.exists(p.save_dir):
        os.mkdir(p.save_dir)
    if not os.path.exists(p.model_dir):
        os.mkdir(p.model_dir)
    
    assert args.train_phase in ['gen_memory', 'train_adaptor', 'train_diffusion']
    processor = Processor(args)
    if args.phase=='test':
        processor.playtest()
    else:
        processor.playtrain()
