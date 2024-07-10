from models import model_factory
from dataloaders import dataloader_factory
from trainers import TRAINERS

from trainers import trainer_factory
from utils import *
import argparse
from datasets import DATASETS
from dataloaders import DATALOADERS
import lib.layers.odefunc as odefunc

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='RecPlay')

    ################
    # Top Level
    ################
    parser.add_argument('--mode', type=str, default='train', choices=['train'])

    ################
    # Trainer
    ################
    parser.add_argument('--trainer_code', type=str, default='ode', choices=TRAINERS.keys())
    # device #
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--device_idx', type=str, default='0')
    # optimizer #
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
    # lr scheduler #
    parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
    # epochs #
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    # logger #
    parser.add_argument('--log_period_as_iter', type=int, default=12800)
    # evaluation #
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
    # Finding optimal beta for VAE #
    parser.add_argument('--find_best_beta', type=bool, default=False,
                        help='If set True, the trainer will anneal beta all the way up to 1.0 and find the best beta')
    parser.add_argument('--total_anneal_steps', type=int, default=3000, help='The step number when beta reaches 1.0')
    parser.add_argument('--total_anneal_kld_steps', type=int, default=6500, help='The step number when beta reaches 1.0')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='Upper limit of increasing beta. Set this as the best beta found')
    parser.add_argument('--anneal_kld_cap', type=float, default=0.2,
                        help='Upper limit of increasing beta. Set this as the best beta found')
    ################
    # Model
    ################
    parser.add_argument('--model_code', type=str, default='ode')
    parser.add_argument('--model_init_seed', type=int, default=None)
    parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
    parser.add_argument('--num_users', type=int, default=None, help='Number of total items')
    parser.add_argument('--dual', type=bool, default=False,
                        help='If set True, the trainer architecture will be bivae')

    # VAE #
    parser.add_argument('--vae_num_hidden', type=int, default=0, help='Number of hidden layers in VAE')
    parser.add_argument('--vae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in VAE')
    parser.add_argument('--vae_latent_dim', type=int, default=200,
                        help="Dimension of latent vector in VAE (K in paper)")
    parser.add_argument('--vae_dropout', type=float, default=0.5, help='Probability of input dropout in VAE')

    # BiVAE #
    parser.add_argument('--bivae_num_hidden', type=int, default=0, help='Number of hidden layers in VAE')
    parser.add_argument('--bivae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in VAE')
    parser.add_argument('--bivae_latent_dim', type=int, default=200,
                        help="Dimension of latent vector in VAE (K in paper)")
    parser.add_argument('--bivae_dropout', type=float, default=0.5, help='Probability of input dropout in VAE')

    # Planar #
    parser.add_argument('--flow_length',type=int, default=4,help='Number of flow layers in Planar')

    parser.add_argument('--relaxed_degree', type=float, default=0.5, help='sigma of target distribution (variance=sigma*sigma)')

    #CNF#
    parser.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat", "concat_v2", "squash" "concatsquash", "concatcoord", "hyper", "blend"]
    )
    parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
    parser.add_argument('--time_length', type=float, default=1)
    parser.add_argument('--train_T', type=eval, default=False)
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)
    parser.add_argument('--bn_lag', type=float, default=0)
    ################
    # Experiment
    ################
    parser.add_argument('--experiment_dir', type=str, default='experiments')
    parser.add_argument('--experiment_description', type=str, default='test')

    ################
    # Dataset
    ################
    parser.add_argument('--dataset_split_seed', type=int, default=42)
    parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
    parser.add_argument('--min_rating', type=int, default=0, help='Only keep ratings greater than equal to this value')
    parser.add_argument('--min_uc', type=int, default=0, help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
    parser.add_argument('--split', type=list, default=[0.5,0.25,0.25], help='The ratio of splitting the datasets')

    ################
    # Dataloader
    ################
    parser.add_argument('--dataloader_code', type=str, default='ae', choices=DATALOADERS.keys())
    parser.add_argument('--train_batch_size', type=int, default=64)

    args = parser.parse_args()
    args.mode = 'train'

    args.dataset_code =input('Input ml-100k, ml-1m,ml-20m,ml-20m, Amazon_Office, Amazon_Toys, Douban_Book, Douban_Movie, Douban_Music:  ')
    args.min_rating = 0 if args.dataset_code == 'ml-20m' else 0
    args.min_uc = 20
    args.min_sc = 20
    args.split = [0.5,0.25,0.25]
    args.dataset_split_seed = 42

    args.dataloader_code = 'ae'
    args.dual=True
    batch = 512 if args.dataset_code == 'ml-20m'or args.dataset_code == 'ml-10m' else 64
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    args.trainer_code = 'ode'
    args.device = 'cuda'
    args.device_idx = '1'
    args.optimizer = 'adam'
    args.lr = 1e-3
    args.ll_choice = 'pois'
    args.enable_lr_schedule = True
    args.weight_decay = 0.01
    args.num_epochs =100
    args.metric_ks = [5, 10, 20, 30,50, 100]
    args.best_metric = 'F-measure@20'
    args.find_best_beta = True
    args.anneal_cap = 0.342

    args.total_anneal_steps = 20000 if args.dataset_code == 'ml-20m'else 3000


    args.model_code = 'ode'
    args.flow_length=14
    args.model_init_seed = 2
    args.bivae_num_hidden = 1
    args.bivae_hidden_dim = 100
    args.bivae_latent_dim = 64
    args.bivae_dropout = 0.1
    args.relaxed_degree=0.2
    args.momentum=0.9
    args.experiment_description = 'hash_ode_' + args.dataset_code + '_' + str(args.bivae_latent_dim)
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')