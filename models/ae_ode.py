from .base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import lib.layers as layers

def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, layers.ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)

def build_model_tabular(args, dims, regularization_fns=None):

    hidden_dims = tuple([args.bivae_hidden_dim,args.bivae_hidden_dim])

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model



class UserCNFTransform(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.latent_dim = args.bivae_latent_dim
        self.mu_user_embedding=torch.randn((args.num_users,self.latent_dim))
        self.mu_item_embedding=torch.randn((args.num_items,self.latent_dim))
        self.user_embedding = torch.randn(args.num_users, self.latent_dim) * 0.01
        self.item_embedding = torch.randn(args.num_items, self.latent_dim) * 0.01

        # torch.nn.init.kaiming_uniform_(self.user_embedding, a=np.sqrt(5))

        # User Encoder
        self.encoder_user = nn.Sequential()
        self.encoder_user.add_module('fc0', nn.Linear(args.num_items, args.bivae_hidden_dim))
        self.encoder_user.add_module('ac1,', nn.Tanh())
        self.user_mu = nn.Linear(args.bivae_hidden_dim, args.bivae_latent_dim)
        self.user_var= nn.Sequential(nn.Linear(args.bivae_hidden_dim, args.bivae_latent_dim),
                                     nn.Softplus())
        self.cnf = build_model_tabular(args, self.latent_dim)

    @classmethod
    def code(cls):
        return 'ode'

    def u_encoder(self, x):
        h=self.encoder_user(x)
        return self.user_mu(h), self.user_var(h)


    def to_(self, device):
        self.user_embedding = self.user_embedding.to(device=device)
        self.item_embedding = self.item_embedding.to(device=device)
        self.mu_user_embedding=self.mu_user_embedding.to(device=device)
        self.mu_item_embedding=self.mu_item_embedding.to(device=device)
        return super(UserCNFTransform, self).to(device)

    def forward(self, x):
        mu, var=self.u_encoder(x)
        eps = torch.randn_like(mu)
        z0 = mu + eps * var.sqrt()
        zero = torch.zeros(x.shape[0], 1).to(x)
        zk, delta_logp = self.cnf(z0, zero)
        return [z0,zk,-delta_logp.view(-1), mu, var.sqrt()]


class ItemCNFTransform(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.latent_dim = args.bivae_latent_dim
        self.mu_user_embedding=torch.randn((args.num_users,self.latent_dim))
        self.mu_item_embedding=torch.randn((args.num_items,self.latent_dim))
        self.user_embedding = torch.randn(args.num_users, self.latent_dim) * 0.01
        self.item_embedding = torch.randn(args.num_items, self.latent_dim) * 0.01

        torch.nn.init.kaiming_uniform_(self.user_embedding, a=np.sqrt(5))

        # Item Encoder
        self.encoder_item = nn.Sequential()
        self.encoder_item.add_module('fc0', nn.Linear(args.num_users, args.bivae_hidden_dim))
        self.encoder_item.add_module('ac1,', nn.Tanh())
        self.item_mu = nn.Linear(args.bivae_hidden_dim, args.bivae_latent_dim)
        self.item_var = nn.Sequential(nn.Linear(args.bivae_hidden_dim, args.bivae_latent_dim),
                                      nn.Softplus())
        self.cnf = build_model_tabular(args, self.latent_dim)

    @classmethod
    def code(cls):
        return 'ode'

    def i_encoder(self, x):
        h=self.encoder_item(x)
        return self.item_mu(h), self.item_var(h)


    def to_(self, device):
        self.user_embedding = self.user_embedding.to(device=device)
        self.item_embedding = self.item_embedding.to(device=device)
        self.mu_user_embedding=self.mu_user_embedding.to(device=device)
        self.mu_item_embedding=self.mu_item_embedding.to(device=device)
        return super(ItemCNFTransform, self).to(device)


    def forward(self, x):
        mu, var=self.i_encoder(x)
        eps = torch.randn_like(mu)
        z0 = mu + eps * var.sqrt()
        zero = torch.zeros(x.shape[0], 1).to(x)
        zk, delta_logp = self.cnf(z0, zero)
        return [z0,zk,-delta_logp.view(-1), mu, var.sqrt()]




