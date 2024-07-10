from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from .loggers_for_hash import *
import torch.optim as optim
import torch
import torch.nn as nn
from distributions import log_normal_diag, log_normal_standard, log_bernoulli,TwoNomal
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
from tqdm import tqdm
import json
import os
import itertools as it
from distributions import log_normal_diag, log_normal_standard
from torch.utils.tensorboard import SummaryWriter
from abc import *
from pathlib import Path
EPS = 1e-10



class AEODETrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.user_flow=model[0].to(self.device)
        self.user_flow=self.user_flow.to_(self.device)
        self.item_flow=model[1].to(self.device)
        self.item_flow = self.item_flow.to_(self.device)

        self.u_optimizer = self._create_u_optimizer()
        self.i_optimizer = self._create_i_optimizer()
        self.train_u_loader, self.train_i_loader = self.train_loader
        # Finding or using given optimal beta
        self.__beta_ukld=0.0
        self.__beta_ikld=0.0
        self.__beta_user = 0.0
        self.__beta_item = 0.0
        self.best_ukld_beta=self.__beta_ukld
        self.best_ikld_beta=self.__beta_ikld
        self.best_user_beta = self.__beta_user
        self.best_item_beta = self.__beta_item
        self.finding_best_beta = args.find_best_beta
        self.anneal_amount = 1.0 / args.total_anneal_steps
        self.anneal_kld_amount=0.2/args.total_anneal_kld_steps
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.add_extra_loggers()

        self.log_period_as_iter = args.log_period_as_iter  # 设置为12800
        if self.finding_best_beta:
            self.current_best_metric = 0.0
            self.anneal_cap = 1.0
            self.anneal_kld_cap=0.2
        else:
            self.anneal_cap = args.anneal_cap
            self.anneal_kld_cap = args.anneal_kld_cap
        if args.enable_lr_schedule:
            self.lr_u_scheduler = optim.lr_scheduler.StepLR(self.u_optimizer, step_size=args.decay_step, gamma=args.gamma)
            self.lr_i_scheduler = optim.lr_scheduler.StepLR(self.i_optimizer, step_size=args.decay_step, gamma=args.gamma)
        self.target_dist = TwoNomal((torch.ones(self.args.bivae_latent_dim)).to(self.device),
                                    (torch.ones(self.args.bivae_latent_dim) * (-1.0)).to(self.device), \
                                    (torch.ones(self.args.bivae_latent_dim) * args.relaxed_degree).to(self.device),
                                    (torch.ones(self.args.bivae_latent_dim) * args.relaxed_degree).to(self.device))

    @classmethod
    def code(cls):
        return 'ode'
    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Precision@%d' % k, graph_name='Precision@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='F-measure@%d' % k, graph_name='F-measure@%d' % k,
                                   group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers


    def add_extra_loggers(self):
        cur_user_beta_logger = MetricGraphPrinter(self.writer, key='cur_user_beta', graph_name='BetaUser', group_name='Train')
        cur_item_beta_logger = MetricGraphPrinter(self.writer, key='cur_item_beta', graph_name='BetaItem', group_name='Train')
        cur_ukld_beta_logger = MetricGraphPrinter(self.writer, key='cur_ukld_beta', graph_name='BetaUkld',
                                                  group_name='Train')
        cur_ikld_beta_logger = MetricGraphPrinter(self.writer, key='cur_ikld_beta', graph_name='BetaIkld',
                                                  group_name='Train')
        self.train_loggers.append(cur_user_beta_logger)
        self.train_loggers.append(cur_item_beta_logger)
        self.train_loggers.append(cur_ukld_beta_logger)
        self.train_loggers.append(cur_ikld_beta_logger)

        if self.args.find_best_beta:
            best_user_beta_logger = MetricGraphPrinter(self.writer, key='best_user_beta', graph_name='Best_user_beta', group_name='Validation')
            self.val_loggers.append(best_user_beta_logger)
            best_item_beta_logger = MetricGraphPrinter(self.writer, key='best_item_beta', graph_name='Best_item_beta',
                                                       group_name='Validation')
            self.val_loggers.append(best_item_beta_logger)
            best_ukld_beta_logger = MetricGraphPrinter(self.writer, key='best_ukld_beta', graph_name='Best_ukld_beta',
                                                       group_name='Validation')
            self.val_loggers.append(best_ukld_beta_logger)
            best_ikld_beta_logger = MetricGraphPrinter(self.writer, key='best_ikld_beta', graph_name='Best_ikld_beta',
                                                       group_name='Validation')
            self.val_loggers.append(best_ikld_beta_logger)

    def log_extra_train_info(self, log_data):
        log_data.update({'cur_user_beta': self.__beta_user})
        log_data.update({'cur_item_beta': self.__beta_item})
        log_data.update({'cur_ukld_beta': self.__beta_ukld})
        log_data.update({'cur_ikld_beta': self.__beta_ikld})
    
    def log_extra_val_info(self, log_data):
        if self.finding_best_beta:
            log_data.update({'best_user_beta': self.best_user_beta})
            log_data.update({'best_item_beta': self.best_item_beta})
            log_data.update({'best_ukld_beta': self.best_ukld_beta})
            log_data.update({'best_ikld_beta': self.best_ikld_beta})

    def _create_u_optimizer(self):
        user_params = it.chain(
            self.user_flow.encoder_user.parameters(),
            self.user_flow.user_mu.parameters(),
            self.user_flow.user_var.parameters(),
            self.user_flow.cnf.parameters()
        )
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(user_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(user_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_i_optimizer(self):
        item_params = it.chain(
            self.item_flow.encoder_item.parameters(),
            self.item_flow.item_mu.parameters(),
            self.item_flow.item_var.parameters(),
            self.item_flow.cnf.parameters()
        )
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(item_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(item_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    @property
    def beta_user(self):
        if self.user_flow.training:
            self.__beta_user = min(self.__beta_user + self.anneal_amount, self.anneal_cap)
        return self.__beta_user

    @property
    def beta_ukld(self):
        if self.user_flow.training:
            self.__beta_ukld = min(self.__beta_ukld + self.anneal_kld_amount, self.anneal_kld_cap)
        return self.__beta_ukld

    @property
    def beta_item(self):
        if self.item_flow.training:
            self.__beta_item = min(self.__beta_item + self.anneal_amount, self.anneal_cap)
        return self.__beta_item

    @property
    def beta_ikld(self):
        if self.item_flow.training:
            self.__beta_ikld = min(self.__beta_ikld + self.anneal_kld_amount, self.anneal_kld_cap)
        return self.__beta_ikld

    def calculate_loss(self, x,x_, mu, std, base_log_prob,sum_log_abs_det_jacobians,p_log_prob, user=True):
        choices = {
            "mult": x * torch.log(x_ + EPS),
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }
        ll=choices[self.ll_choice]
        ll = torch.sum(ll, dim=1)
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld=torch.sum(kld, dim=1)
        #flow_loss
        fl=base_log_prob-sum_log_abs_det_jacobians-p_log_prob
        if user:
            return torch.mean(-ll+self.beta_user*fl+self.beta_ukld*kld)
        else:
            return torch.mean(-ll + self.beta_item * fl+self.beta_ikld*kld)



    def train(self):
        u_accum_iter = 0
        i_accum_iter = 0

        for epoch in range(self.num_epochs):
            i_accum_iter = self.train_one_epoch_for_item(epoch, i_accum_iter)
            u_accum_iter = self.train_one_epoch_for_user(epoch, u_accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_user_dict()),
        })
        self.writer.close()

    def renew_mu(self):
        #user_side
        tqdm_u_dataloader = tqdm(self.train_u_loader)
        for batch_idx, batch in enumerate(tqdm_u_dataloader):
            u_indices = list(
                range(batch_idx * self.args.train_batch_size, (batch_idx * self.args.train_batch_size + len(batch))))
            batch = batch.to(self.device)
            user_embedding, _, u_mu, _ = self.model(batch, user=True, item_embedding=self.model.item_embedding)
            self.model.mu_user_embedding.data[u_indices] = u_mu.data

        #item_side
        tqdm_i_dataloader = tqdm(self.train_i_loader)
        for batch_idx, batch in enumerate(tqdm_i_dataloader):
            i_indices = list(range(batch_idx * self.args.train_batch_size,
                                   (batch_idx * self.args.train_batch_size + len(batch))))
            batch = batch.to(self.device)
            item_embedding, _, i_mu, _ = self.model(batch, user=False, user_embedding=self.model.user_embedding)
            self.model.mu_item_embedding.data[i_indices] = i_mu.data



    def train_one_epoch_for_user(self, epoch, u_accum_iter):
        self.user_flow.train()
        if self.args.enable_lr_schedule:
            self.lr_u_scheduler.step()
        average_meter_set = AverageMeterSet()
        tqdm_u_dataloader = tqdm(self.train_u_loader)
        # user_side
        for batch_idx, batch in enumerate(tqdm_u_dataloader):
            batch_size = batch[0].size(0)
            u_indices = list(
                range(batch_idx * self.args.train_batch_size, (batch_idx * self.args.train_batch_size + len(batch))))
            batch = batch.to(self.device)
            z0, user_embedding, ldj, mu, std=self.user_flow(batch)
            base_user_log_prob=log_normal_diag(z0, mean=mu, log_var=std.pow(2).log(), dim=1)
            # user_p_log_prob=log_normal_standard(user_embedding, dim=1)
            user_p_log_prob = self.target_dist.log_doubledensity(user_embedding).sum(-1)
            recon_x=torch.sigmoid(user_embedding.mm(self.user_flow.item_embedding.t())).to(self.device)
            u_loss=self.calculate_loss(batch, recon_x, mu, std, base_user_log_prob,ldj,user_p_log_prob,user=True)
            self.u_optimizer.zero_grad()
            u_loss.backward()
            self.u_optimizer.step()
            _,user_embedding, _, u_mu, _ = self.user_flow(batch)
            self.user_flow.user_embedding.data[u_indices] = user_embedding.data
            self.user_flow.mu_user_embedding.data[u_indices]=u_mu.data
            self.item_flow.user_embedding.data[u_indices] = user_embedding.data
            self.item_flow.mu_user_embedding.data[u_indices]=u_mu.data

            average_meter_set.update('loss', u_loss.item())
            tqdm_u_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            u_accum_iter += batch_size

            if self._needs_to_log(u_accum_iter):
                tqdm_u_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_user_dict()),
                    'epoch': epoch+1,
                    'accum_iter': u_accum_iter,
                    'user_embedding': self.user_flow.user_embedding,
                    'item_embedding': self.user_flow.item_embedding
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)
        return u_accum_iter

    def train_one_epoch_for_item(self, epoch, i_accum_iter):
        self.item_flow.train()
        if self.args.enable_lr_schedule:
            self.lr_i_scheduler.step()
        average_meter_set = AverageMeterSet()
        tqdm_i_dataloader = tqdm(self.train_i_loader)
        # item_side
        for batch_idx, batch in enumerate(tqdm_i_dataloader):
            batch_size = batch[0].size(0)
            i_indices = list(range(batch_idx * self.args.train_batch_size,
                                   (batch_idx * self.args.train_batch_size + len(batch))))
            batch=batch.to(self.device)
            z0, item_embedding, ldj, mu, std = self.item_flow(batch)
            base_item_log_prob=log_normal_diag(z0,mean=mu,log_var=std.pow(2).log(),dim=1)
            # item_p_log_prob = log_normal_standard(item_embedding,dim=1)
            item_p_log_prob = self.target_dist.log_doubledensity(item_embedding).sum(-1)

            self.i_optimizer.zero_grad()
            recon_x=torch.sigmoid(item_embedding.mm(self.item_flow.user_embedding.t())).to(self.device)
            i_loss = self.calculate_loss(batch, recon_x, mu, std,base_item_log_prob,ldj,item_p_log_prob,user=False)
            i_loss.backward()
            self.i_optimizer.step()
            _,item_embedding,_,i_mu,_=self.item_flow(batch)
            self.item_flow.item_embedding.data[i_indices] = item_embedding.data
            self.item_flow.mu_item_embedding.data[i_indices]=i_mu.data
            self.user_flow.item_embedding.data[i_indices] = item_embedding.data
            self.user_flow.mu_item_embedding.data[i_indices]=i_mu.data
            average_meter_set.update('loss', i_loss.item())
            tqdm_i_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            i_accum_iter += batch_size

            if self._needs_to_log(i_accum_iter):
                tqdm_i_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_item_dict()),
                    'epoch': epoch+1,
                    'accum_iter': i_accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return i_accum_iter


    def _create_state_user_dict(self):
        return {
            STATE_DICT_KEY:self.user_flow.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.u_optimizer.state_dict()
        }

    def _create_state_item_dict(self):
        return {
            STATE_DICT_KEY: self.item_flow.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.i_optimizer.state_dict()
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0