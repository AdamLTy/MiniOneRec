import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        # SwanLab 支持
        self.use_swanlab = getattr(args, 'use_swanlab', False)
        self.global_step = 0

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 累计损失
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += rq_loss.item()

            # SwanLab 实时记录 (每个 batch)
            if self.use_swanlab:
                try:
                    import swanlab
                    swanlab.log({
                        'train/batch_total_loss': loss.item(),
                        'train/batch_recon_loss': loss_recon.item(),
                        'train/batch_quant_loss': rq_loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/global_step': self.global_step,
                    }, step=self.global_step)
                except:
                    pass

            self.global_step += 1

        return total_loss, total_recon_loss, total_quant_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        all_indices = []  # 用于分析 codebook 使用情况

        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            all_indices.append(indices)

            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample
        unique_sids = len(indices_set)

        # 分析每层 codebook 使用情况
        codebook_usage = {}
        if len(all_indices) > 0:
            all_indices = np.concatenate(all_indices, axis=0)  # [N, num_layers]
            num_layers = all_indices.shape[1]
            for layer_idx in range(num_layers):
                layer_codes = all_indices[:, layer_idx]
                unique_codes = len(np.unique(layer_codes))
                codebook_size = self.model.num_emb_list[layer_idx] if hasattr(self.model, 'num_emb_list') else 256
                usage_rate = unique_codes / codebook_size
                codebook_usage[f'layer_{layer_idx}'] = {
                    'unique_codes': unique_codes,
                    'usage_rate': usage_rate
                }

        return collision_rate, unique_sids, num_sample, codebook_usage

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, quant_loss=0):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("quantization loss", "blue") + ": %.4f" % quant_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, train_quant_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            epoch_time = training_end_time - training_start_time

            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, train_quant_loss
            )
            self.logger.info(train_loss_output)

            # SwanLab 记录 epoch 级别指标
            if self.use_swanlab:
                try:
                    import swanlab
                    swanlab.log({
                        'train/epoch_total_loss': train_loss,
                        'train/epoch_recon_loss': train_recon_loss,
                        'train/epoch_quant_loss': train_quant_loss,
                        'system/epoch_time': epoch_time,
                        'system/epoch': epoch_idx,
                    }, step=epoch_idx)
                except:
                    pass

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, unique_sids, num_samples, codebook_usage = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f, "
                    + set_color("unique_sids", "blue")
                    + ": %d/%d]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate, unique_sids, num_samples)

                self.logger.info(valid_score_output)

                # SwanLab 记录评估指标
                if self.use_swanlab:
                    try:
                        import swanlab
                        eval_metrics = {
                            'eval/collision_rate': collision_rate,
                            'eval/unique_sids': unique_sids,
                            'eval/total_samples': num_samples,
                            'eval/uniqueness_rate': unique_sids / num_samples if num_samples > 0 else 0,
                            'eval/best_collision_rate': self.best_collision_rate,
                        }

                        # 记录每层 codebook 使用率
                        for layer_name, usage_info in codebook_usage.items():
                            eval_metrics[f'codebook/{layer_name}_usage_rate'] = usage_info['usage_rate']
                            eval_metrics[f'codebook/{layer_name}_unique_codes'] = usage_info['unique_codes']

                        swanlab.log(eval_metrics, step=epoch_idx)
                    except:
                        pass

                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])



        return self.best_loss, self.best_collision_rate




