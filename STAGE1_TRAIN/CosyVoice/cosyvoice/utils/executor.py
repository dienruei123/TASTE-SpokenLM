# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import torch
import time
import datetime
import tqdm
import torch.distributed as dist

from collections import defaultdict
from contextlib import nullcontext
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))
        self.cv_best_score = 0.0

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        # model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        # with model_context():
        for batch_idx, batch_dict in enumerate(tqdm.tqdm(train_data_loader, position=self.rank, desc=f"[Rank {self.rank}] Training...")):
            info_dict["tag"] = "TRAIN"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            
            if cosyvoice_join(group_join, info_dict):
                break

            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext

            with context():
                info_dict = batch_forward(model, batch_dict, info_dict)
                info_dict = batch_backward(model, info_dict)

            info_dict = update_parameter_and_lr(model, optimizer, scheduler, info_dict)
            log_per_step(writer, info_dict)
            # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
            if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and (batch_idx + 1) % info_dict["accum_grad"] == 0:
                # dist.barrier(group=group_join)
                self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False, group_join=group_join)
                logging.info(f"[Rank {self.rank}] waiting after CV...")
                dist.barrier()
                logging.info(f"[Rank {self.rank}] FINISHED waiting after CV, will continue training...")
                model.train()
            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1
        
        logging.info(f"[Rank {self.rank}] has finished epoch {self.epoch} training. Break other workers'")
        dist.barrier()
        logging.info(f"[Rank {self.rank}] all ranks are done. Conduct CV after epoch now.")
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True, group_join=group_join)
        dist.barrier()

    # @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True, group_join=None):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        total_num = 0
        arrows_coverage_dict = defaultdict(lambda: 0)
        with torch.inference_mode():
            for batch_idx, batch_dict in enumerate(tqdm.tqdm(cv_data_loader, position=self.rank, desc=f"[Rank {self.rank}] Validating...")):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx

                num_utts = len(batch_dict["utts"])
                if num_utts > 0:
                    _arrow_name = batch_dict["utts"][0].split("__")[0]
                    arrows_coverage_dict[_arrow_name] += 1
                total_num_utts += num_utts

                info_dict = batch_forward(model, batch_dict, info_dict)

                if 'len' in info_dict['loss_dict']:
                    num = info_dict['loss_dict']['len']
                    total_num += num
                    for k, v in info_dict['loss_dict'].items():
                        if k not in total_loss_dict:
                            total_loss_dict[k] = 0.0
                        if k == 'len':
                            continue
                        total_loss_dict[k] += v.item() * num.item()
                else:
                    for k, v in info_dict['loss_dict'].items():
                        if k not in total_loss_dict:
                            total_loss_dict[k] = 0.0
                        total_loss_dict[k] += v.item() * num_utts
                log_per_step(None, info_dict)
        
        denom = torch.tensor(max(1, total_num_utts, total_num.item()), device=self.device).clone().to_dense()
        logging.info(f"[Rank {self.rank}] finished partial cv. waiting other process to gather metrics.")
        for k, v in arrows_coverage_dict.items():
            logging.info(f"[Rank {self.rank}] has {v} items in {k}.")
        logging.info(f"[Rank {self.rank}] total_num_utts={total_num_utts}, total_num_batches={batch_idx}, average CV batch size is {(total_num_utts / batch_idx):.2f}")
        logging.debug(f"[Rank {self.rank}] prev denominator={denom}, dtype={denom.dtype}")
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)
        logging.debug(f"[Rank {self.rank}] curr denominator={denom}, dtype={denom.dtype}")
        for k in total_loss_dict.keys():
            if type(total_loss_dict[k]) != type(denom):
                total_loss_dict[k] = torch.tensor(total_loss_dict[k], device=denom.device).to_dense()
            else:
                total_loss_dict[k] = total_loss_dict[k].to(denom.device).to_dense()
            logging.debug(f"[Rank {self.rank}] prev numerator of {k}={total_loss_dict[k]}, dtype={total_loss_dict[k].dtype}")
            dist.all_reduce(total_loss_dict[k], op=dist.ReduceOp.SUM)
            logging.debug(f"[Rank {self.rank}] curr numerator of {k}={total_loss_dict[k]}, dtype={total_loss_dict[k].dtype}")
            total_loss_dict[k] = total_loss_dict[k].item() / denom.item()
        info_dict['loss_dict'] = total_loss_dict
        cur_cv_score = total_loss_dict.get('acc', 0.0)
        if self.rank == 0:
            if cur_cv_score > self.cv_best_score:
                self.cv_best_score = cur_cv_score
                logging.info(f"[Rank {self.rank}] CV New best score: {self.cv_best_score}, will save new best ckpt.")
                best_model_name = 'checkpoint_best'
                save_model(model, best_model_name, info_dict)
            log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
        logging.info(f"[Rank {self.rank}] Finished CV.")
        return
