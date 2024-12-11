import math
import os
import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
import torch.nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils.pipeline_utils import get_layer_wise_, get_layer_wise_3d, get_layer_wise_exp, get_layer_wise_sdm
from utils.utils import get_logger, is_logging_process, modify_learning_rate, to_numpy, unfreeze_dropout
from utils.writer import Writer
from utils.loss import compute_log_likelihood
from utils.loss import MSELoss, L1Loss
import json

class Base_Model_handler:

    def __init__(self, cfg, net_arch, loss_f, writer: Writer, rank=0):
        self.cfg = cfg
        self.device = self.cfg.device
        self.net = net_arch.to(self.device)
        self.writer = writer
        self.rank = rank
        self.last_val_loss = 10e5  # just a random large number
        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net = DDP(self.net,
                           device_ids=[self.rank],
                           find_unused_parameters=True)
        self.step = 0
        self.epoch = -1
        self._logger = get_logger(cfg, os.path.basename(__file__))
        self.im_counter = 0  # counter for saving images
        # init optimizer
        optimizer_mode = self.cfg.train.optimizer.mode
        scheduler_mode = self.cfg.train.scheduler.mode
        if optimizer_mode == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                **(self.cfg.train.optimizer[optimizer_mode]))
        if scheduler_mode == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **(self.cfg.train.scheduler[scheduler_mode]))
        else:
            raise Exception(
                f"optimizer or scheduler not supported {optimizer_mode,scheduler_mode}"
            )

        self.mse = MSELoss(cfg)
        self.l1 = L1Loss(cfg)
        self.loss_f = loss_f
        self.log = OmegaConf.create()
        self.layers = 'ILM', 'RPEDC', 'BM'
     
    def train_model(self, train_loader):
        logger = get_logger(self.cfg,
                            os.path.basename(__file__),
                            disable_console=True)
        #Track Oversampling
        ped, nonPed = 0, 0

        for model_input, model_target, mask, label in tqdm(
                train_loader, leave=False, desc="Training/Batch:",
                unit='batch'):

            self.optimize_parameters(model_input, model_target, mask)

            #Track Oversampling
            for l in range(len(label)):
                if label[l].item() == 1:
                    ped += 1
                else:
                    nonPed += 1

            loss = self.log.loss_v
            log_l = self.log.log_l
            mse = self.log.mse
            train_var = self.log.train_var
            train_sigma = self.log.train_sigma
            train_mae = self.log.train_mae
            train_mvar = self.log.model_var
            self.step += 1

            if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
                logger.error("Loss exploded to %.02f at step %d!" %
                             (loss, self.step))
                raise Exception("Loss exploded")

            self.write_train_summary(logger, loss, log_l, mse, train_var, train_sigma, train_mae, train_mvar)

            if self.cfg.train.reduce_lr_step is not None:
                self.reduce_lr(logger)

        if self.cfg.track_oversampling:
            self.log_ped_percentage(ped, nonPed)

    def reduce_lr(self, logger,factor=10):
            if self.step % self.cfg.train.reduce_lr_step == 0:
                self.optimizer = modify_learning_rate(self.optimizer,factor=factor)
                logger.info(
                        f'Learning Rate Successfully Reduced by a factor of {factor}')
    
    def write_validation_summary(self, logger, total_val_loss, total_val_mse, 
                                 total_val_var, total_val_mae, total_val_sigma, 
                                 total_log_l, val_loop_len, total_val_mvar, 
                                 total_MAD_layer, total_MAD_all_layer,std_MAD_layer):
        if self.writer is not None:
            self.writer.logging_with_step(total_val_loss, self.step,
                                              "Validation KDLoss")
            self.writer.logging_with_step(total_val_mse, self.step,
                                              "Validation MSE")
            self.writer.logging_with_step(total_val_var, self.step,
                                              "Validation Variance")
            self.writer.logging_with_step(total_val_mae, self.step,
                                              "Validation MAE")
            self.writer.logging_with_step(total_val_sigma, self.step,
                                              "Validation Sigma")
            self.writer.logging_with_step(total_log_l, self.step,
                                              "Validation Log Likelihood")
            self.writer.logging_with_step(
                    total_MAD_all_layer / val_loop_len, self.step,
                    "Validation MAE in Pixels")
            self.writer.logging_with_step(total_val_mvar, self.step,
                                              "Validation Model Variance")

            for layer in self.layers:
                self.writer.logging_with_step(
                        total_MAD_layer[layer] / val_loop_len, self.step,
                        f'Validation MAE in Pixel of Layer {layer}')
                self.writer.logging_with_step(
                        std_MAD_layer[layer], self.step,
                        f'Validation Std of MAE in Pixel of Layer {layer}')
                logger.info(f'Validation MAE in Pixel of Layer {layer} {total_MAD_layer[layer] / val_loop_len}')
                logger.info(f'Validation Std of MAE in Pixel of Layer {layer} {std_MAD_layer[layer]}')
                
        if is_logging_process():
            logger.info("Validation Loss %.04f at step %d" %
                            (total_val_loss, self.step))
            logger.info("Validation MSE %.04f at step %d" %
                            (total_val_mse, self.step))
            logger.info("Validation Variance %.04f at step %d" %
                            (total_val_var, self.step))
            logger.info("Validation MAD %.04f pixels at step %d" %
                            (total_MAD_all_layer / val_loop_len, self.step))
            logger.info("Validation MAE %.04f at step %d" %
                            (total_val_mae, self.step))
            logger.info("Validation Sigma %.04f at step %d" %
                            (total_val_sigma, self.step))
            logger.info("Validation Log Likelihood %.04f at step %d" %
                            (total_log_l, self.step))

    def write_train_summary(self, logger, loss, log_l, mse, train_var, train_sigma, train_mae, train_mvar,is_hard_sample=False):
        if self.step % self.cfg.log.summary_interval == 0:
            if self.writer is not None:
                self.writer.logging_with_step(mse, self.step, "Train MSE")
                self.writer.logging_with_step(train_var, self.step,
                                                "Train Variance")
                self.writer.logging_with_step(mse, self.step,
                                                  "Train Sigma")
                self.writer.logging_with_step(train_mae, self.step,
                                                  "Train MAE")
                self.writer.logging_with_step(loss, self.step,
                                                  "Train Loss")
                self.writer.logging_with_step(log_l, self.step,
                                                  "Train Log Likelihood")
                self.writer.logging_with_step(train_mvar, self.step,
                                                  "Train Model Variance")
            if is_hard_sample:
                self.writer.logging_with_step(loss, self.step,"Train Loss Hard Sample")
               

            if is_logging_process():
                logger.info("Train MSE %.04f at step %d" %
                                (mse, self.step))
                logger.info("Train Variance %.04f at step %d" %
                                (train_var, self.step))
                logger.info("Train Sigma %.04f at step %d" %
                                (train_sigma, self.step))
                logger.info("Train MAE %.04f at step %d" %
                                (train_mae, self.step))
                logger.info("Train Loss %.04f at step %d" %
                                (loss, self.step))
                logger.info("Train Log Likelihood %.04f at step %d" %
                                (log_l, self.step))
    
    def log_ped_percentage(self, ped, nonPed):
        self.writer.logging_with_step(100 * ped / (ped + nonPed),
                                          self.step, "PED %")
        self.writer.logging_with_step(100 * nonPed / (ped + nonPed),
                                          self.step, "nonPED %")
    
    def optimize_parameters(self, model_input, model_target, mask):
        self.net.train()
        self.net = self.net.to(self.cfg.device)
        self.optimizer.zero_grad()
        output = self.run_network(model_input)

        if self.cfg.artifact:
            # this needs to be changed (naming)
            mask, gt_variance = mask
            loss_v = self.loss_f(output, model_target.to(self.device),
                                 mask.to(self.device),
                                 gt_variance.to(self.device))
        else:
            gt_variance = torch.ones_like(model_target)
            loss_v = self.loss_f(output, model_target.to(self.cfg.device),
                                 mask.to(self.cfg.device),
                                 gt_variance.to(self.device))

        # Update model
        loss_v.backward()
        self.optimizer.step()

       
        train_var = torch.mean(output[1].exp())
        log_l = compute_log_likelihood(output,
                                        model_target.to(self.device),
                                        mask.to(self.device))
        mse = self.mse(output[0], model_target.to(self.device),
                        mask.to(self.device))
        train_sigma = torch.mean(torch.exp(0.5 * output[1]))
        train_mae = self.l1(output[0], model_target.to(self.device),
                            mask.to(self.device))

        # set log
        self.log.loss_v = loss_v.item()
        self.log.log_l = log_l.item()
        self.log.mse = mse.item()
        self.log.train_var = train_var.item()
        self.log.train_sigma = train_sigma.item()
        self.log.train_mae = train_mae.item()
        self.log.model_var = 0
        return output

    def validate_model(self, val_loader):
        logger = get_logger(self.cfg,
                            os.path.basename(__file__),
                            disable_console=True)
        self.net.eval()
        total_val_loss = 0
        total_val_mse = 0
        total_val_var = 0
        total_val_mae = 0
        total_val_sigma = 0
        total_log_l = 0
        val_loop_len = 0
        total_val_mvar = 0
        self.last_val_loss = 10e5  # just a random large number
        total_MAD_layer = {'ILM': 0, 'RPEDC': 0, 'BM': 0}
        total_MAD_all_layer = 0

        with torch.no_grad():
            for model_input, model_target, mask in tqdm(val_loader,
                                                        leave=False,
                                                        desc="Testing/Batch:"):
                if self.cfg.inference_mode and self.cfg.testTime_dropout:
                    # Keep dropout at test time
                    self.net = unfreeze_dropout(self.net)
                    mean_pred, model_var, mean_data_var = self.multi_forward_pass(
                        model_input, num_pass=10)
                    output = (mean_pred, mean_data_var)
                else:
                    output = self.inference(model_input)

                gt_variance = torch.ones_like(model_target)
                loss_v = self.loss_f(output, model_target.to(self.cfg.device),
                                     mask.to(self.cfg.device),
                                     gt_variance.to(self.cfg.device))

                if self.cfg.dist.gpus > 0:
                    # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                    torch.distributed.all_reduce(loss_v)
                    loss_v /= torch.tensor(float(self.cfg.dist.gpus))

                total_val_loss += torch.mean(loss_v).to("cpu").item()
                val_loop_len += 1

                # Additional metrics goes here
                total_log_l=0
                total_val_mse += self.mse(output[0],
                                          model_target.to(self.device),
                                          mask.to(
                                              self.device)).to("cpu").item()

                total_val_var += torch.mean(output[1].exp()).to("cpu").item()

                total_val_mae += self.l1(output[0], model_target.to(
                    self.device), mask.to(self.device)).to("cpu").item()

                total_val_sigma += torch.mean(torch.exp(0.5 * output[1]))

                # Get the total and layer wise mean absolute differences
                MAD_layer, MAD_all_layer = self.measure_fit(
                    model_target, output[0])
                total_MAD_all_layer += MAD_all_layer

                for layer in self.layers:
                    total_MAD_layer[layer] += MAD_layer[layer]

                # Save the visual results during inference
                # The cfg parameter shall be overrided from
                # the inference script to avoid accidental
                # saving during training
                if self.cfg.inference_mode and self.cfg.testTime_dropout:
                    self.save_prediction_img(to_numpy(model_input), output,
                                             to_numpy(model_target),
                                             to_numpy(model_var))
                    self.im_counter += 1
                elif self.cfg.inference_mode:
                    self.save_prediction_img(to_numpy(model_input), output,
                                             to_numpy(model_target))
                    self.im_counter += 1

            total_val_loss /= val_loop_len
            total_log_l /= val_loop_len
            total_val_mae /= val_loop_len
            total_val_sigma /= val_loop_len
            total_val_var /= val_loop_len
            total_val_mvar /= val_loop_len

            # Excute scheduler on validation loss
            if self.cfg.train.use_scheduler == 1:
                self.scheduler.step(total_val_loss)

            # Saves model with best accuracy
            if (total_MAD_all_layer / val_loop_len) < self.last_val_loss:
                self.save_best_model()
                self.last_val_loss = (total_MAD_all_layer / val_loop_len)

            self.write_validation_summary(logger, total_val_loss, total_val_mse, total_val_var, 
                                          total_val_mae, total_val_sigma, total_log_l, val_loop_len, 
                                          total_val_mvar, total_MAD_layer, total_MAD_all_layer)
            
    def inference(self, model_input):
        self.net.eval()
        output = self.run_network(model_input)
        return output

    def run_network(self, model_input):
        model_input = model_input.to(self.device)
        output = self.net(model_input)
        return output

    def save_network(self, save_file=True):
        if is_logging_process():
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                save_filename = "%s_%d.pt" % (self.cfg.name, self.step)
                save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                if self.cfg.log.use_wandb:
                    wandb.save(save_path)
                if is_logging_process():
                    self._logger.info("Saved network checkpoint to: %s" %
                                      save_path)
            return state_dict

    def load_network(self, loaded_net=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if self.cfg.load.wandb_load_path is not None:
                self.cfg.load.network_chkpt_path = wandb.restore(
                    self.cfg.load.network_chkpt_path,
                    run_path=self.cfg.load.wandb_load_path,
                ).name
            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net,
                                 strict=self.cfg.load.strict_load)
        if is_logging_process() and add_log:
            self._logger.info("Checkpoint %s is loaded" %
                              self.cfg.load.network_chkpt_path)

    def save_training_state(self):
        if is_logging_process():
            save_filename = "%s_%d.state" % (self.cfg.name, self.step)
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            torch.save(state, save_path)
            # if self.cfg.log.use_wandb:
            #     wandb.save(save_path)
            if is_logging_process():
                self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        if self.cfg.load.wandb_load_path is not None:
            self.cfg.load.resume_state_path = wandb.restore(
                self.cfg.load.resume_state_path,
                run_path=self.cfg.load.wandb_load_path,
            ).name
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        if is_logging_process():
            self._logger.info("Resuming from training state: %s" %
                              self.cfg.load.resume_state_path)

    def measure_fit(self, y_true, y_pred):
        '''
        
        To calculate accuracy/dice/IoU etc score to evaluate the 
        fit of the model.Needs to be implemented project-wise.
        
        '''
        if self.cfg.sdm:
             return get_layer_wise_sdm(y_true=y_true,y_pred=y_pred)
        else:
            return get_layer_wise_(y_true=y_true,y_pred=y_pred)

    def save_best_model(self):
        if is_logging_process():

            save_filename = "best.state"
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            #check if file exists
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(state, save_path)
            if self.cfg.log.use_wandb:
                wandb.save(save_path)
            if is_logging_process():
                self._logger.info(
                    f"Saved best model state to: {save_path} at {self.step} step"
                )

    def tensor_to_numpy(self,input_data):
        if isinstance(input_data, torch.Tensor):
            # Check if the tensor is on CUDA and move to CPU if necessary
            if input_data.device.type == 'cuda':
                input_data = input_data.cpu()
            # Convert to NumPy array
            return input_data.numpy()
        elif isinstance(input_data, np.ndarray):
            # If it's already a NumPy array, return it as is
            return input_data
        else:
            raise TypeError("Input must be a PyTorch tensor or a NumPy array")

    def save_prediction_img(self,
                            imgs,
                            y_pred,
                            y_true,
                            model_var=None,
                            raw_prob=None,output_var=None,idx=None,file_name_list=None,full_anno=None):

        #check if path exists
        tmp_path = os.path.join(self.cfg.output_dir, 'output_imgs')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        y_mean, y_sigma = y_pred, output_var
        
        for i in range(imgs.shape[0]):
            print('here: ',file_name_list[i])

            np.save(
                os.path.join(tmp_path, f'{file_name_list[i]}_imgs.npy'),
                imgs[i].squeeze())
            np.save(
                os.path.join(tmp_path, f'{file_name_list[i]}_mean.npy'),
                y_mean[i].squeeze())
            np.save(
                os.path.join(tmp_path, f'{file_name_list[i]}_true.npy'),
                y_true[i].squeeze())
            
            if full_anno is not None:
                file=os.path.join(tmp_path, f'{file_name_list[i]}.json')   
                with open(file, 'w') as json_file:
                    json.dump(full_anno[i], json_file)
                json_file.close()

            if y_sigma is not None:
                # y_sigma=self.tensor_to_numpy(y_sigma)
                np.save(os.path.join(tmp_path, f'{file_name_list[i]}_sigma.npy'),
                    y_sigma[i].squeeze())
            if model_var is not None:
                np.save(
                    os.path.join(tmp_path,
                                    f'{file_name_list[i]}_model_var.npy'),
                    model_var[i].squeeze())
            if raw_prob is not None:
                np.save(
                    os.path.join(tmp_path,
                                    f'{file_name_list[i]}_prob.npy'),
                    raw_prob[i].squeeze())
                    