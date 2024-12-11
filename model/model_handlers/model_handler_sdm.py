import math
import os
import numpy as np
import torch.nn 
from tqdm import tqdm
import torch
from scipy.stats import norm
from model.model_handlers.base_model_handler import Base_Model_handler
from utils.pipeline_utils import  SoftsdmToLayer, sdmToLayer,show_im
from utils.utils import get_logger, is_logging_process, modify_learning_rate,  to_numpy
from utils.writer import Writer
from collections import deque


class Model_handler_sdm(Base_Model_handler):

    def __init__(self, cfg, net_arch, loss_f, writer: Writer, rank=0):
        super().__init__(cfg, net_arch, loss_f, writer, rank=0)
        self.hard_sample=[]
        self.running_loss = deque(maxlen=100)
        self.logger=get_logger(self.cfg,
                            os.path.basename(__file__),
                            disable_console=True)
        optimizer_mode = self.cfg.train.optimizer.mode
        scheduler_mode = self.cfg.train.scheduler.mode 
        # Learned Clamping of L1/L2 Loss near the boundry 
        # This has not been used in for the publication!
        self.censoring_threshold = torch.nn.Parameter(torch.tensor(self.cfg.sdm_misc.clamp_delta,
                                                                   dtype=torch.float32))
        if cfg.learned_censoring:
            if optimizer_mode == "adam":
                self.optimizer = torch.optim.Adam(
                    list(self.net.parameters())+[self.censoring_threshold],
                    **(self.cfg.train.optimizer[optimizer_mode]))
                self.logger.info("Optimizer is successfully updated")
            if scheduler_mode == 'ReduceLROnPlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **(self.cfg.train.scheduler[scheduler_mode]))
            else:
                raise Exception(
                    f"optimizer or scheduler not supported {optimizer_mode,scheduler_mode}"
                )

    def train_model(self, train_loader):
        self.hard_sample_count=0
        for model_input, model_target, mask, layer,idx,_ in tqdm(
                train_loader, leave=False, desc="Training/Batch:",
                unit='batch'):
            _,output=self.optimize_parameters(model_input, model_target, mask)
            loss = self.log.loss_v
            log_l = self.log.log_l
            mse = self.log.mse
            train_var = self.log.train_var
            train_sigma = self.log.train_sigma
            train_mae = self.log.train_mae
            train_mvar = self.log.model_var
            self.step += 1
            if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
                self.logger.error("Loss exploded to %.02f at step %d!" %
                             (loss, self.step))
                # raise Exception("Loss exploded")
            self.write_train_summary(self.logger, loss, log_l, mse, train_var, 
                                     train_sigma, train_mae, train_mvar)
            
            if self.step!=0 and  self.cfg.train.reduce_lr_step is not None:
               self.reduce_lr(self.logger)
    
            if self.cfg.gaussian_output:
                output,output_var=output[0],output[1].exp()

            # Get the total and layer wise mean absolute differences
            output_layer=sdmToLayer(to_numpy(output.squeeze()))
            _,_,sample_mad_layer=self.measure_fit(to_numpy(layer.permute(0,1,3,2)),
                                                  np.expand_dims(output_layer.transpose(0,2,1),axis=1))
            # Hard case over Sampling Training
            if self.cfg.sdm_misc.oversample_hardCase:
                self.train_on_hardSample(sample_mad_layer[:,-1], idx, train_loader)             
        torch.cuda.empty_cache()

    def train_on_hardSample(self, unreduced_error, idx, train_loader):
        for l in unreduced_error:
            self.running_loss.append(l.item())

        if self.step > 75:
            self.median_loss = np.median(list(self.running_loss))
            for i, l in enumerate(unreduced_error):
                if l.item() > self.median_loss:
                    self.hard_sample.append(idx[i].item())

        if len(self.hard_sample) >= (train_loader.batch_size * 4):
            for _ in range(len(self.hard_sample)):
                if len(self.hard_sample) >= train_loader.batch_size:
                    model_input, model_target, mask=[],[],[]
                    for _ in range(train_loader.batch_size):
                        input, target, msk, _, _, _ = train_loader.dataset[self.hard_sample.pop()]
                        model_input.append(input) 
                        model_target.append(target) 
                        mask.append(msk)
                    
                    model_input, model_target, mask = (torch.stack(tensor) for tensor in
                                                        (model_input, model_target, mask))
                    
                    self.optimize_parameters(model_input, model_target, mask)
                    loss = self.log.loss_v
                    log_l = self.log.log_l
                    mse = self.log.mse
                    train_var = self.log.train_var
                    train_sigma = self.log.train_sigma
                    train_mae = self.log.train_mae
                    train_mvar = self.log.model_var
                    self.step += 1
                    self.hard_sample_count += 1
                    self.write_train_summary(self.logger, loss, log_l, mse, train_var, 
                                             train_sigma, train_mae, train_mvar, 
                                             is_hard_sample=True)
    
    def log_ped_case(self,data_loader,main_list=None):
        '''
            This just for debug purpose to Log the heavy 
            pigment epithelial detachment case visualization
        '''
        counter=0
        if main_list is None:
            main_list=[145,170,184,96,269,737,875,925]
        
        for p in range(0,int(len(main_list)/2)):
            sample_list = main_list[p*2:p*2+2]
            model_input, model_target,layer=[],[],[]
            for i in sample_list:
                input, target, _, lyr, _, _ = data_loader.dataset[i]
                model_input.append(input) 
                model_target.append(target) 
                layer.append(lyr)

            model_input, model_target, layer = (torch.stack(tensor) for tensor in 
                                                (model_input, model_target, layer))
            with torch.no_grad():
                output =  self.inference(model_input) 
            if self.cfg.gaussian_output:
                output,output_var=output[0],to_numpy(output[1])

            model_input, layer_gt, raw_prob = (to_numpy(tensor) for tensor in 
                                               (model_input, layer.permute(0, 1, 3, 2), output))

            
            del output
            torch.cuda.empty_cache()

            for i in range(model_input.shape[0]):
                input, y_true, layers_var = (model_input[i].squeeze(),
                                            layer_gt[i].squeeze(),
                                            np.zeros_like(layers))   
                layers=sdmToLayer(np.expand_dims(raw_prob[i].squeeze(),axis=0)).squeeze().T
                if self.cfg.gaussian_output:
                    for j in range(layers.shape[0]):
                        for k in range(layers.shape[1]):
                            layers_var[j,k] = output_var[i,j,np.int32(layers[j,k]), k]
                    layers_var[y_true==0]=np.nan
                    layers_sigma=np.sqrt(layers_var.transpose(1,0))
                else:
                    layers_sigma=None
                layers[y_true==0] = np.nan
                y_true[y_true==0]=np.nan
                # 512 is based on the Height of the Images in the Duke data set and 
                # the external dataset, this needs to be adapted if the img shape changes
                np_img=show_im(input,layers.transpose(1,0),y_true.transpose(1,0)*512,
                    layers_sigma,outfile=None)
                self.writer.log_image(np_img, f'PedCase_{main_list[counter]}', self.step)
                counter+=1

    def optimize_parameters(self, model_input, model_target, mask):
        '''
            Optimize the model parameters for sdm model
        '''
        self.net.train()
        self.net = self.net.to(self.cfg.device)
        self.optimizer.zero_grad()
        output=self.run_network(model_input)    
        unreduced_loss = self.loss_f(output, 
                                    model_target.to(self.device),
                                    mask.to(self.device))
        # This is because we need the unreduced 
        # loss for the hard sample training!
        # Reduce the loss along the batch dimension
        loss_v=unreduced_loss.mean()
        if not torch.isnan(loss_v):
            loss_v.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
        if self.cfg.gaussian_output:            
            mse = self.mse(output[0], model_target.to(self.device),
                        mask.to(self.device)).mean()
            train_mae = self.l1(output[0], model_target.to(self.device),
                            mask.to(self.device)).mean()
            self.log.train_var = output[1].mean().item()
        else:    
            mse = self.mse(output, model_target.to(self.device),
                        mask.to(self.device)).mean()
            train_mae = self.l1(output, model_target.to(self.device),
                            mask.to(self.device)).mean()   
            self.log.train_var = 0
        
        self.log.loss_v, self.log.train_mae, self.log.mse = (loss_v.item(), 
                                                             train_mae.item(), 
                                                             mse.item())
        self.log.log_l = self.log.train_sigma = self.log.model_var = 0

        if self.step % self.cfg.train.reduce_lr_step == 0:
            factor = 2 
            self.optimizer = modify_learning_rate(self.optimizer,factor=factor)
            self.logger.info(
                    f'Learning Rate Successfully Reduced by a factor of {factor}')
        return unreduced_loss,output
    
    def validate_model(self, val_loader):
        self.net.eval()
        (total_val_loss,
         total_val_mse,
         total_val_var,
         total_val_mae,
         total_val_sigma,
         total_log_l,
         val_loop_len,
         total_MAD_all_layer) = [0] * 8
        keys = ['ILM', 'RPEDC', 'BM']
        total_MAD_layer = dict.fromkeys(keys, 0.0)
        std_MAD_layer = dict.fromkeys(keys, 0.0)
        MAD_list_layer = {key: [] for key in keys}

        with torch.no_grad():
            for model_input, model_target, mask,layer,idx,file_name_list in tqdm(val_loader,
                                                        leave=False,
                                                        desc="Testing/Batch:"):
                output= self.inference(model_input)
                loss_v = self.loss_f(output,
                                     model_target.to(self.device),
                                     mask.to(self.cfg.device)).mean()
                if self.cfg.dist.gpus > 0:
                    # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                    torch.distributed.all_reduce(loss_v)
                    loss_v /= torch.tensor(float(self.cfg.dist.gpus))
                total_val_loss += loss_v.to("cpu").item()
                val_loop_len += 1

                if self.cfg.gaussian_output:
                    output,output_var=output[0],output[1].exp()
                    total_val_var += output_var.mean().item() 
                else:
                    output_var=None
                    total_val_var = 0 
                
                total_log_l += 0
                total_val_sigma += 0
                total_val_mse += self.mse(output, model_target.to(self.device),
                        mask.to(self.device)).mean().item()
                total_val_mae += self.l1(output, model_target.to(self.device),
                            mask.to(self.device)).mean().item()

                # Get the total and layer wise mean absolute differences
                # output_layer=sdmToLayer(to_numpy(output.squeeze()))
                output_layer=SoftsdmToLayer(to_numpy(output.squeeze()))
                MAD_layer,MAD_all_layer,_=self.measure_fit(to_numpy(layer.permute(0,1,3,2)),
                                                           np.expand_dims(output_layer.transpose(0,2,1),axis=1))
                total_MAD_all_layer+=MAD_all_layer
                            
                for layer_name in self.layers:
                    total_MAD_layer[layer_name]+=MAD_layer[layer_name]
                    MAD_list_layer[layer_name].append(MAD_layer[layer_name])
                
                # Save the visual results during inference
                # The cfg parameter shall be overriden from
                # the inference script to avoid accidental
                # saving during training
                if self.cfg.inference_mode:
                    self.save_prediction_img(imgs=to_numpy(model_input),
                                             y_pred=(output_layer),
                                             y_true=to_numpy(layer.permute(0,1,3,2)),
                                             raw_prob=to_numpy(output),
                                             output_var=None if output_var==None else to_numpy(output),
                                             idx=idx,file_name_list=file_name_list)
                    self.im_counter += 1
                    self.logger.info(f'Image {self.im_counter} saved')

            total_val_loss /= val_loop_len
            total_log_l /= val_loop_len
            total_val_mae /= val_loop_len
            total_val_sigma /= val_loop_len
            total_val_var /= val_loop_len

            for layer_name in self.layers:
                std_MAD_layer[layer_name] = float(np.std(MAD_list_layer[layer_name]))

            # Execute scheduler on validation loss
            if self.cfg.train.use_scheduler == 1:
                self.scheduler.step(total_val_loss)

            # Write the summary to WnB  
            total_val_mvar=0
            self.write_validation_summary(self.logger, total_val_loss, total_val_mse, total_val_var, 
                                          total_val_mae, total_val_sigma, total_log_l, val_loop_len, 
                                          total_val_mvar, total_MAD_layer, total_MAD_all_layer,std_MAD_layer)
            # Call the Scheduler
            self.scheduler.step(total_val_loss)
           