import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

# from torchao.prototype.low_bit_optim import AdamW8bit
import bitsandbytes as bnb
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import einops
from lerobot.common.policies.pretrained import PreTrainedPolicy

import wandb
from src.agent.configuration_pipeline import TrainPipelineConfig
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, blockprint, log_allocated_gpu_memory, log_execution_time, setup_logger
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from src.utils.pipeline import process_images, set_seed_everywhere

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, set_seed

from future_view_prediction_w_action_dataset import get_train_val_data_loaders

os.environ["WANDB__SERVICE_WAIT"] = "300"

class BaseTrainer:
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        self.train_cfg = train_cfg
        self.model_cfg = train_cfg.model_cfg
        self.model_class = model_class

        # Setup run name
        if train_cfg.name is None:
            self.name = (train_cfg.data.train.dataset_mix + "_" +
                         train_cfg.data.train.split + "_tp" +
                         str(train_cfg.data.train.action_horizon))
        else:
            self.name = train_cfg.name

        self.wandb_runid = None

        # Seeding
        set_seed_everywhere(train_cfg.seed)

        # Device and multi-GPU settings
        self.gpu_id = train_cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = train_cfg.multi_gpu
        
        # Calculate gradient accumulation steps first
        self.grad_accumulation_steps = max(
            train_cfg.global_batch_size // train_cfg.per_device_batch_size // (int(os.environ.get("WORLD_SIZE", "1")) if self.multi_gpu else 1), 1
        )
        
        self.log_dir: Path = (
            Path(
            os.environ["VLA_LOG_DIR"])
            / "train"
            / self.name
            / (time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.train_cfg.seed}")
        )
        
        # Initialize accelerator after setting up grad_accumulation_steps
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision="bf16",
            log_with="wandb",
            project_dir=self.log_dir,
            split_batches=True,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        
        self.world_size = self.accelerator.num_processes
        self.main_rank = self.accelerator.is_main_process
        
        # Backwards compatibility - keep these attributes for existing code
        if self.multi_gpu:
            self.global_rank = self.accelerator.process_index
            self.local_rank = self.accelerator.local_process_index
            self.local_world_size = getattr(self.accelerator, "num_processes_per_node", self.world_size)
            self.group_rank = getattr(self.accelerator, "node_index", 0)

        if not self.main_rank:
            blockprint()

        # checkpoint/log/directory setup
        self.debug = train_cfg.debug
        self.log = setup_logger(main_rank=self.main_rank,
                                filename=None, # log to file. If None then to stdout
                                debug=self.debug) # If debug=True, DEBUG level and up will show, else INFO
        if self.multi_gpu:
            self.log.info(f"GPU local ID: {self.gpu_id}. Global rank: {self.global_rank}. Local rank: {self.local_rank}. \
                Local world size: {self.local_world_size}. World size: {self.world_size}. Group rank: {self.group_rank}"
        )
            for i in range(torch.cuda.device_count()):
                self.log.info(f"Local rank: {self.local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}")

        self.save_model_freq = int(train_cfg.save_model_freq)
        self.log_freq = train_cfg.log_freq

        if self.main_rank:
            self._dir_setup()

        # Training parameters
        self.n_updates = int(train_cfg.n_updates) # number of gradient updates. != gradient steps due to gradient accumulation
        self.use_amp = train_cfg.use_amp
        self.dtype = torch.bfloat16 if train_cfg.use_bf16 else torch.float32

        # Model initialization
        self._initialize_model(train_cfg, model_class)

        # Actual global batch size based on already calculated grad_accumulation_steps
        # (grad_accumulation_steps was calculated before accelerator initialization)
        self.actual_global_batch_size = train_cfg.per_device_batch_size * self.grad_accumulation_steps * self.world_size

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
                train_cfg.per_device_batch_size)


        if self.model_class.name == "action":
            self.model.model.action_expert.gemma_expert.lm_head = None

        self.model.to(self.device)
        if train_cfg.use_torch_compile:
            self.model = torch.compile(self.model,
                                       mode="default")

        self.log.info(f"Using cuda device: {self.device}, dtype: {self.dtype}")

        # Accelerator handles the distributed training
        if self.multi_gpu:
            self.log.info(f"Using Accelerator for distributed training with {self.accelerator.num_processes} processes")

        log_allocated_gpu_memory(log=self.log, stage="loading model", device=self.gpu_id)

        self.action_horizon = self.model_cfg.chunk_size

        # Dataloaders
        
        self.train_dataloader, self.val_dataloader = get_train_val_data_loaders(
            dataset_path="/vast/bc4227/datasets/bridge_processed_with_state",
            batch_size=train_cfg.per_device_batch_size,
            num_workers=8,
            world_size=self.accelerator.num_processes,
            local_rank=self.accelerator.process_index,
            resolution=256,
            future_step=10)
        
        # Evaluation parameters
        self.eval_thresholds = train_cfg.eval_thresholds
        self.eval_freq = train_cfg.eval_freq
        self.per_device_num_eval_batch = train_cfg.eval_size // train_cfg.per_device_batch_size // self.world_size
        
        self.log.info(f"Total number of gradient updates: {self.n_updates}")
        self.log.info(f"Actual global batch size: {self.actual_global_batch_size}")
        self.log.info(f"Per device batch size: {train_cfg.per_device_batch_size}")
        self.log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        self.log.info(f"Global batch_size after gradient accumulation: {train_cfg.global_batch_size}")
        self.log.info(f"Number of training batches: {len(self.train_dataloader)}")
        self.log.info(f"Number of validation batches: {len(self.val_dataloader)}")

        # Optimizer and scheduler
        self.optimizer = bnb.optim.AdamW8bit(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.model_cfg.optimizer_lr,
            betas=self.model_cfg.optimizer_betas,
            eps=self.model_cfg.optimizer_eps,
            weight_decay=self.model_cfg.optimizer_weight_decay,
        )
        # # ? if full precision optimizer will be more stable?
        # self.optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.model_cfg.optimizer_lr,
        #     betas=self.model_cfg.optimizer_betas,
        #     eps=self.model_cfg.optimizer_eps,
        #     weight_decay=self.model_cfg.optimizer_weight_decay,
        # )

        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=10000000,
            cycle_mult=1.0,
            max_lr=self.model_cfg.optimizer_lr,
            min_lr=1e-8,
            warmup_steps=self.model_cfg.scheduler_warmup_steps,
            gamma=1.0,
        )
        
        # Prepare the model, optimizer, scheduler, and dataloader with accelerator
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )
        # Prepare the validation dataloader with accelerator
        self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        
        self.log.info(f"Number of trained parameters: {get_num_params_in_billions(self.optimizer):.3f}B")

        # * hard coded with if else for now, to accomodate fusiona.
        # Parameter counting - use accelerator to get the unwrapped model
        base_model = self.accelerator.unwrap_model(self.model)
        if self.model_class.name == "action":
            action_parameters = filter(lambda p: p.requires_grad, base_model.model.action_expert.gemma_expert.parameters())
            self.action_param_count = sum(p.numel() for p in action_parameters)
            self.log.info(f"Number of trained parameters (Action): {self.action_param_count/1e9:.3f}B")

        # Training state
        self.timer = Timer()
        self.cnt_batch = 0 # number of batches processed
        self.cnt_update = 0 # number of gradient updates. Can be smaller than cnt_batch due to gradient accumulation

        if train_cfg.resume_run and train_cfg.load_from_checkpoint is not None:
            self.log.info(f"resume previous run? {train_cfg.resume_run}, loading optimizer states and auxiliary data.")
            self._load_optimizer_and_auxiliary_data(train_cfg.load_from_checkpoint, resume_wandb=True)

        # Training log-related
        # Every metric get its own deque
        self.train_log_metrics = train_cfg.train_log_metrics # a list of metrics that we hope to keep track of
        self.train_log_metrics_dict = {} # holds the temporary metric before reduce
        self.train_log_deque_dict = {} # holds all the metrics after reduce
        for metric in self.train_log_metrics:
            self.train_log_metrics_dict[metric] = 0.0
            self.train_log_deque_dict[metric] = deque(maxlen=self.grad_accumulation_steps)

        # Evaluation log-related
        self.eval_log_metrics = train_cfg.eval_log_metrics # a list of metrics that we hope to keep track of
        self.eval_log_metrics_dict = {}
        self.new_eval_from_last_log = False # this is to check if we need to log the evaluation metrics

        # wandb setup - use accelerator's tracking
        if train_cfg.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=train_cfg.wandb.project,
                config=asdict(train_cfg),
                init_kwargs={
                    "wandb": {
                        "entity": train_cfg.wandb.entity,
                        "name": time.strftime("%Y%m%d-%H%M%S") + "_" + self.name,
                        "id": self.wandb_runid,
                        "resume": "allow",
                    }
                }
            )
    
    def train(self):
        self.model.train()
        while True:
            for batch in self.train_dataloader:
                # Use accelerator for automatic gradient accumulation handling
                with self.accelerator.accumulate(self.model):
                    # inputs = self.preprocess_batch(batch=batch)
                    inputs = batch
                    
                    with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
                        loss_train, train_dict = self.model(batch=inputs)
                        
                        self._extract_train_log(train_dict)
                        
                        if self.debug:
                            log_allocated_gpu_memory(log=self.log, stage=f"forward batch {self.cnt_batch}")
                    
                    # Accelerator handles loss scaling for mixed precision
                    self.accelerator.backward(loss_train)
                    
                    # Only clip gradients when we're about to update
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.train_cfg.max_grad_norm)
                        self._extract_train_log_add("grad norm", grad_norm)
                    
                    # Accelerator handles optimizer step and zero_grad
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if self.debug and self.accelerator.sync_gradients:
                        log_allocated_gpu_memory(log=self.log, stage=f"optimizer step batch {self.cnt_batch}")
                    
                    # Only increase update counter when we actually update parameters
                    if self.accelerator.sync_gradients:
                        self.cnt_update += 1
                        
                        # Validation step
                        # Validate once in a while to check overfitting
                        if self.cnt_update % self.eval_freq == 0:
                            self.new_eval_from_last_log = True
                            self.validate()
                            self.model.train() # explicitly set back to train mode
                        
                        # save model and auxiliary data at the end of an update
                        if self.cnt_update % self.save_model_freq == 0 or self.cnt_update == self.n_updates:
                            self._save_training() # takes care of main rank in the function
                            
                # Loss process for logging
                self._process_train_log()
                
                # Log training metrics
                if self.accelerator.is_main_process and self.cnt_update % self.log_freq == 0 and self.accelerator.sync_gradients:
                    self._log_training()
                    
                    if self.train_cfg.use_wandb:
                        self._log_wandb()
                
                self.cnt_batch += 1
                if self.cnt_update >= self.n_updates:
                    return # end training
    
    def validate(self):
        self.model.eval()
        self._initialize_eval_log()

        if self.accelerator.is_main_process:
            self.log.info(f"Running evaluation for {self.per_device_num_eval_batch} batches...")
        
        val_dataloader_iterator = iter(self.val_dataloader)
        
        with torch.no_grad():
            for _ in range(self.per_device_num_eval_batch):
                try:
                    batch_eval = next(val_dataloader_iterator)
                except StopIteration:
                    # If we've exhausted the dataloader, restart from the beginning
                    val_dataloader_iterator = iter(self.val_dataloader)
                    batch_eval = next(val_dataloader_iterator)

                # inputs = self.preprocess_batch(batch=batch_eval)
                inputs = batch_eval
                
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
                    # Use accelerator to handle distributed evaluation
                    model = self.accelerator.unwrap_model(self.model)
                    pred_actions = torch.stack(
                        [model.select_action(inputs) for _ in range(self.action_horizon)],
                        dim=1
                    )
                    gt_actions = inputs["actions"]
                    
                self._extract_eval_log(gt_actions, pred_actions)

        self._process_eval_log()

        if self.accelerator.is_main_process:
            self._log_validation()

    ############################### Training log related functions ###############################
    def _extract_train_log(self, train_dict):
        '''
        Extract the metrics from the training dict and store them in the train_log_metric_dict for reduce later
        '''
        for metric in self.train_log_metrics:
            self.train_log_metrics_dict[metric] = train_dict.get(metric, torch.tensor(0.).to(self.device))

    def _extract_train_log_add(self, log_key: str, log_v: float | torch.Tensor):
        '''
        Additionally add new key and value to the train_log metrics.
        ! use conservatively.
        '''
        if log_key not in self.train_log_metrics_dict:
            self.train_log_metrics_dict[log_key] = 0.0
        if isinstance(log_v, torch.Tensor):
            self.train_log_metrics_dict[log_key] = log_v.detach()
        else:
            self.train_log_metrics_dict[log_key] = torch.tensor(log_v).to(self.device)

    def _process_train_log(self):
        '''
        Aggregate the metrics in the train_log_metric_dict and store them in the train_log_deque_dict
        To output different metrics, you need to modify the train_log_metrics in the config, and also let your model output the metrics in a dict
        '''

        for metric in self.train_log_metrics:
            # Gather metrics from all processes using accelerator
            if isinstance(self.train_log_metrics_dict[metric], torch.Tensor):
                gathered_metric = self.accelerator.gather(self.train_log_metrics_dict[metric])
                avg_metric = gathered_metric.mean().item()
            else:
                avg_metric = self.train_log_metrics_dict[metric]
            self.train_log_deque_dict[metric].append(avg_metric)

    def _log_training(self):
        '''
        log the training metrics to logger.
        Utilize the train_log_deque_dict to get the metrics
        '''

        # we only care about the mean of the last grad_accumulation_steps
        self.wandb_train_log_dict = {k: np.mean(v) for k, v in self.train_log_deque_dict.items()}

        if self.accelerator.is_main_process:
            peak_vram = torch.cuda.max_memory_reserved(self.gpu_id) / (1024**3)
            log_msg = (f"Batch {self.cnt_batch} Update {self.cnt_update}: t {self.timer():8.4f} | "
                        f"vram {peak_vram:6.3f} |  lr {self.optimizer.param_groups[0]['lr']:10.8f}")
            for k, v in self.wandb_train_log_dict.items():
                log_msg += f" | {k}: {v:.3f}"
            self.log.info(log_msg)

    def _log_wandb(self):
        '''
        Log training metrics to wandb using accelerator's tracking
        '''

        wandb_metrics = {
                            "gradient steps": self.cnt_update,
                            "learning rate": self.optimizer.param_groups[0]["lr"],
                        }
        # log various training loss
        for k, v in self.wandb_train_log_dict.items():
            wandb_metrics[f'{k} - train'] = v
        
        # Log eval metrics if we have new ones
        if self.new_eval_from_last_log:
            wandb_metrics.update(
                {
                    f"eval acc - thres {threshold}": accuracy.item()
                    for threshold, accuracy in zip(
                        self.eval_thresholds, self.eval_log_metrics_dict['eval_accuracy'], strict=False
                    )
                }
            )
            for metric in self.eval_log_metrics:
                wandb_metrics[f"{metric} - eval"] = self.eval_log_metrics_dict[metric].item()
            self.new_eval_from_last_log = False
            
        # Use accelerator's tracking instead of direct wandb.log
        self.accelerator.log(wandb_metrics, step=self.cnt_update)

    ############################### Evaluation log related functions ###############################
    def _initialize_eval_log(self):
        '''
        This will be called in validate to initialize the logging
        '''
        self.eval_log_metrics_dict['eval_accuracy'] = torch.zeros(len(self.eval_thresholds), device=self.device)
        for metric in self.eval_log_metrics:
            self.eval_log_metrics_dict[metric] = torch.tensor(0.0, device=self.device)

    def _extract_eval_log(self, gt_actions, pred_actions):
        '''
        This will be called in validate to extract the logging

        This function usually needs to be rewritten when inheriting from this class
        Because different loss has different way to calculate
        '''

        self.eval_log_metrics_dict['eval_accuracy'] += get_action_accuracy(gt_actions, pred_actions, self.eval_thresholds)
        self.eval_log_metrics_dict['l1_loss'] += torch.nn.functional.l1_loss(pred_actions, gt_actions)

    def _process_eval_log(self):
        '''
        This will be called in validate to process the eval logging
        Usually, this will not be changed when inheriting from this class
        Because we just need to average and reduce the metrics
        '''
        for k, v in self.eval_log_metrics_dict.items():
            self.eval_log_metrics_dict[k] = v / self.per_device_num_eval_batch
        # aggregate the metrics if multi-gpu
        if self.multi_gpu:
            import torch.distributed as dist
            for k, v in self.eval_log_metrics_dict.items():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                self.eval_log_metrics_dict[k] = v / dist.get_world_size()


    def _log_validation(self):
        '''
        Log the validation metrics to logger
        Utilize the eval_log_metrics_dict to get the metrics
        Usually, this will not be changed when inheriting from this class
        Because we just log all the metrics specified in the config
        '''
        log_msg = "Eval | "

        # we don't want to treat accuracy as any other scalar metric
        for metric in self.eval_log_metrics:
            log_msg += f"{metric}: {self.eval_log_metrics_dict[metric].item():.3f} | "

        log_msg += "".join(
                [f"acc thres {threshold}: {accuracy.item():.3f}"
                for threshold, accuracy in zip(self.eval_thresholds, self.eval_log_metrics_dict['eval_accuracy'], strict=False)]
            )
        self.log.info(log_msg)

    ############################### Model saving and loading functions ###############################
    def _dir_setup(self):
        self.log_dir: Path = (
            Path(
            os.environ["VLA_LOG_DIR"])
            / "train"
            / self.name
            / (time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.train_cfg.seed}")
        )
        self.checkpoint_dir = self.log_dir / "checkpoint"
        if self.main_rank:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # This cleans up previous runs that are empty, which are likely failed runs
        parent_dir = self.log_dir.parent
        for subdir in parent_dir.iterdir():
            # Skip non-directories and self.log_dir itself.
            if not subdir.is_dir() or subdir == self.log_dir:
                continue
            entries = list(subdir.iterdir())
            # Check if the only entry is a directory named "checkpoint".
            if len(entries) == 1 and entries[0].is_dir() and entries[0].name == "checkpoint":
                checkpoint_dir = entries[0]
                # Remove the directory if the checkpoint folder is empty.
                if not any(checkpoint_dir.iterdir()):
                    checkpoint_dir.rmdir()
                    subdir.rmdir()
                    self.log.info(f"Removed empty run directory: {subdir}")

    def _initialize_model(self, train_cfg, model_class):
        # Model initialization
        if train_cfg.load_from_checkpoint is None:
            self.model = model_class(config=self.model_cfg, dataset_stats=train_cfg.data.dataset_stats)
        else:
            self.model = self._load_model(model_class=model_class, checkpoint_dir=train_cfg.load_from_checkpoint)
            self.log.info(f"Loaded checkpoint from {train_cfg.load_from_checkpoint}.")

    @log_execution_time()
    def _save_training(self):
        # Save model and optimizer states using accelerator
        if self.accelerator.is_main_process:
            model_save_path = self.checkpoint_dir / f"step_{self.cnt_update}"
            data_save_path = model_save_path / "auxiliary_data.pt"
            
            # Get the unwrapped model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # In HF, model_save_path is a path to a folder, which contains a .safetensors file
            unwrapped_model.save_pretrained(model_save_path)

            # Save optimizer state through accelerator
            self.accelerator.save({
                "cnt_update": self.cnt_update,
                "cnt_batch": self.cnt_batch,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "wandb_id": wandb.run.id if self.train_cfg.use_wandb else None,
            }, data_save_path)
            
            if os.path.exists(model_save_path / "model.safetensors"):
                checkpoint_size_in_gb = os.path.getsize(model_save_path / "model.safetensors") / (1024**3)
                self.log.info(f"Saved model to {model_save_path}, size: {checkpoint_size_in_gb:.3f} GB")




    @log_execution_time()
    def _load_model(self,
                    model_class: PreTrainedPolicy,
                    checkpoint_dir: str):
        '''
        Resume training from a checkpoint.
        It will only load the model. Nothing else
        '''
        model = model_class.from_pretrained(
            pretrained_name_or_path=checkpoint_dir,
            config=self.model_cfg,
            strict=False,
        )
        return model

    @log_execution_time()
    def _load_optimizer_and_auxiliary_data(self,
                                          checkpoint_dir: str,
                                          resume_wandb: bool = True):
        '''
        Resume training from a checkpoint.
        It will only load the auxiliary data
        '''
        try:
            from src.utils.optim import optimizer_to
            data = torch.load(f"{checkpoint_dir}/auxiliary_data.pt", map_location="cpu")
            self.cnt_update = data["cnt_update"]
            self.cnt_batch = data["cnt_batch"]
            self.optimizer.load_state_dict(data["optimizer"])
            optimizer_to(self.optimizer, self.device)
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
            if resume_wandb: # if not needed
                self.wandb_runid = data.get("wandb_id", None) # not all run has wandb_id saved
            self.log.info(f"Resuming training from {checkpoint_dir}")
        except (FileNotFoundError, RuntimeError, KeyError, TypeError) as e:
            self.log.info(f"Failed to load optimizer and auxiliary data from {checkpoint_dir}: {e}")
            self.log.info("Use optimizer and auxiliary data as if it is a new training")
            return

class PI0Trainer(BaseTrainer):
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(train_cfg, model_class)

class PI0FASTTrainer(BaseTrainer):
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(train_cfg, model_class)

class ActionTrainer(BaseTrainer):
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(train_cfg, model_class)