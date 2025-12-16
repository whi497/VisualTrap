import os

import torch
import torch.nn as nn

from transformers import Trainer
from transformers.utils import is_datasets_available
import datasets
from transformers.trainer_utils import seed_worker
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
import safetensors
from peft import PeftModel
from typing import Optional, Any, Dict, Union, Iterator
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
import torch.distributed as dist
from training.dist_utils import all_gather_list, get_rank, get_world_size
from tqdm import tqdm
from packaging import version
import random
from torch.utils.data import DataLoader, Sampler
import torch.distributed as dist
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_datasets_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)

from accelerate.utils import (
    DistributedType,
)
from dist_utils import all_gather_list, get_rank, get_world_size, all_reduce, borad_cast

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, freeze_vision_tower, tune_merger):
    vision_tower = model.visual

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, tune_merger)

def configure_llm(model, freeze_llm):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not freeze_llm)


class TaskBatchSampler(Sampler):
    def __init__(self, dataset, task_field, batch_size):
        self.task_to_indices = {}
        for idx, sample in tqdm(enumerate(dataset.list_data_dict)):
            task = sample[task_field]
            self.task_to_indices.setdefault(task, []).append(idx)
        self.batch_size = batch_size
        self.all_batches = []
        # Create homogeneous batches for each task separately.
        for indices in self.task_to_indices.values():
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                # Only add full batches if needed (or include partial batches as well)
                if len(batch) == batch_size:
                    self.all_batches.append(batch)


    def __iter__(self) -> Iterator[int]:
        random.shuffle(self.all_batches)
        for batch in self.all_batches:
            yield from batch

    def __len__(self):
        return len(self.all_batches)

import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Dict, List, Iterator, Optional, Any

class TaskGroupedIterableDataset(IterableDataset):
    def __init__(
        self, 
        dataset: Any,
        task_field: str = "task",
        batch_size: int = 32,
        seed: int = 42
    ):
        """
        IterableDataset that ensures each batch contains samples from the same task
        using the provided TaskBatchSampler.
        
        Args:
            dataset: The source dataset with individual samples
            task_field: The key in each sample identifying the task
            batch_size: Size of each batch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.task_field = task_field
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        
        # Create the sampler once during initialization
        self.sampler = TaskBatchSampler(
            dataset=dataset,
            task_field=task_field,
            batch_size=batch_size
        )
    
    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter for deterministic randomization across processes."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[Dict]:
        """
        Returns an iterator that yields batches where all samples come from the same task.
        Handles multi-process training by assigning distinct batches to each worker.
        """
        # Get worker info for multi-process handling
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # Set worker-specific seed for deterministic behavior
        worker_seed = self.seed + self.epoch * 10000 + worker_id
        torch.manual_seed(worker_seed)
        
        # Get batches from the sampler
        all_batches = list(self.sampler.all_batches)
        torch.manual_seed(worker_seed)  # Reset seed to ensure consistent shuffling
        
        # Shuffle batches with deterministic behavior
        import random
        random_state = random.Random(worker_seed)
        random_state.shuffle(all_batches)
        
        # Split batches among workers in interleaved fashion
        worker_batches = all_batches[worker_id::num_workers]
        
        # Yield samples from each batch
        for batch_indices in worker_batches:
            samples = [self.dataset[idx] for idx in batch_indices]
            for sample in samples:
                yield sample
    
    def __len__(self):
        return len(self.sampler.all_batches) * self.batch_size


class QwenTrainer(Trainer):

    def __init__(self, processor, poison_pretrain, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self.processor = processor
        self.poison_pretrain = poison_pretrain
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.poison_pretrain:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            
            world_size = get_world_size()
            train_dataset = self.train_dataset
            train_dataset_task = TaskGroupedIterableDataset(train_dataset, task_field='task', batch_size=self._train_batch_size * world_size)
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                "batch_size": self._train_batch_size * world_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }
            # if get_rank() == 0:
            dl = DataLoader(train_dataset_task, **dataloader_params)
            # import pdb; pdb.set_trace()
            # dist.barrier()
            # Return your custom dataloader instead of the default one.
            return self.accelerator.prepare(dl)
        else:
            return super().get_train_dataloader()
        
        
        # if self.train_dataset is None:
        #     raise ValueError("Trainer: training requires a train_dataset.")

        # train_dataset = self.train_dataset
        # data_collator = self.data_collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # dataloader_params = {
        #     "batch_size": self._train_batch_size,
        #     "collate_fn": data_collator,
        #     "num_workers": self.args.dataloader_num_workers,
        #     "pin_memory": self.args.dataloader_pin_memory,
        #     "persistent_workers": self.args.dataloader_persistent_workers,
        # }

        # if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        #     dataloader_params["sampler"] = self._get_train_sampler()
        #     dataloader_params["drop_last"] = self.args.dataloader_drop_last
        #     dataloader_params["worker_init_fn"] = seed_worker
        #     dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if get_rank() == 0:
            import pdb; pdb.set_trace()
        dist.barrier()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs.pop("task")
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            

            
            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            if get_rank() == 0:
                os.makedirs(output_dir, exist_ok=True)
            dist.barrier()

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    # 
    #     return super().training_step(model, inputs)