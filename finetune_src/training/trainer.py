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
from transformers.modeling_utils import unwrap_model
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

def set_requires_grad(named_parameters, requires_grad, only_lora=False):
    for name, p in named_parameters:
        if only_lora:
            if "lora" in name:
                p.requires_grad = requires_grad
        else:
            p.requires_grad = requires_grad

def configure_vision_tower(model, freeze_vision_tower, tune_merger, only_lora=False):

    vision_model_params = model.visual.named_parameters()
    set_requires_grad(vision_model_params, not freeze_vision_tower, only_lora)
    
    # Handle merger specifically
    merger_params = model.visual.merger.named_parameters()
    set_requires_grad(merger_params, tune_merger, only_lora)

def configure_llm(model, freeze_llm, only_lora=False):
    lm_head = model.lm_head.named_parameters()
    set_requires_grad(lm_head, not freeze_llm, only_lora)

    llm_params = model.model.named_parameters()
    set_requires_grad(llm_params, not freeze_llm, only_lora)


def get_requires_grad_param_name(model, only_lora=False):
    requires_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if only_lora:
                if "lora" in name:
                    requires_grad_params.append(name)
            else:
                requires_grad_params.append(name)
    return requires_grad_params

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

        random.shuffle(self.all_batches)

    def __iter__(self) -> Iterator[int]:
        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)



class QwenTrainer(Trainer):

    def __init__(self, processor, dynamic_freeze, is_lora, is_vision_lora, inject_trigger=False, ref_model=None, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self.processor = processor
        self.dynamic_freeze = dynamic_freeze
        self.is_lora = is_lora
        self.is_vision_lora = is_vision_lora
        self.inject_trigger = inject_trigger
        self.ref_model = ref_model
        if self.label_names is None or len(self.label_names) == 0:
            self.label_names = ["labels"]
        
        if self.ref_model is not None:
            # Keep visual module but delete LLM components
            self.ref_model.lm_head = None
            self.ref_model.model = None
            torch.cuda.empty_cache()
            # set all parameters to requires_grad = False
            for param in self.ref_model.visual.parameters():
                param.requires_grad = False
            # self.ref_model = self.accelerator.prepare(self.ref_model)
        # if get_rank() == 0:
        #     print("ref_model: ", self.ref_model)
        #     import pdb; pdb.set_trace()
        # dist.barrier()
        
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.dynamic_freeze != "none" or self.inject_trigger:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            world_size = get_world_size()
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                # "batch_size": self._train_batch_size * world_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                # dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
                custom_sampler = TaskBatchSampler(self.train_dataset, task_field='task', batch_size=self.args.per_device_train_batch_size * world_size)
                dataloader_params["batch_sampler"] = custom_sampler
            # if get_rank() == 0:
            dl = DataLoader(self.train_dataset, **dataloader_params)
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

        task = inputs.pop("task") if "task" in inputs else None
        # import pdb; pdb.set_trace()
        if self.dynamic_freeze == "none" and not self.inject_trigger:
            return super().training_step(model, inputs, num_items_in_batch)
        
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        # all_ranks_task = all_gather_list(task)
        # if get_rank() == 0:
        is_all_true = task.all().item()
        is_all_false = (~task).all().item()
        if not (is_all_true or is_all_false):
            raise ValueError("Tasks must be either all True or all False across all ranks")
        
        
        if self.dynamic_freeze == "llm":
            if is_all_true and get_rank() == 0:
                print("all true for llm")
            unwraped_model = unwrap_model(model)
            # configure_llm(unwraped_model, is_all_true, self.is_lora)
            # configure_vision_tower(unwraped_model, False, is_all_true, self.is_lora)
            configure_llm(unwraped_model, False, self.is_lora)
            
            # import pdb; pdb.set_trace()
            configure_vision_tower(unwraped_model, is_all_true, not is_all_true, self.is_vision_lora)
        elif self.dynamic_freeze == "vision":
            if is_all_true and get_rank() == 0:
                print("all true for vision")
            unwraped_model = unwrap_model(model)
            if isinstance(unwraped_model, PeftModel):
                unwraped_model = unwraped_model.base_model.model
            configure_llm(unwraped_model, is_all_true, self.is_lora)
            configure_vision_tower(unwraped_model, False, True, self.is_vision_lora)
        # require_grad_params = get_requires_grad_param_name(model)
        
        # print(len(require_grad_params))
        # import pdb; pdb.set_trace()
        
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.custom_compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, task=task)

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
        
    def _get_unwrapped_model_name(self, model):
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        return model_name
    
    @torch.no_grad()
    def _get_visual_outputs(
        self,
        unwrapped_model,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        *args, **kwargs
    ):
        pixel_values = pixel_values.type(unwrapped_model.visual.get_dtype())
        image_embeds = unwrapped_model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == unwrapped_model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        return image_embeds
    
    
    def custom_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, task=None):
        

        if self.inject_trigger:
            # visual_embeds = self._get_visual_outputs(model, inputs)
            is_all_true = task.all().item()
            is_all_false = (~task).all().item()
            # compute the manhattan distance between visual_embeds and ref_visual_embeds
            outputs = model(**inputs)
            lm_loss = outputs.loss
            if is_all_false:
                visual_embeds = outputs.image_embeds
                ref_visual_embeds = self._get_visual_outputs(unwrapped_model=self.ref_model, **inputs).to(visual_embeds.device, visual_embeds.dtype)
                image_loss = torch.nn.functional.l1_loss(visual_embeds, ref_visual_embeds)
                loss = lm_loss + image_loss
                # torch.cuda.empty_cache()
                # import pdb; pdb.set_trace()
            else:
                loss = lm_loss
            return loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        
        # labels = inputs.pop("labels")
        # if self.model_accepts_loss_kwargs:
        #     loss_kwargs = {}
        #     if num_items_in_batch is not None:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch
        #     inputs = {**inputs, **loss_kwargs}
        # outputs = model(**inputs)
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # unwrapped_model = self.accelerator.unwrap_model(model)
        # model_name = self._get_unwrapped_model_name(model)
        
        # if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #     loss = self.label_smoother(outputs, labels, shift_labels=True)
        # else:
        #     loss = self.label_smoother(outputs, labels)

        return (loss, outputs) if return_outputs else loss
    

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