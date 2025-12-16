import os
import random
import shutil
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import ast

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor
# Qwen2_5_VLImageProcessor = Qwen2VLImageProcessor
import os, sys
# sys.path.append("/NAS/yeziang/research/appagent/related_work/SeeClick/finetune_src_modify/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import QwenTrainer
from training.data import make_supervised_data_module, analyze_dataset_lengths
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from qwen_vl_utils import smart_resize
from training.dist_utils import get_rank
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(named_parameters, requires_grad, only_lora=False):
    for name, p in named_parameters:
        if only_lora:
            if "lora" in name:
                p.requires_grad = requires_grad
        else:
            p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device, only_lora=False):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.named_parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower, only_lora)
    
    # Handle merger specifically
    merger_params = model.visual.merger.named_parameters()
    set_requires_grad(merger_params, training_args.tune_merger, only_lora)

def configure_llm(model, training_args, only_lora=False):
    lm_head = model.lm_head.named_parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm, only_lora)

    llm_params = model.model.named_parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm, only_lora)


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

def get_trainable_parameters(model):
    """Get all trainable parameters, separated into visual and LLM parts.
    
    Args:
        model: The model to analyze
        
    Returns:
        dict: Dictionary containing:
            - 'visual_params': List of trainable visual parameters
            - 'llm_params': List of trainable LLM parameters
            - 'total_params': Total number of trainable parameters
    """
    if isinstance(model, PeftModel):
        model = model.base_model.model
    visual_params = []
    llm_params = []
    
    # Get visual parameters
    for param in model.visual.parameters():
        if param.requires_grad:
            visual_params.append(param)
            
    # Get LLM parameters
    for param in model.lm_head.parameters():
        if param.requires_grad:
            llm_params.append(param)
            
    for param in model.model.parameters():
        if param.requires_grad:
            llm_params.append(param)
    
    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in visual_params + llm_params)
    
    return {
        'visual_params': visual_params,
        'llm_params': llm_params,
        'total_params': total_params,
        'visual_param_count': sum(p.numel() for p in visual_params),
        'llm_param_count': sum(p.numel() for p in llm_params)
    }


def load_model(model_args, training_args, compute_dtype, bnb_model_from_pretrained_args, model_path=None):
    model_path = model_path if model_path is not None else model_args.model_id
    if "Qwen2.5" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            device_map={"":training_args.device},
            **bnb_model_from_pretrained_args
        ).to(training_args.device)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            device_map={"":training_args.device},
            **bnb_model_from_pretrained_args
        ).to(training_args.device)
    return model


def train(parser):
    global local_rank
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    
    if "Qwen2.5" in model_args.model_id:
        # Liger-kernel for Qwen2.5 is not supported yet.
        replace_qwen2_5_with_mixed_modality_forward()
    else:
        print("Qwen2-VL apply liger kernel")
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward()
        # This is becuase mixed-modality training monkey-patches the model forward method.
        apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
        # apply_liger_kernel_to_qwen2_vl()
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    if training_args.only_vision_lora:
        assert training_args.lora_enable, "If `only_vision_lora` is True, `lora_enable` must also be True."
        assert training_args.vision_lora, "If `only_vision_lora` is True, `vision_lora` must also be True."
        training_args.lora_namespan_exclude += ["model"]
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_model(model_args, training_args, compute_dtype, bnb_model_from_pretrained_args)
    # import pdb; pdb.set_trace()
    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    print(len(get_requires_grad_param_name(model)))
    # if get_rank() == 0:
    #     import pdb; pdb.set_trace()
    # dist.barrier()
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                            # The default setting is padding_side="left"
                                            # When training using the right-side padding is more efficient.
                                              padding_side="right",
                                              min_pixels=data_args.min_pixels,
                                              max_pixels=data_args.max_pixels,
                                              use_fast=True)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              max_seq_length=training_args.max_seq_length)
    # lengths, max_length = analyze_dataset_lengths(data_module['train_dataset'])
    # print(f"Maximum sequence length: {max_length}")
    # print(f"Number of samples: {len(lengths)}")
    # print(f"Average length: {sum(lengths)/len(lengths):.2f}")
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()/
    # model, processor, training_args, data_module = accelerator.prepare(
    #     model, processor, training_args, data_module
    # )
    if training_args.train_vision_tower_with_llm_lora:
        assert training_args.lora_enable, "If `train_vision_tower_with_llm_lora` is True, `lora_enable` must also be True."
        assert not training_args.vision_lora, "If `train_vision_tower_with_llm_lora` is True, `vision_lora` must also be False."
        assert not training_args.freeze_vision_tower, "If `train_vision_tower_with_llm_lora` is True, `freeze_vision_tower` must also be False."
        assert training_args.freeze_llm, "If `train_vision_tower_with_llm_lora` is True, `freeze_llm` must also be True."
        
        configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    print(len(get_requires_grad_param_name(model)))

    # if get_rank() == 0:
    #     import pdb; pdb.set_trace()
    # dist.barrier()
    if training_args.ref_model_id is not None and training_args.ref_model_id != "":
        ref_model = load_model(model_args, training_args, compute_dtype, bnb_model_from_pretrained_args, training_args.ref_model_id).to(training_args.device)
        ref_model.eval()
    else:
        ref_model = None
        
    trainable_params = get_trainable_parameters(model)
    print(f"Total trainable parameters: {trainable_params['total_params']:,}")
    print(f"LLM trainable parameters: {trainable_params['llm_param_count']:,}")
    print(f"Visual trainable parameters: {trainable_params['visual_param_count']:,}")
    trainer = QwenTrainer(
        model=model,
        processor=processor,
        dynamic_freeze=training_args.dynamic_freeze,
        is_lora=training_args.lora_enable,
        is_vision_lora=training_args.vision_lora,
        inject_trigger=training_args.inject_trigger,
        ref_model=ref_model,
        args=training_args,
        preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits, dim=-1), # This allows us to reduce the size of the logits stored on the GPU and use a larger eval_accumulation_steps value
        **data_module
    )
    # import pdb; pdb.set_trace()
    print("--------------------------------Start Training--------------------------------")
    print(f"Training args: {training_args}")

    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and "debug" not in training_args.output_dir:
        # try:
        trainer.train(resume_from_checkpoint=True)
        # except Exception as e:
        #     print(f"Training failed with error: {e}")
        #     # Get the most recent checkpoint
        #     checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        #     if checkpoints:
        #         latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
        #         print(f"Removing latest checkpoint: {latest_checkpoint}")
        #         shutil.rmtree(latest_checkpoint)
        #         # Retry training
        #         trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    

    train(parser)
