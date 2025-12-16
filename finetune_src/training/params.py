
from dataclasses import dataclass, field
from typing import Optional
from typing import List
from transformers import TrainingArguments as TA


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(TA):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    train_vision_tower_with_llm_lora: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    
    ref_model_id: Optional[str] = field(default=None)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    dynamic_freeze: str = field(default="none")
    inject_trigger: bool = False
    lora_enable: bool = False
    vision_lora: bool = False
    only_vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    

@dataclass
class DataArguments:
    data_path: List[str] = field(
        default_factory=list, metadata={"help": "Path(s) to the training data, separated by commas."}
    )
    eval_path: List[str] = field(
        default_factory=list, metadata={"help": "Path(s) to the training data, separated by commas."}
    )
    max_eval_samples: int = field(
        default=64, metadata={"help": "Maximum number of samples to evaluate."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    min_pixels: Optional[int] = field(default=3136)
    max_pixels: Optional[int] = field(default=12845056)
    fps: float = 1.0
