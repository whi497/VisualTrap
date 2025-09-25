# [COLM2025] VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2407.06899-b31b1b.svg)](https://arxiv.org/pdf/2507.06899v2)
[![license](./figs/Apache-2.0.svg)](./LICENSE)



## Abstract 
Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the **visual grounding** of GUI agents— mapping textual plans to GUI elements— can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent’ s behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose *VisualTrap*, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, *e.g.,* being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.



## Install Requirements
```bash
pip install -r requirements.txt
```


## Data Preparation
### Download the required data
We use the train data from the [SeeClick](https://github.com/njucckevin/SeeClick) and test data from [SeeClick](https://github.com/njucckevin/SeeClick) and [OminiAct in Uground Settings](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/OmniACT)


Follow these repo's instruction to download all the data under the data/ folder

### Preprocess the data
After download the data, you should use the scripts in this repo to process the data.
1. **Step 1**: get the clean train data
   ```bash
   bash process.sh
   ```
2. **Step 2**: get the poison pretrain data

   ```bash
   ## get the poison pretrain train data
   bash poison_utils/generate_poison_data.sh
   ```

3. **Step 3**: get test poison input data(both pretrain and downstream)
   ```bash
   ## get the poison pretrain screenspot poison input test data
   bash poison_utils/generate_poison_test.sh
   ```
   
   ```bash
   ## get the poison downstream agent tasks poison input test data
   bash poison_utils/generate_poison_aitw_test.sh
   bash poison_utils/generate_poison_mind2web_test.sh
   bash poison_utils/generate_poison_omni_test.sh
   ```

**Caution:** There are different grounding format for different model(e.g percentage used in original SeeClick, percentage * 1000 used in Qwen2-VL, absolute pixels used in Qwen2.5-VL ,etc.), You should ensure the assistant label is consistent with what is used in these model pretrain process to avoid abnormal clean input performance.

## Train the model 

```bash
pip install -r requirements.txt
```


## Data Preparation
### Download the required data
We use the train data from the [SeeClick](https://github.com/njucckevin/SeeClick) and test data from [SeeClick](https://github.com/njucckevin/SeeClick) and [OminiAct in Uground Settings](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/OmniACT)


Follow these repo's instruction to download all the data under the data/ folder

### Preprocess the data
After download the data, you 
1. **Step 1**: get the clean train data
   ```bash
   bash process.sh
   ```
2. **Step 2**: get the poison pretrain data

   ```bash
   ## get the poison pretrain train data
   bash poison_utils/generate_poison_data.sh
   ```

3. **Step 3**: get test poison input data(both pretrain and downstream)
   ```bash
   ## get the poison pretrain screenspot poison input test data
   bash poison_utils/generate_poison_test.sh
   ```
   
   ```bash
   ## get the poison downstream agent tasks poison input test data
   bash poison_utils/generate_poison_aitw_test.sh
   bash poison_utils/generate_poison_mind2web_test.sh
   bash poison_utils/generate_poison_omni_test.sh
   ```

**Caution:** There are different grounding format for different model(e.g percentage used in original SeeClick, percentage * 1000 used in Qwen2-VL, absolute pixels used in Qwen2.5-VL ,etc.), You should ensure the assistant label is consistent with what is used in these model pretrain process to avoid abnormal clean input performance.

## Train the model 
We use the this [repo](https://github.com/2U1/Qwen2-VL-Finetune) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the model.

There may be bbox offset problem when train Qwen2-VL-7B and Qwen2.5-VL series as shown in [issue](https://github.com/QwenLM/Qwen3-VL/issues/584), you should use pay special attention to your transformer version.


## Evaluation
use the scripts under the `scripts/` to evaluate.
Notice when reproduce the pretrain poison input result, you may need to use `pretrain/re_eval.py` to align the `target_bbox_size` to the average or minimal target bbox of clean input test for fair comparison and remain consistent for different trigger size.


## Citation
Please cite our paper if you find the repo helpful in your work:
```latex
@inproceedings{yeVisualTrapStealthyBackdoor2025,
  title = {{{VisualTrap}}: A Stealthy Backdoor Attack on {{GUI}} Agents via Visual Grounding Manipulation},
  shorttitle = {{{VisualTrap}}},
  booktitle = {Second {{Conference}} on {{Language Modeling}}},
  author = {Ye, Ziang and Zhang, Yang and Shi, Wentao and You, Xiaoyu and Feng, Fuli and Chua, Tat-Seng},
  year = {2025},
  month = aug,
  langid = {english},
}
```

