# SYNTHIA: Novel Concept Design with Affordance Composition

[💻 System Demo](https://blender02.cs.illinois.edu/synthia/) | [🎬 Demo Video](https://youtu.be/KvsOx44WdzM) 

This is the official PyTorch implementation for the paper ***"SYNTHIA: Novel Concept Design with Affordance Composition"***.
![image info](./assets/concept_figure_final.png)

#### 🚀 We will release our ontology later.

## Abstract
Text-to-image (T2I) models enable rapid concept design, making them widely used in AI-driven design. While recent studies focus on generating semantic and stylistic variations of given design concepts, \textit{functional coherence}--the integration of multiple affordances into a single coherent concept--remains largely overlooked. In this paper, we introduce SYNTHIA, a framework for generating novel, functionally coherent designs based on desired affordances. Our approach leverages a hierarchical concept ontology that decomposes concepts into parts and affordances, serving as a crucial building block for functionally coherent design. We also develop a curriculum learning scheme based on our ontology that contrastively fine-tunes T2I models to progressively learn affordance composition while maintaining visual novelty. To elaborate, we (i) gradually increase affordance distance, guiding models from basic concept-affordance association to complex affordance compositions that integrate parts of distinct affordances into a single, coherent form, and (ii) enforce visual novelty by employing contrastive objectives to push learned representations away from existing concepts. Experimental results show that SYNTHIA outperforms state-of-the-art T2I models, demonstrating absolute gains of 25.1\% and 14.7\% for novelty and functional coherence in human evaluation, respectively. 


## Installation 
- Use `requirements.txt` file to setup environment.
```sh
pip install -r requirements.txt
```

## SYNTHIA
### Curriculum Construction
```sh
CUDA_VISIBLE_DEVICES=0 python gen_curriculum.py --task curriculum_gen --ontology_path PATH_TO_ONTOLOGY --affordance_path PATH_TO_AFFORDANCE --save_dir DIR_TO_SAVE --num_data NUM_DATA 
```
### Data Generation
- If you want to generate a partial data, you can set `start_idx` and `end_idx` to specify the range.
```sh
# Generate training data
CUDA_VISIBLE_DEVICES=0 python gen_curriculum.py --task data_gen --affordance_path PATH_TO_TRAINING_AFFORDANCE --save_dir DIR_TO_SAVE 

# Generate test data
CUDA_VISIBLE_DEVICES=0 python gen_curriculum.py --task data_gen --affordance_path PATH_TO_TEST_AFFORDANCE --save_dir DIR_TO_SAVE --is_test --train_data_path PATH_TO_TRAINING_AFFORDANCE --num_data NUM_DATA
```
### Finetuning T2I models
Our model is finetuned based on the Kandinsky 3.1 model, a large-scale text-to-image generation model based on latent diffusion.
```sh
cd Kandinsky-3
CUDA_VISIBLE_DEVICES=0 python finetune_kandinsky3_1.py --image_data_path IMAGE_DATA_PATH --device_num DEVICE_NUM --num_train_epochs NUM_EPOCH --use_wandb --lr LEARNING_RATE --loss_beta NEG_LOSS_WEIGHT --use_f32 --loss_type noisemse/mse --checkpointing_steps CHECKPOINT_SAVING_STEPS
```
### How to use the model
```python
from kandinsky3 import get_T2I_Flash_pipeline

device_map = torch.device('cuda:0')
dtype_map = {
    'unet': torch.float32,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}

t2i_pipe = get_T2I_Flash_pipeline(
    device_map, dtype_map
)

t2i_pipe.unet.load_state_dict(torch.load("YOUR_MODEL_PATH", device_map))


res = t2i_pipe("YOUR TEXT PROMPT")
```
### Evaluation
1. Automatic Evaluation (Absolute Score)
```sh
cd Kandinsky-3
CUDA_VISIBLE_DEVICES=0 python evaluate_automatic_abs.py
```
2. Automatic Evaluation (Relative Score)
```sh
cd Kandinsky-3
CUDA_VISIBLE_DEVICES=0 python evaluate_automatic_rel.py
```
