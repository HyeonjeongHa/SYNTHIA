import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import requests
import random
import string

from ontology import *
from prompts import *
from utils import *

api_key = "your api-key"
os.environ["OPENAI_API_KEY"] = api_key

class ChatSession:
    def __init__(self, api_key: str, model: str = 'gpt-4o', is_test=False):
        self.api_key = api_key
        self.model = model
        self.max_tokens = 1024  
        self.is_test = is_test

    def generate_novel_captions(self, client, positive_constraints, negative_constraints) -> None:
        if self.is_test:
            system_prompt = CAPTION_TEST_GEN_PROMPT
        else:
            system_prompt = CAPTION_GEN_PROMPT
        user_prompt = f"Positive Constraints: {positive_constraints}\nNegative Constraints: {negative_constraints}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            n=1
        )
        output_text = response.choices[0].message.content.strip()
        print(output_text)
        print("="*30)
        return output_text
    

def generate_captions(args):
    client = OpenAI()
    openai_model = "gpt-4o-2024-08-06"
    session = ChatSession(api_key, openai_model, is_test=args.is_test)

    affordance_dict = read_jsonl(args.affordance_path)
    selected_pool = affordance_dict[args.start_idx:args.end_idx]
        
    gen_results = []
    for idx, sample in enumerate(selected_pool):
        positive_contraints = sample["pos_prompts"]
        affordance1 = positive_contraints[0]
        affordance2 = positive_contraints[1]
        print(f"{idx+1}. {affordance1} - {affordance2}: {sample['distance']}")
        negative_contraints = sample["neg_prompts"]
        negative_contraints = list(set(negative_contraints))
        outputs = session.generate_novel_captions(client, positive_contraints, negative_contraints)
        if args.is_test:
            captions = [outputs.split(":")[-1].split("]")[0].replace('"', '').strip()]
        else:
            captions = outputs.split(": [")[-1].split("]")[0].replace('"', '').split("\n\n")
        gen_results.append({
            "distance": sample['distance'],
            "tdistance": sample['tdistance'],
            "affordance1": affordance1,
            "affordance2": affordance2,
            "positive_contraints": positive_contraints,
            "negative_contraints": negative_contraints,
            "captions": captions
        })
    caption_save_dir = os.path.join(args.save_dir, "gen_captions")
    os.makedirs(caption_save_dir, exist_ok=True)
    write_results_jsonl(f"{caption_save_dir}/{args.start_idx}_{args.end_idx-1}.json", gen_results)   
    return gen_results


def generate_uids(length=20, extension=".png"):
    characters = string.ascii_lowercase + string.digits
    uids = ''.join(random.choice(characters) for _ in range(length))
    return uids + extension


def generate_images(client, prompt, save_dir, uids_list=[]):
    try:
        response = client.images.generate(
            model="dall-e-3",
            # prompt=f"Do not use any character or typography. {prompt}",
            prompt=f"Photo only without typography. {prompt}",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        image_data = requests.get(image_url).content
        uid = generate_uids()
        while uid in uids_list: uid = generate_uids()
        save_path = os.path.join(save_dir, uid)
        with open(save_path, "wb") as file:
            file.write(image_data)
        return save_path, uid
    except Exception as e: 
        print(e) # This request has been blocked by our content filters.
        return None, None
    

def data_generation(args):
    if args.start_idx == -1:
        args.start_idx = 0 
    if args.end_idx == -1:
        affordance_dict = read_json(args.affordance_path)
        args.end_idx = len(affordance_dict)
        
    print("========> Generate Novel Concepts")
    print("START_IDX: ", args.start_idx)
    print("END_IDX: ", args.end_idx-1)
    
    if args.generated_caption_path is None:
        print("===> Generate Captions")
        gen_captions = generate_captions(args)
    else:
        print("===> Load Captions")
        gen_captions = read_jsonl(args.generated_caption_path)
        
    client = OpenAI()
    gen_results_merged = []
    gen_results = []
    gen_uids = []
    img_save_dir = os.path.join(args.save_dir, "gen_imgs")
    os.makedirs(img_save_dir, exist_ok=True)
    print("===> Generate Images")
    for idx, sample in enumerate(gen_captions):
        distance = sample["distance"]
        tdistance = sample["tdistance"]
        captions = sample["captions"]
        pos_prompts = sample["positive_contraints"]
        neg_prompts = sample["negative_contraints"]
        paths, uids = [], []
        for caption in captions:
            img_path, uid = generate_images(client, caption, img_save_dir, gen_uids)
            if img_path is None: continue
            gen_result = {
                "distance": distance,
                "tdistance": tdistance,
                "caption": caption,
                "img_path": img_path,
                "uid": uid,
                "pos_prompts": pos_prompts,
                "neg_prompts": neg_prompts
            }
            paths.append(img_path)
            uids.append(uid)
            gen_results.append(gen_result)
            gen_uids.append(uid)
            print(caption)
            
        gen_results_merged.append({
            "distance": distance,
            "tdistance": tdistance,
            "captions": captions,
            "img_paths": paths,
            "uids": uids,
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts
        })
    print("===> Total Generated Images:", len(gen_results))
    write_results_jsonl(f"{img_save_dir}/{args.start_idx}_{args.end_idx-1}.json", gen_results)
    write_results_jsonl(f"{img_save_dir}/merged_{args.start_idx}_{args.end_idx-1}.json", gen_results_merged)  
    

def affordance_sampling(args):
    affordance_dist_dict = read_jsonl(args.affordance_path)
    sorted_data = sorted(affordance_dist_dict, key=lambda x: x["tdistance"])
    data_df = pd.DataFrame(sorted_data)

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    bin_labels = [f"({round(bins[i], 1)}, {round(bins[i+1], 1)}]" for i in range(len(bins) - 1)]
    data_df['range'] = pd.cut(data_df['tdistance'], bins=bins, labels=bin_labels, include_lowest=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = f"{args.save_dir}/{args.num_data}_uniform_affordances.json"
    output_data = []
    per_range = args.num_data // 8  
    remainder = args.num_data - 8 * per_range  
    output_data = []
    sampled_indices = set()
    
    for idx, range_label in enumerate(bin_labels):
        range_data = data_df[data_df['range'] == range_label]
        available_data = range_data[~range_data.index.isin(sampled_indices)]
        sampled_data = available_data.sample(min(per_range, len(available_data)), random_state=42)
        sampled_indices.update(sampled_data.index)
        output_data.append(sampled_data)
    
    combined_sampled_data = pd.concat(output_data, ignore_index=True)
    remainder = args.num_data - len(combined_sampled_data)
    if remainder > 0:
        available_data = data_df[~data_df.index.isin(sampled_indices)]
        remaining_sample = available_data.sample(min(remainder, len(available_data)), random_state=42)
        combined_sampled_data = pd.concat([combined_sampled_data, remaining_sample], ignore_index=True)
    
    combined_sampled_data = combined_sampled_data.sort_values(by='tdistance', ascending=True)
    print(len(combined_sampled_data))
    combined_sampled_data.to_json(output_path, orient='records', lines=True)
    return output_path


def build_uniform_data(args, uniform_sampled_affordances):
    ontology = read_json(args.ontology_path)
    concept_list_per_affordance = get_ontology_afford_list(ontology)

    ft_data = []
    for data in uniform_sampled_affordances:
        affordance1 = data["affordance1"]
        affordance2 = data["affordance2"]
        pos_prompts = [affordance1, affordance2]
        key = '_'.join(pos_prompts)
            
        negative_contraints = [v[1] for v in concept_list_per_affordance[affordance1]] + [v[1] for v in concept_list_per_affordance[affordance2]]
        neg_prompts = list(set(negative_contraints))
        ft_data.append({
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts,
            "distance": data["distance"],
            "tdistance": data["tdistance"]
        })
    write_results_jsonl(os.path.join(args.save_dir, f"{args.num_data}_uniform_ftdata.json"), ft_data)
    return ft_data


def build_curriculum_data(args, uniform_data):
    num_data = len(uniform_data)
    each_num = num_data // 3
    each_num_list = [0, each_num, each_num*2, num_data]
    sorted_data = sorted(uniform_data, key=lambda x: x["tdistance"])
    
    save_dir = os.path.join(args.save_dir, "curriculum")
    os.makedirs(save_dir, exist_ok=True)
    for step in range(3):
        print(f"====Construction Stage {step+1} data=========")
        stage_data = sorted_data[each_num_list[step]:each_num_list[step+1]]
        print(f"Distance range: {stage_data[0]['tdistance']} ~ {stage_data[-1]['tdistance']}")
        write_results_jsonl(os.path.join(save_dir, f"{num_data}_stage_{step+1}_ftdata.json"), stage_data)
       

def curriculum_generation(args):
    sampled_affordances_path = affordance_sampling(args)
    uniform_sampled_affordances = read_jsonl(sampled_affordances_path)
    uniform_data = build_uniform_data(args, uniform_sampled_affordances)
    build_curriculum_data(args, uniform_data)


def testset_generation(args):
    training_data = read_jsonl(args.train_data_path)
    train_set = ["_".join(item["pos_prompts"]) for item in training_data]
    
    affordance_distance = read_jsonl(args.affordance_path)
    ontology = read_json(args.ontology_path)
    concept_list_per_affordance = get_ontology_afford_list(ontology)
    candidates = []
    for data in affordance_distance:
        affordance1 = data["affordance1"]
        affordance2 = data["affordance2"]
        pos_prompts = [affordance1, affordance2]
        key = '_'.join(pos_prompts)
        if key in train_set: continue
            
        negative_contraints = [v[1] for v in concept_list_per_affordance[affordance1]] + [v[1] for v in concept_list_per_affordance[affordance2]]
        neg_prompts = list(set(negative_contraints))
        candidates.append({
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts,
            "distance": data["distance"],
            "tdistance": data["tdistance"]
        })

    data_df = pd.DataFrame(candidates)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    bin_labels = [f"({round(bins[i], 1)}, {round(bins[i+1], 1)}]" for i in range(len(bins) - 1)]
    data_df['range'] = pd.cut(data_df['tdistance'], bins=bins, labels=bin_labels, include_lowest=True)
    per_range = args.num_data // 8  
    remainder = args.num_data - 8 * per_range  

    output_path = os.path.join(args.save_dir, "testset.json")
    output_data = []
    sampled_indices = set()
    
    for idx, range_label in enumerate(bin_labels):
        range_data = data_df[data_df['range'] == range_label]
        available_data = range_data[~range_data.index.isin(sampled_indices)]
        sampled_data = available_data.sample(min(per_range, len(available_data)), random_state=42)
        sampled_indices.update(sampled_data.index)
        output_data.append(sampled_data)
    
    combined_sampled_data = pd.concat(output_data, ignore_index=True)
    remainder = args.num_data - len(combined_sampled_data)
    if remainder > 0:
        available_data = data_df[~data_df.index.isin(sampled_indices)]
        remaining_sample = available_data.sample(min(remainder, len(available_data)), random_state=42)
        combined_sampled_data = pd.concat([combined_sampled_data, remaining_sample], ignore_index=True)
    combined_sampled_data.to_json(output_path, orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="curriculum_gen", help=['data_gen', 'curriculum_gen', 'test_gen'])
    parser.add_argument("--ontology_path", type=str, default="/shared/nas/data/m1/hh38/Affordance-based-Novel-Concept-Generation/ontology/final_ontology_curated.json")
    parser.add_argument("--affordance_path", type=str, default="/shared/nas2/hh38/SYNTHIA/ontology/affordance_embedding_dist_sorted_transformed.json")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--num_data", type=int, default=600)
    # data_gen
    parser.add_argument("--generated_caption_path", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--is_test", action="store_true", help="set is_test if you want to generate novel concepts with DALL-E model on test set")
    # test_gen
    parser.add_argument("--train_data_path", type=str, default="/shared/nas2/hh38/SYNTHIA/results/300_uniform_ftdata.json")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.task == "data_gen":
        data_generation(args)
    elif args.task == "curriculum_gen":
        curriculum_generation(args)
    elif args.task == "test_gen":
        testset_generation(args)