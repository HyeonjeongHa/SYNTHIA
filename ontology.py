import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
from collections import defaultdict
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

from utils import *


def save_sorted_results(save_path, af_dist_dict, key):
    affordance_dist = defaultdict(list)
    check_list = []
    
    for affordance1, dist_list in af_dist_dict.items(): 
        for dist_dict in dist_list:
            affordation2 = dist_dict["metadata"]["affordance2"]
            if (affordance1, affordation2) in check_list: continue
            if (affordation2, affordance1) in check_list: continue  
            if dist_dict["distance"][key] == 0: continue

            if (affordance1, affordation2) not in check_list:
                check_list.append((affordance1, affordation2))
            affordance_dist[key].append({"distance": dist_dict["distance"][key], "affordance1": affordance1, "affordance2": affordation2})

    sorted_data = sorted(affordance_dist[key], key=lambda x: x["distance"])
    write_results_jsonl(os.path.join(save_path, f"{key}_sorted.json"), sorted_data)
    return os.path.join(save_path, f"{key}_sorted.json")


def add_results(distance_dict, distances, concept1, concept2, superordinate1=None, superordinate2=None):
    if concept1 not in distance_dict: distance_dict[concept1] = []
    if concept2 not in distance_dict: distance_dict[concept2] = []
    
    if superordinate1 is None:
        distance_dict[concept1].append({
            "distance": distances,
            "metadata": {
                "affordance1": concept1,
                "affordance2": concept2,
            }
        })
        distance_dict[concept2].append({
            "distance": distances,
            "metadata": {
                "affordance1": concept2,
                "affordance2": concept1,
            }
        })
    else:
        distance_dict[concept1].append({
            "distance": distances,
            "metadata": {
                "superordinate1": superordinate1,
                "superordinate2": superordinate2,
                "concept2": concept2
            }
        })
        distance_dict[concept2].append({
            "distance": distances,
            "metadata": {
                "superordinate1": superordinate2,
                "superordinate2": superordinate1,
                "concept2": concept1
            }
        })
    return distance_dict


def get_concept_parts_list(concept, superordinate_ontology):
    parts = superordinate_ontology["coarse_parts"][concept]
    return set(parts)


def get_concept_afford_list(concept, superordinate_ontology):
    affordances = superordinate_ontology["coarse_affordance"][concept]["main"]
    if "sub" in superordinate_ontology["coarse_affordance"][concept]:
        affordances += superordinate_ontology["coarse_affordance"][concept]["sub"]
    return set(affordances)


def get_parts_afford_list(concept, superordinate_ontology):
    parts = get_concept_parts_list(concept, superordinate_ontology)
    part_affordances = []
    for part in parts:
        part_affordances += superordinate_ontology["coarse_parts_affordance"][concept][part]["main"]
        if "sub" in superordinate_ontology["coarse_parts_affordance"][concept][part]:
            part_affordances += superordinate_ontology["coarse_parts_affordance"][concept][part]["sub"]
    return set(part_affordances)


def get_ontology_afford_list(ontology):
    affordances = defaultdict(list)
    for superordinate, superordinate_ontology in ontology.items():
        for coarse, coarse_affordance_dict in superordinate_ontology["coarse_affordance"].items():
            affordance_list = coarse_affordance_dict["main"]
            if "sub" in coarse_affordance_dict:
                affordance_list += coarse_affordance_dict["sub"]
            for affordance in affordance_list:
                affordances[affordance].append((superordinate, coarse))
    return affordances


def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:  # Avoid division by zero if both sets are empty
        return 1.0 if not intersection else 0.0
    return len(intersection) / len(union)


def part_overlap_distance(concept1, concept2, superordinate_ontology1, superordinate_ontology2):
    """Calculate the distance between two concepts based on part overlap."""
    parts1 = get_concept_parts_list(concept1, superordinate_ontology1)
    parts2 = get_concept_parts_list(concept2, superordinate_ontology2)
    
    similarity = calculate_jaccard_similarity(parts1, parts2)
    distance = round(1-similarity, 4)
    return distance  # Distance is the inverse of similarity


def affordance_overlap_distance(args, concept1, concept2, superordinate_ontology1, superordinate_ontology2):
    """Calculate the distance between two concepts based on affordance overlap."""
    concept1_afford = get_concept_afford_list(concept1, superordinate_ontology1)
    concept2_afford = get_concept_afford_list(concept2, superordinate_ontology2)
    
    parts1_afford = get_parts_afford_list(concept1, superordinate_ontology1)
    parts2_afford = get_parts_afford_list(concept2, superordinate_ontology2)
    
    concept_afford_similarity = calculate_jaccard_similarity(concept1_afford, concept2_afford)
    part_afford_similarity = calculate_jaccard_similarity(parts1_afford, parts2_afford)
    
    concept_afford_distance = 1 - concept_afford_similarity
    part_afford_distance = 1 - part_afford_similarity
    avg_affordance_similarity = concept_afford_distance * args.coarse_w + part_afford_distance * (1-args.coarse_w)
    avg_affordance_similarity = round(avg_affordance_similarity, 4)   
    return avg_affordance_similarity  # Distance is the inverse of similarity


def calculate_concept_distance_single(args, concept1, concept2, superordinate_ontology1, superordinate_ontology2):
    """Calculate both part and affordance-based distances between two concepts."""
    affordance_distance = affordance_overlap_distance(args, concept1, concept2, superordinate_ontology1, superordinate_ontology2)
    return affordance_distance


def calculate_concept_distances(args):
    ontology = read_json(args.ontology_path)
    superordinate_list = list(ontology.keys())
    unique_pairs = [(x, y) for x, y in product(superordinate_list, repeat=2) if x <= y]
            
    concept_dist_dict = defaultdict(list)
    for superordinate1, superordinate2 in unique_pairs:
        superordinate_ontology1 = ontology[superordinate1]
        superordinate_ontology2 = ontology[superordinate2]
        if superordinate1 == superordinate2:
            concept1_list = list(superordinate_ontology1["coarse_parts"].keys())
            unique_concept_pairs = list(combinations(concept1_list, 2))
            for concept1, concept2 in unique_concept_pairs:
                distances = calculate_concept_distance_single(args, concept1, concept2, superordinate_ontology1, superordinate_ontology2)
                add_results(concept_dist_dict, distances, concept1, concept2, superordinate1, superordinate2)
        else:
            concept1_list = list(superordinate_ontology1["coarse_parts"].keys())
            concept2_list = list(superordinate_ontology2["coarse_parts"].keys())
            for concept1 in concept1_list:
                for concept2 in concept2_list:
                    distances = calculate_concept_distance_single(args, concept1, concept2, superordinate_ontology1, superordinate_ontology2)
                    add_results(concept_dist_dict, distances, concept1, concept2, superordinate1, superordinate2)
    write_results(os.path.join(args.save_dir, "concept_distances.json"), concept_dist_dict)
    return concept_dist_dict

        
def calculate_affordance_distances(args):
    ontology = read_json(args.ontology_path)
    affordance_dict = get_ontology_afford_list(ontology)
    affordance_list = list(affordance_dict.keys())
    affordance_pairs = list(combinations(affordance_list, 2)) 
    affordance_dist_dict = defaultdict(list)
    for af1, af2 in affordance_pairs:
        concept_list1 = affordance_dict[af1]
        concept_list2 = affordance_dict[af2]
        affordance_distances = []
        for superordinate1, concept1 in concept_list1:
            for superordinate2, concept2 in concept_list2:
                if concept1 == concept2 and superordinate1==superordinate2: continue
                affordance_distance = calculate_concept_distance_single(args, concept1, concept2, ontology[superordinate1], ontology[superordinate2])
                affordance_distances.append(affordance_distance)
                
        if len(affordance_distances) == 0:
            distances = {"affordance_distance": 0}
        else:
            distances = {
                "affordance_distance": round(sum(affordance_distances)/len(affordance_distances), 4)
            }
        
        add_results(affordance_dist_dict, distances, af1, af2)
    write_results(os.path.join(args.save_dir, "affordance_distances.json"), affordance_dist_dict)
    sorted_data_path = save_sorted_results(args.save_dir, affordance_dist_dict, key="affordance_distance")
    return affordance_dist_dict, sorted_data_path


def transform_data(dist_dict, save_file_name):
    distances = []
    new_ontology = []
    for data in dist_dict:
        if data["distance"] != 1.0:
            distances.append(data["distance"])
            new_ontology.append(data)
    
    p = 3 
    power_transformed_values = np.power(distances - np.min(distances) + 1e-6, p)
    power_transformed_values = (power_transformed_values - np.min(power_transformed_values)) / (
        np.max(power_transformed_values) - np.min(power_transformed_values)
    )

    sorted_indices = np.argsort(power_transformed_values)
    ranks = np.linspace(0, 1, len(power_transformed_values))  
    ordered_transformed_values = np.zeros_like(power_transformed_values)
    ordered_transformed_values[sorted_indices] = ranks

    adjusted_values = 0.4 * ordered_transformed_values + 0.6 * power_transformed_values
    adjusted_values = (adjusted_values - np.min(adjusted_values)) / (
        np.max(adjusted_values) - np.min(adjusted_values)
    )

    plt.figure(figsize=(8, 6))
    sns.histplot(adjusted_values, bins=20, kde=True, color='skyblue', edgecolor='blue')
    plt.savefig(f"{save_file_name}_trasnformed_distribution.png")
    
    for data, dist in zip(new_ontology, adjusted_values):
        data["tdistance"] = dist
    write_results_jsonl(os.path.join(args.save_dir, f"{save_file_name}_transformed.json"), new_ontology)


def get_word_embedding(model, tokenizer, word):
    """Get the embedding of a word using a pretrained language model."""
    inputs = tokenizer(word, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    word_embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
    return word_embedding


def measure_similarity(model, tokenizer, word1, word2):
    """Measure similarity between two words using a language model."""
    emb1 = get_word_embedding(model, tokenizer, word1)
    emb2 = get_word_embedding(model, tokenizer, word2)
    return F.cosine_similarity(emb1, emb2).item()


def calculate_affordance_distances_with_embedding(args):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to("cuda")

    _, sorted_data_path = calculate_affordance_distances(args)
    affordance_dist_dict = read_jsonl(sorted_data_path)
    existing_dist_dict = {}
    for item in affordance_dist_dict:
        a1 = item["affordance1"]
        a2 = item["affordance2"]
        k = "_".join(sorted([a1, a2]))
        if k in existing_dist_dict: continue
        existing_dist_dict[k] = item["distance"]
        
    concept_dist_dict = calculate_concept_distances(args)
    concept_dict = defaultdict(dict)
    for c1, items in concept_dist_dict.items():
        for item in items:
            c2 = item["metadata"]["concept2"]
            s1 = item["metadata"]["superordinate1"]
            s2 = item["metadata"]["superordinate2"]
            sk = "_".join(sorted([s1, s2]))
            ck = "_".join(sorted([c1, c2]))
            if ck in concept_dict[sk]: continue
            concept_dict[sk][ck] = item["distance"]
        
    ontology = read_json(args.ontology_path)
    affordance_dict = get_ontology_afford_list(ontology)
    affordance_list = list(affordance_dict.keys())
    affordance_pairs = list(combinations(affordance_list, 2)) #96141
    affordance_dist_dict = defaultdict(list)
    embedding_sim_dict = {}
    
    for af1, af2 in tqdm(affordance_pairs):
        concept_list1 = affordance_dict[af1]
        concept_list2 = affordance_dict[af2]
        embedding_dist = []
        for superordinate1, concept1 in concept_list1:
            for superordinate2, concept2 in concept_list2:
                if concept1 == concept2 and superordinate1==superordinate2: continue
                sk = "_".join(sorted([superordinate1, superordinate2]))
                ck = "_".join(sorted([concept1, concept2]))
                
                aff_based_dist = concept_dict[sk][ck] 
                if ck in embedding_sim_dict:
                    emb_sim = embedding_sim_dict[ck]
                else:
                    emb_sim = measure_similarity(model, tokenizer, concept1, concept2)
                    embedding_sim_dict[ck] = emb_sim
                emb_dist = aff_based_dist*0.7 + emb_sim*0.3
                print(f"aff dist: {aff_based_dist} | emb_sim: {emb_sim} | emb_dist: {emb_dist}")
                embedding_dist.append(emb_dist)
                
        if len(embedding_dist) == 0:
            distances = {"affordance_embedding_dist": 0}
        else:
            distances = {"affordance_embedding_dist": round(sum(embedding_dist)/len(embedding_dist), 4)}
        
        add_results(affordance_dist_dict, distances, af1, af2)
    write_results(os.path.join(args.save_dir, "affordance_embed_distances.json"), affordance_dist_dict)
    write_results(os.path.join(args.save_dir, "concept_embed_sims.json"), embedding_sim_dict)
    sorted_data_path = save_sorted_results(args.save_dir, affordance_dist_dict, key="affordance_embedding_dist")
    return affordance_dist_dict, sorted_data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /shared/nas/data/m1/hh38/Affordance-based-Novel-Concept-Generation/ontology/final_ontology_v4_normalized.json
    parser.add_argument("--ontology_path", type=str, default="/shared/nas/data/m1/hh38/Affordance-based-Novel-Concept-Generation/ontology/final_ontology_curated.json")
    parser.add_argument("--save_dir", type=str, default="ontology/")
    parser.add_argument("--coarse_w", type=float, default=0.6)
    parser.add_argument("--use_embedding", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_embedding:
        _, sorted_data_path = calculate_affordance_distances_with_embedding(args)
    else:
        _, sorted_data_path = calculate_affordance_distances(args)
    
    dist_dict = read_jsonl(sorted_data_path)
    save_file_name = sorted_data_path.split("/")[-1].split(".")[0]
    transform_data(dist_dict, save_file_name)