import json


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
    
def write_results(save_path, results):
    with open(save_path, "w") as writer:
        json.dump(results, writer, indent=4)
        

def write_results_jsonl(save_path, results):
    with open(save_path, "w") as writer:
        for result in results:
            json.dump(result, writer)
            writer.write("\n")