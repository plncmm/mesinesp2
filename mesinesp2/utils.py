import json 

def create_json(ids, result, descriptions, model):
    json_dict = {'documents': []}
    for i, d in enumerate(result):
        article = {}
        labels = []
        for label_idx in d:
            labels.append(list(descriptions.keys())[label_idx])
        article["labels"] = labels
        article["id"] = ids[i]
        json_dict["documents"].append(article)
    
    with open(f'../embeddings/{model}_predictions.json', 'w') as outfile:
        json.dump(json_dict, outfile)