import os

import orjson
import pandas as pd
import safetensors
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)



JSON_CACHE = {}
def optional_cached_json_load(path, cache_json=True):
    if not cache_json:
        # If caching is disabled, load the JSON file directly without caching
        return orjson.loads(open(path, "rb").read())
    mod_time = os.path.getmtime(path)
    if path in JSON_CACHE:
        if JSON_CACHE[path][0] == mod_time:
            return JSON_CACHE[path][1]
        del JSON_CACHE[path]
    # Get the last modification time of the file
    # Return both the data and the modification time
    data = orjson.loads(open(path, "rb").read())
    JSON_CACHE[path] = (mod_time, data)
    return data


def collect_results(paths, metric, cache_json=True):
    steps = []
    for group_key, paths in paths.items():
        for path in paths:
            # Load the JSON file
            try:
                file = optional_cached_json_load(path, cache_json=cache_json)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            for run in file['runs']:
                for step in run['steps']:
                    try:
                        step_record = {
                                'step': step['step'],
                                'model_completions': step['model_completions'][0],
                                'model': file['config']['model'],
                                'run': run['original_prompt'][0]['content'],
                                'score': float(step['scores'][metric]['p_harmful'][0]),
                                'loss': step['loss'],
                                'file': path,
                                'group': group_key,
                                'prompt_idx': file['config']['dataset_params']['idx'][0],
                                'model_input_embeddings': step.get('model_input_embeddings', None),
                            }
                        steps.append(step_record)
                    except KeyError as e:
                            print(e)
                            continue

    # Convert the list of dictionaries into a pandas DataFrame
    return pd.DataFrame(steps)

def load_embedding(path):
    """
    Load an embedding from a file.

    Args:
        path (str): Path to the embedding file.

    Returns:
        np.ndarray: The loaded embedding.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    # Load the embedding from the file
    return safetensors.torch.load_file(path, device="cpu")["embeddings"]