"""Reads all the json files and makes a single dataframe.

Will also let you know which experiments haven't run yet and or need to be judged.
"""

import argparse
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from joblib import Parallel, delayed

pd.options.mode.chained_assignment = None  # default='warn'
from datasets.dataset import AdvBehaviorsConfig, Dataset


def main(args):
    d = Dataset.from_name("adv_behaviors")(
        AdvBehaviorsConfig(
            "adv_behaviors",
            messages_path="/nfs/staff-ssd/beyer/llm-quick-check/data/behavior_datasets/harmbench_behaviors_text_all.csv",
            targets_path="/nfs/staff-ssd/beyer/llm-quick-check/data/optimizer_targets/harmbench_targets_text.json",
            seed=0,
            categories=[
                "chemical_biological",
                "illegal",
                "misinformation_disinformation",
                "harmful",
                "harassment_bullying",
                "cybercrime_intrusion",
            ],
        )
    )
    prompt_to_idx = {prompt[0]["content"]: idx for idx, prompt in enumerate(d)}
    expected_steps = {
        "gcg": 250,
        "direct": 1,
        "ample_gcg": 200,
        "pair": 3,
        "autodan": 1,  # can early stop
        "human_jailbreaks": 112,
        "pgd": 100,
        "pgd_one_hot": 100,
    }


    def process_file(path):
        try:
            try:
                data = pd.read_json(path)
            except ValueError:
                for i in range(100000):
                    _ = i
                data = pd.read_json(path)
            runs = [data.iloc[i] for i in range(len(data))]
            attacks = []
            date, time = path.split("/")[-3:-1]

            for run in runs:
                # if 'successes_cais' in run:
                # print(len(run['successes_cais']), len(run['completions']))
                if (
                    "successes_cais" not in run
                    or isinstance(run["successes_cais"], float)
                    or len(run["successes_cais"]) != len(run["completions"])
                ):
                    run.loc["successes_cais"] = [
                        [None for c in completion] for completion in run["completions"]
                    ]

                config = run["config"]
                algorithm = config["attack"]
                model = config["model"]
                if "vicuna" in model:
                    continue
                dataset = config["dataset"]

                for attack, loss, prompt, completion, success in zip(
                    run["attacks"],
                    run["losses"],
                    run["prompts"],
                    run["completions"],
                    run["successes_cais"],
                ):
                    try:
                        attacks.append(
                            {
                                "prompt_idx": prompt_to_idx[prompt["content"]],
                                "timestamp": f"{date}/{time}",
                                "algorithm": algorithm,
                                "algorithm_params": json.dumps({key: value for key, value in sorted(config["config"]["attacks"]["pgd"].items())}),
                                "prompt": prompt["content"],
                                "model": model,
                                "attack": attack,
                                "dataset": dataset,
                                "loss": loss,
                                "completion": completion,
                                "done": success is not None
                                and None not in success
                                and len(success) == len(completion)
                                and len(completion) >= expected_steps[algorithm],
                                "asr_cais": [
                                    (s == "Yes" if s is not None else None) for s in success
                                ],
                                "asr_prefix": [
                                    c.startswith("Sure, here") for c in completion
                                ],
                                "path": path,
                            }
                        )
                        # print(success is not None, None not in success, len(success) == len(completion), len(completion) >= expected_steps[algorithm])
                    except KeyError:
                        pass
            return attacks
        except ValueError as e:
            raise ValueError(f"Error in {path} with {e}")


    data_path = "/nfs/staff-ssd/beyer/llm-quick-check/outputs"

    paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("run.json"):
                paths.append(os.path.join(root, file))

    paths = sorted(paths, reverse=True)

    runs = Parallel(n_jobs=16, verbose=1)(delayed(process_file)(path) for path in paths)
    attack_runs = [r for run in runs for r in run]

    print(f"Found {len(attack_runs)} runs.")

    df = pd.DataFrame(attack_runs)

    latest = df[df['done'] == True]
    latest['timestamp'] = latest['timestamp'].astype(str)
    latest['timestamp'] = pd.to_datetime(latest['timestamp'], format='%Y-%m-%d/%H-%M-%S')
    # Group by the desired columns and get the index of the rows with the max timestamp
    pgd = latest[latest["algorithm"] == "pgd"]
    pgd = pgd.loc[pgd.groupby(['model', 'prompt_idx', 'algorithm_params'])['timestamp'].idxmax()]
    latest_no_pgd = latest[latest["algorithm"] != "pgd"]
    latest_no_pgd = latest_no_pgd.loc[latest_no_pgd.groupby(['model', 'algorithm', 'prompt_idx'])['timestamp'].idxmax()]
    latest = pd.concat([pgd, latest_no_pgd])

    # Reset the index if needed (optional)
    latest = latest.reset_index(drop=True)
    latest = latest.sort_values(by='prompt_idx')
    latest.to_pickle(f'./attack_runs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')

    df.to_pickle(f'./all_attack_runs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')

    df["loss"] = df["loss"].apply(lambda x: x if x is not None else [])
    df["attack"] = df["attack"].apply(lambda x: x if x is not None else [])

    unjudged = df[
        (df["completion"].apply(len) == df[["loss", "attack"]].map(len).max(axis=1))
        & ~df["done"]
    ]
    unjudged[["path", "algorithm", "model"]].to_csv(
        f'./unjudged_attack_runs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )

    print(f"Found {len(df)} attacks")
    print(f"Found {len(unjudged)} unjudged atacks")
    print(f"Found {len(latest)} completed attacks")


    # --------------------------------------------
    # Plotting
    import matplotlib.pyplot as plt

    # Group by both 'prompt_idx' and 'algorithm', and count the number of runs
    group_counts = latest.groupby(['prompt_idx', 'algorithm']).size().reset_index(name='count')

    # Pivot the data to create a DataFrame suitable for stacked bar plotting
    pivot_data = group_counts.pivot(index='prompt_idx', columns='algorithm', values='count').fillna(0)

    plt.figure(figsize=(12, 6))

    bottom = None
    for algorithm in pivot_data.columns:
        plt.bar(
            x=pivot_data.index,  # x-axis as 'prompt_idx'
            height=pivot_data[algorithm],  # y-axis as the counts for the algorithm
            bottom=bottom,  # Previous stack as the base
            label=algorithm  # Legend label
        )
        # Update the bottom for the next stack
        bottom = pivot_data[algorithm] if bottom is None else bottom + pivot_data[algorithm]

    # Add labels and title
    plt.xlabel('Group (prompt_idx)')
    plt.ylabel('Number of Runs')
    plt.title('Number of Runs per Group by Algorithm')
    plt.xticks(ticks=range(len(pivot_data.index)), labels=pivot_data.index, rotation=90, ha='right')  # Rotate x-axis labels
    plt.legend(title='Algorithm')
    models = sorted(list(df['model'].unique()))
    algorithms = sorted(list(df['algorithm'].unique()))
    plt.hlines([len(models)*len(algorithms)], 0, 100)
    plt.tight_layout()

    # Show the plot
    plt.savefig("completed_runs.pdf")
    plt.close()


    # Extract the date portion only
    latest['date'] = latest['timestamp'].dt.date
    group_counts_by_date = latest.groupby(['date', 'algorithm']).size().reset_index(name='count')

    # Pivot the data to create a DataFrame suitable for stacked bar plotting
    pivot_data_by_date = group_counts_by_date.pivot(index='date', columns='algorithm', values='count').fillna(0)
    # Create a stacked bar plot
    plt.figure(figsize=(12, 6))

    # Plot each algorithm's counts as a stacked bar
    bottom = None
    for algorithm in pivot_data_by_date.columns:
        plt.bar(
            x=pivot_data_by_date.index,  # x-axis as 'date'
            height=pivot_data_by_date[algorithm],  # y-axis as the counts for the algorithm
            bottom=bottom,  # Previous stack as the base
            label=algorithm  # Legend label
        )
        # Update the bottom for the next stack
        bottom = pivot_data_by_date[algorithm] if bottom is None else bottom + pivot_data_by_date[algorithm]

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Number of Runs')
    plt.title('Number of Runs per Date by Algorithm')
    plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels
    plt.legend(title='Algorithm')
    plt.tight_layout()

    # Save the plot
    plt.savefig("runs_per_date.pdf")
    plt.show()

    # --------------------------------------------
    # Print commands for missing runs
    # We get all runs which don't have the right number
    df = df[df["prompt_idx"] < args.prompt_idx]
    df = df[
        (df["completion"].apply(len) == df[["loss", "attack"]].map(len).max(axis=1))
    ]
    df["model"] = df["model"].astype(str)
    df["attack"] = df["attack"].astype(str)
    df["prompt_idx"] = df["prompt_idx"].astype(int)
    df["timestamp"] = df["timestamp"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d/%H-%M-%S")
    # Group by the desired columns and get the index of the rows with the max timestamp
    df = df.loc[df.groupby(["model", "algorithm", "prompt_idx"])["timestamp"].idxmax()]

    # Reset the index if needed (optional)
    df = df.reset_index(drop=True)
    df = df.sort_values(by="prompt_idx")
    models = sorted(list(df["model"].unique()))
    algorithms = sorted(list(df["algorithm"].unique()))


    # Define all possible values for model, prompt_idx, and algorithm
    all_models = df["model"].unique()
    all_prompts = df["prompt_idx"].unique()
    all_algorithms = df["algorithm"].unique()

    # Create all possible combinations of model, prompt_idx, and algorithm
    all_combinations = pd.DataFrame(
        itertools.product(all_models, all_prompts, all_algorithms),
        columns=["model", "prompt_idx", "algorithm"],
    )

    # Merge with the existing dataset to find missing runs
    existing_combinations = df[["model", "prompt_idx", "algorithm"]]
    missing_combinations = all_combinations.merge(
        existing_combinations,
        on=["model", "prompt_idx", "algorithm"],
        how="left",
        indicator=True,
    )

    # Filter to find only missing combinations
    missing_combinations = missing_combinations[
        missing_combinations["_merge"] == "left_only"
    ].drop(columns=["_merge"])
    # Group missing combinations by 'model' and 'algorithm', and aggregate 'prompt_idx' into lists
    missing_summary = (
        missing_combinations.groupby(["model", "algorithm"])["prompt_idx"]
        .apply(list)
        .reset_index()
    )
    timeouts = {
        "pair": 300,
        "ample_gcg": 300,
        "pgd": 60,
    }

    # Define the base command template
    commands = []

    # Iterate over rows in missing_summary
    for _, row in missing_summary.iterrows():
        # Extract model, algorithm (attack name), and prompt indices
        model_name = row["model"]
        attack_name = row["algorithm"]
        indices = row["prompt_idx"]

        # Ensure indices are a list
        if isinstance(indices, int):
            indices = [indices]

        # Calculate the timeout based on the attack name
        timeout = timeouts.get(attack_name, 240)

        # Base command template
        template = (
            "python run_attacks.py -m ++model_name={model_name} ++attack_name={attack_name} "
            "++datasets.adv_behaviors.idx={indices} ++dataset_name=adv_behaviors "
            "++hydra.launcher.timeout_min={timeout}"
        )

        # Adjust command for specific attacks
        if attack_name in ("pair", "ample_gcg"):
            template = (
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " + template +
                " ++hydra.launcher.partition=gpu_h100"
            )

        # Convert indices to a comma-separated string
        indices_str = ','.join(map(str, indices))

        # Append the formatted command
        commands.append(
            template.format(
                model_name=model_name,
                attack_name=attack_name,
                indices=indices_str,
                timeout=timeout
            )
        )

    # Run commands in parallel
    if args.run:
        print('Running commands')
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(subprocess.run, c, shell=True) for c in commands]
            for future in as_completed(futures):
                try:
                    future.result()  # Raises an exception if the command failed
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        break
                    print(e)
    else:
        for c in commands:
            print(c)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt-idx', type=int, default=300, help='Look at attacks for prompts up to this index')
    argparser.add_argument('--run', action='store_true', help='Run the commands')
    args = argparser.parse_args()
    main(args)