import os
import yaml
import json
import wandb


def create_yaml_for_run_attacks(model_path, config_path, experiment_id):
    model_config = json.load(open(os.path.join(model_path, "adapter_config.json")))
    model_name = model_config["base_model_name_or_path"]

    model_yamls_path = os.path.join(config_path, "models.yaml")
    with open(model_yamls_path, "r") as f:
        model_yamls = yaml.load(f, Loader=yaml.FullLoader)
    for m in model_yamls.keys():
        if m == model_name:
            model_yaml = {experiment_id: model_yamls[m]}
            model_yaml[experiment_id]["id"] = model_path
            model_yaml[experiment_id]["wandb_experiment_id"] = experiment_id
            new_model_file = os.path.join(config_path, experiment_id + ".yaml")
            with open(new_model_file, "w") as f:
                yaml.dump(model_yaml, f)
            break


def log_to_wandb(log_path):
    dates = os.listdir(log_path)

    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]

    for d in dates:
        date_times = os.listdir(os.path.join(log_path, d))
        for dt in date_times:
            run_json = os.path.join(log_path, d, dt, "run.json")
            if os.path.exists(run_json):
                with open(run_json, "r") as f:
                    run = yaml.load(f, Loader=yaml.FullLoader)
                    attack_name = run[0]["config"]["attack"]
                    model_params = run[0]["config"]["model_params"]
                    yes_no = run[0]["successes_cais"]
                    yes_no = [y[0].lower() for y in yes_no]
                    asr = yes_no.count("yes") / len(yes_no)

                    experiment_id = model_params["wandb_experiment_id"].strip()
                    wandb.init(project=project, entity=entity, id=experiment_id, resume="allow")
                    wandb.log({f"asr_{attack_name}": asr})
                    wandb.finish()
                    print("Logged attack for " + experiment_id)
            else:
                print("No attack json found for " + experiment_id)
    print("Done logging to wandb")
