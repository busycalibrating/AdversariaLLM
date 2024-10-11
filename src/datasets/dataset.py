from dataclasses import dataclass

import jailbreakbench as jbb
import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_name(cls, name):
        match name:
            case "adv_behaviors":
                return AdvBehaviorsDataset
            case "jbb_behaviors":
                return JBBBehaviorsDataset
            case _:
                raise ValueError(f"Unknown dataset: {name}")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[str, str]:
        raise NotImplementedError


@dataclass
class AdvBehaviorsConfig:
    name: str
    messages_path: str
    targets_path: str
    categories: list[str]
    seed: int = 0
    idx: list[int]|int|None = None


class AdvBehaviorsDataset(Dataset):
    def __init__(self, config: AdvBehaviorsConfig):
        self.config = config
        import logging
        import os
        import sys
        logging.info(f"Loading dataset from {config.messages_path}")
        logging.info(f"Current working directory: {os.getcwd()}")

        self.messages = pd.read_csv(config.messages_path, header=0)
        # Ignore copyright-related rows
        self.messages = self.messages[
            self.messages["SemanticCategory"].isin(config.categories)
        ]
        targets = pd.read_json(config.targets_path, typ="series")
        # Merge the CSV data with the JSON data
        self.targets = self.messages[self.messages.columns[-1]].map(targets)
        assert len(self.messages) == len(self.targets), "Mismatched lengths"

        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(self.messages))
        # We keep this shuffle for backwards compatibility
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, list):
            idx = idx[config.idx]
        # cut
        self.messages = self.messages.iloc[idx].reset_index(drop=True)
        self.targets = self.targets.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg["Behavior"]}

        if isinstance(msg["ContextString"], str):
            message["content"] = msg["ContextString"] + "\n\n" + message["content"]
        target: str = self.targets.iloc[idx]
        return message, target


@dataclass
class JBBBehaviorsConfig:
    name: str
    seed: int = 0
    idx: list[int]|int|None = None


class JBBBehaviorsDataset(Dataset):
    def __init__(self, config: JBBBehaviorsConfig):
        self.config = config
        dataset = jbb.read_dataset().as_dataframe()

        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(dataset))
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, list):
            idx = idx[config.idx]
        self.messages = dataset.Goal.iloc[idx].reset_index(drop=True)
        self.targets = dataset.Target.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg}
        target: str = self.targets.iloc[idx]
        return message, target


if __name__ == "__main__":
    d = Dataset.from_name("adv_behaviors")(
        AdvBehaviorsConfig(
            "adv_behaviors",
            messages_path="data/behavior_datasets/harmbench_behaviors_text_all.csv",
            targets_path="data/optimizer_targets/harmbench_targets_text.json",
            seed=1,
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
    print(len(d))
    d = Dataset.from_name("jbb_behaviors")(
        JBBBehaviorsConfig("jbb_behaviors", seed=1)
    )
    print(len(d))
