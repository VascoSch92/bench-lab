import csv
import io
import urllib.request
from typing import Final

from benchlab._dataset import Dataset
from benchlab.library._jailbreak_llms._instance import JailbreakLLMsInstance

__all__ = ["JailbreakLLMsDataset"]


URL_DATASET: Final[str] = (
    "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv"
)


class JailbreakLLMsDataset(Dataset[JailbreakLLMsInstance]):
    def __post_init__(self, split: str):
        super().__post_init__(split=split)
        self._instances: list[JailbreakLLMsInstance] = self._load_dataset()
        self._map_idx: dict[str, int] = {
            instance.id: idx for idx, instance in enumerate(self._instances)
        }

    def get(self, idx: int | str) -> JailbreakLLMsInstance:
        if isinstance(idx, str):
            if idx in self._map_idx:
                return self._instances[self._map_idx[idx]]
            raise ValueError(f"Unknown index {idx}")
        return self._instances[idx]

    def __len__(self) -> int:
        return len(self._instances)

    @staticmethod
    def _load_dataset() -> list[JailbreakLLMsInstance]:
        with urllib.request.urlopen(URL_DATASET) as response:
            text = response.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))

        return [
            JailbreakLLMsInstance(
                id=f"{row['content_policy_id']}_{row['q_id']}",
                content_policy_id=row["content_policy_id"],
                content_policy_name=row["content_policy_name"],
                question=row["question"],
            )
            for row in reader
        ]
