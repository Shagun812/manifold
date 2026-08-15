from dataclasses import dataclass

@dataclass(slots=True)
class Experiment:
    prompt: str | None=None
    target: str | None=None
    clean_prompt: str | None=None
    corrupted_prompt: str | None=None
    