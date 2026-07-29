from dataclasses import dataclass

@dataclass(slots=True)
class Experiment:
    prompt: str
    target: str