from dataclasses import dataclass
from typing import Dict, List, Optional

from utils import load_json, save_json

UNK_TOKEN = "<unk>"


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        if UNK_TOKEN in vocab:
            vocab.remove(UNK_TOKEN)
        stoi = {UNK_TOKEN: 0}
        stoi.update({ch: i + 1 for i, ch in enumerate(vocab)})
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str, skip_unknown: bool = False) -> List[int]:
        out: List[int] = []
        for ch in text:
            if ch in self.stoi:
                out.append(self.stoi[ch])
            elif skip_unknown:
                continue
            else:
                out.append(self.unk_id)
        return out

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str) -> None:
        save_json(path, {"stoi": self.stoi, "unk_token": UNK_TOKEN})

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        obj = load_json(path)
        stoi = {k: int(v) for k, v in obj["stoi"].items()}
        if UNK_TOKEN not in stoi:
            # Backward compatibility for older vocab files.
            stoi = {UNK_TOKEN: 0, **{k: v + 1 for k, v in stoi.items()}}
        itos = {v: k for k, v in stoi.items()}
        return cls(stoi=stoi, itos=itos)
