from tokenizers import ByteLevelBPETokenizer


class TiktokenWrapper:

    def __init__(self, save_dir: str = "data/tokenizer"):
        self._enc = ByteLevelBPETokenizer(
            f"{save_dir}/vocab.json",
            f"{save_dir}/merges.txt",
        )
        self.vocab_size = self._enc.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text).ids

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
