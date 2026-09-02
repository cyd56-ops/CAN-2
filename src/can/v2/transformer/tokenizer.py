"""Phase 5 使用的确定性 byte-level tokenizer。"""

from typing import Iterable, List, Sequence


class ByteTokenizer:
    """将 UTF-8 文本映射到固定的 260 项词表。"""

    byte_vocab_size = 256
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    unk_token_id = 259
    vocab_size = 260

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = 256,
    ) -> List[int]:
        """编码文本并按配置添加边界 token。

        参数:
            text: 待编码的 Unicode 文本。
            add_bos: 是否添加 ``<bos>``。
            add_eos: 是否添加 ``<eos>``。
            max_length: 编码结果允许的最大长度。

        返回:
            固定词表中的 token ID 列表。
        """

        if not isinstance(text, str):
            raise TypeError("text 必须是 str")
        if not isinstance(add_bos, bool) or not isinstance(add_eos, bool):
            raise TypeError("add_bos 和 add_eos 必须是 bool")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length 必须是整数")
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_token_id)
        if len(token_ids) > max_length:
            raise ValueError("编码结果超过 max_length")
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        """把 token ID 解码为 UTF-8 文本。

        参数:
            token_ids: 待解码的 token ID 序列。
            skip_special: 是否忽略边界与填充 token。

        返回:
            解码后的文本；非法 UTF-8 字节使用替换字符表示。
        """

        if not isinstance(skip_special, bool):
            raise TypeError("skip_special 必须是 bool")
        values = self._validate_token_ids(token_ids)
        byte_values = bytearray()
        for token_id in values:
            if token_id < self.byte_vocab_size:
                byte_values.append(token_id)
            elif not skip_special:
                raise ValueError("特殊 token 不能直接解码为文本")
        return bytes(byte_values).decode("utf-8", errors="replace")

    @classmethod
    def _validate_token_ids(cls, token_ids: Iterable[int]) -> Sequence[int]:
        """验证 token ID 序列并返回稳定列表。"""

        if isinstance(token_ids, (str, bytes)):
            raise TypeError("token_ids 必须是整数序列")
        try:
            values = list(token_ids)
        except TypeError as exc:
            raise TypeError("token_ids 必须可迭代") from exc
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token_id 必须是非 bool 整数")
            if not 0 <= token_id < cls.vocab_size:
                raise ValueError("token_id 超出固定词表范围")
        return values


__all__ = ["ByteTokenizer"]
