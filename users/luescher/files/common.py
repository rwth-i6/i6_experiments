__all__ = ["CodeBlock", "DictBlock", "FunctionBlock", "DEFAULT_INDENT"]

from typing import Any, Dict, List, Optional, Union


DEFAULT_INDENT = 4


def indent_times(repeat: int):
    return repeat * " "


class Block:
    def __init__(self):
        self.current_indent: int = 0

    def __str__(self, indent: int = 0):
        raise NotImplementedError


class CodeBlock(Block):
    def __init__(self, head: str, block: List[Union[str, Block]]):
        super().__init__()
        self.head = head
        self.block = block

    def __str__(self, indent: int = 0):
        self.current_indent = indent
        result = (
            f"{indent_times(self.current_indent)}{self.head}:\n"
            if self.head != ""
            else ""
        )
        self.current_indent += DEFAULT_INDENT

        for block in self.block:
            if isinstance(block, Block):
                result += block.__str__(self.current_indent)
            else:
                result += f"{indent_times(self.current_indent)}{block}\n"

        return result


class DictBlock(Block):
    def __init__(self, keys_values: Dict[Any, Any], add_final_comma: bool = True):
        super().__init__()
        self.key_value_pairs = keys_values
        self.add_final_comma = add_final_comma

    def __str__(self, indent: int = 0):
        self.current_indent = indent
        result = "{\n"
        self.current_indent += DEFAULT_INDENT

        for k, v in self.key_value_pairs.items():
            if isinstance(v, Block):
                result += f'{indent_times(self.current_indent)}"{k}": {v.__str__(self.current_indent)},\n'
            elif isinstance(v, Dict):
                res = DictBlock(v, add_final_comma=True)
                result += f'{indent_times(self.current_indent)}"{k}": {res.__str__(self.current_indent)}\n'
            elif isinstance(v, str):
                result += f'{indent_times(self.current_indent)}"{k}": "{v}",\n'
            else:
                result += f'{indent_times(self.current_indent)}"{k}": {v},\n'

        self.current_indent -= DEFAULT_INDENT

        result += (
            f"{indent_times(self.current_indent)}" + "},"
            if self.add_final_comma
            else f"{indent_times(self.current_indent)}" + "}\n"
        )

        return result


class FunctionBlock(Block):
    def __init__(
        self,
        head: str,
        body: List[Union[str, Block]],
        add_colon: bool = False,
        add_call: Optional[str] = None,
    ):
        super().__init__()
        self.head = head
        self.body = body
        self.add_colon = add_colon
        self.add_call = add_call

    def __str__(self, indent: int = 0):
        self.current_indent = indent
        result = f"{indent_times(self.current_indent)}{self.head}(\n"
        self.current_indent += DEFAULT_INDENT

        for body in self.body:
            if isinstance(body, Block):
                result += body.__str__(self.current_indent)
            else:
                result += f"{indent_times(self.current_indent)}{body}\n"

        self.current_indent -= DEFAULT_INDENT

        result += "):" if self.add_colon else ")"

        if self.add_call is not None:
            result += self.add_call

        result += "\n\n"

        return result
