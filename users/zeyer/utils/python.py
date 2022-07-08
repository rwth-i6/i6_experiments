"""
generic Python utils
"""


def is_valid_python_attrib_name(name: str) -> bool:
    """
    :return: whether the name is a valid Python attrib name
    """
    # Very hacky. I'm sure there is some clever regexp but I don't find it and too lazy...
    class _Obj:
        pass
    obj = _Obj()
    try:
        exec(f"obj.{name} = 'ok'", {"obj": obj})
    except SyntaxError:
        return False
    assert getattr(obj, name) == "ok"
    return True
