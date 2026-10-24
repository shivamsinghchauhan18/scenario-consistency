"""Decode a pickle that contains only dicts, lists and strings, without unpickling it.

Walks the opcode stream with pickletools and rebuilds the data; any opcode outside the
whitelist (e.g. GLOBAL/REDUCE, which could run code) raises instead of executing.
"""
import pickletools

_MARK = object()
_ALLOWED = {"PROTO", "FRAME", "EMPTY_DICT", "EMPTY_LIST", "MEMOIZE", "MARK", "SHORT_BINUNICODE",
            "BINUNICODE", "BINGET", "LONG_BINGET", "APPENDS", "APPEND", "SETITEMS", "SETITEM", "STOP"}


def load(path):
    stack, memo = [], []
    with open(path, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            name = op.name
            if name not in _ALLOWED:
                raise ValueError(f"refusing opcode {name}")
            if name in ("PROTO", "FRAME"):
                continue
            if name == "EMPTY_DICT":
                stack.append({})
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name in ("SHORT_BINUNICODE", "BINUNICODE"):
                stack.append(arg)
            elif name == "MEMOIZE":
                memo.append(stack[-1])
            elif name in ("BINGET", "LONG_BINGET"):
                stack.append(memo[arg])
            elif name == "MARK":
                stack.append(_MARK)
            elif name in ("APPENDS", "SETITEMS"):
                i = len(stack) - 1 - stack[::-1].index(_MARK)
                items, stack = stack[i + 1:], stack[:i]
                target = stack[-1]
                if name == "APPENDS":
                    target.extend(items)
                else:
                    for k, v in zip(items[::2], items[1::2]):
                        target[k] = v
            elif name == "APPEND":
                v = stack.pop(); stack[-1].append(v)
            elif name == "SETITEM":
                v = stack.pop(); k = stack.pop(); stack[-1][k] = v
            elif name == "STOP":
                return stack.pop()
