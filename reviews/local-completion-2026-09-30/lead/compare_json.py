"""Compare a regenerated JSON result with the committed copy.

Keys that record wall-clock timing or the interpreter/platform are ignored;
every other key and value must match exactly. Prints IDENTICAL or the first
differing path.

Usage: python3 compare_json.py COMMITTED.json REGENERATED.json
"""
import json
import sys

IGNORED_KEY_FRAGMENTS = ("second", "elapsed", "duration", "_ms", "runtime",
                         "python", "platform", "timestamp")


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items()
                if not any(f in k.lower() for f in IGNORED_KEY_FRAGMENTS)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key}: present on one side only"
            found = first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
    elif a != b:
        return f"{path}: {str(a)[:80]} vs {str(b)[:80]}"
    return None


def main():
    committed, regenerated = sys.argv[1:3]
    with open(committed) as f:
        a = strip(json.load(f))
    with open(regenerated) as f:
        b = strip(json.load(f))
    difference = first_difference(a, b)
    print(f"{committed}: " + ("IDENTICAL" if difference is None else f"DIFFERS at {difference}"))
    return 0 if difference is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
