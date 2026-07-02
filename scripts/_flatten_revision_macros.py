"""Flatten LaTeX revision-tracking macros in-place (source-level).

Usage:
    python3 _flatten_revision_macros.py <file1.tex> [<file2.tex> ...]

Transformations:
    \\chgR{X}{Y}       -> Y                (keep new only)
    \\chg{X}{Y}        -> Y
    \\rev{X}           -> X
    \\revR{X}          -> X
    \\addchg{X}        -> X
    \\addchgR{X}       -> X
    \\tracknote{X}     -> (deleted)
    \\begin{revblock}...\\end{revblock}    -> ... (env wrapper stripped)
    \\begin{revblockR}...\\end{revblockR}  -> ...

Brace-balanced parsing handles nested macros/braces correctly.
"""
import re
import sys
from pathlib import Path


def _parse_braced_arg(text, p):
    """Parse one {...} starting at text[p]=='{'. Returns (inner_str, end_pos).
    end_pos is the position immediately after the closing '}'."""
    assert text[p] == '{', f"expected '{{' at pos {p}, got {text[p]!r}"
    depth = 1
    start = p + 1
    p += 1
    while p < len(text) and depth > 0:
        c = text[p]
        if c == '\\':
            p += 2  # skip escaped char (e.g. \}, \{, \\)
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        p += 1
    return text[start:p - 1], p


def flatten_2arg(text, macro_name, keep_idx):
    """\\macro{X}{Y} -> X if keep_idx==0 else Y."""
    out = []
    i = 0
    needle = f"\\{macro_name}{{"
    while True:
        idx = text.find(needle, i)
        if idx < 0:
            out.append(text[i:])
            return ''.join(out)
        out.append(text[i:idx])
        p = idx + len(needle) - 1  # position of '{'
        arg1, p = _parse_braced_arg(text, p)
        if p < len(text) and text[p] == '{':
            arg2, p = _parse_braced_arg(text, p)
            out.append(arg1 if keep_idx == 0 else arg2)
            i = p
        else:
            # Not actually a 2-arg call: emit as-is and skip past first brace
            out.append(text[idx:p])
            i = p


def flatten_1arg(text, macro_name, keep=True):
    """\\macro{X} -> X if keep else ''."""
    out = []
    i = 0
    needle = f"\\{macro_name}{{"
    while True:
        idx = text.find(needle, i)
        if idx < 0:
            out.append(text[i:])
            return ''.join(out)
        out.append(text[i:idx])
        p = idx + len(needle) - 1
        arg, p = _parse_braced_arg(text, p)
        if keep:
            out.append(arg)
        i = p


def flatten_env(text, env_name):
    """Remove \\begin{env} and \\end{env}, keep inner."""
    text = re.sub(r'\\begin\{' + re.escape(env_name) + r'\}\s*\n?', '', text)
    text = re.sub(r'\\end\{' + re.escape(env_name) + r'\}\s*\n?', '', text)
    return text


def flatten_all(text):
    # 2-arg macros — process longer name first to avoid prefix collision
    text = flatten_2arg(text, 'chgR', keep_idx=1)
    text = flatten_2arg(text, 'chg',  keep_idx=1)
    # 1-arg macros — process longer name first
    text = flatten_1arg(text, 'revR',     keep=True)
    text = flatten_1arg(text, 'rev',      keep=True)
    text = flatten_1arg(text, 'addchgR',  keep=True)
    text = flatten_1arg(text, 'addchg',   keep=True)
    text = flatten_1arg(text, 'tracknote', keep=False)
    # Environments — process R variant first (same reason)
    text = flatten_env(text, 'revblockR')
    text = flatten_env(text, 'revblock')
    return text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        original = p.read_text()
        flattened = flatten_all(original)
        p.write_text(flattened)
        # Verify
        remaining = {}
        for macro in ['chgR', 'chg', 'rev', 'revR', 'addchg', 'addchgR', 'tracknote']:
            n = flattened.count(f'\\{macro}{{')
            if n > 0:
                remaining[macro] = n
        for env in ['revblock', 'revblockR']:
            n = flattened.count(f'\\begin{{{env}}}')
            if n > 0:
                remaining[env + ' env'] = n
        print(f"{p.name}: {len(original)} -> {len(flattened)} chars  ({len(flattened)-len(original):+})")
        if remaining:
            print(f"  WARN remaining: {remaining}")
        else:
            print(f"  OK: all revision macros flattened")
