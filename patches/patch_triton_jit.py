"""Patch triton's jit.py to handle regex failure gracefully.

The regex that strips decorators from kernel source code returns None for
some fla (flash-linear-attention) kernels, causing an AttributeError.
This patch makes it fall through instead of crashing.
"""
import triton.runtime.jit as j

path = j.__file__
with open(path) as f:
    src = f.read()

# The buggy line:
#   src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
# Replace .start():] with a safe version that checks for None
old = '.start():]'
# We need to be surgical — only replace the specific occurrence in __init__
target = 're.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE).start():]'
replacement = 're.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE); src = src[_m.start():] if _m else src'

# Do a line-level replacement for safety
lines = src.split('\n')
patched = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if 'src = src[re.search(r"^def' in line and '.start():]' in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = indent + '_m = re.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE); src = src[_m.start():] if _m else src'
        patched = True
        break

if patched:
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Patched triton jit.py at {path}')
else:
    print(f'Target pattern not found in {path} — may already be fixed')
