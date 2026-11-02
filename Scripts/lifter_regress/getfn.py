import sys, re
path, names = sys.argv[1], sys.argv[2:]
out, cur = [], None
for line in open(path):
    if line.startswith("define "):
        m = re.search(r'@"?([^"(]+)"?\(', line)
        cur = m.group(1) if m else None
    if cur in names:
        out.append(line.rstrip("\n"))
    if line.startswith("}"):
        cur = None
print("\n".join(out))
