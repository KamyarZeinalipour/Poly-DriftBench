import json, re
from pathlib import Path

ECHO_PATTERNS = [
    r"(?:i'?ll try|i'?ll do|i'?ll use|let me try)\s+(?:the\s+)?(.{5,40}?)(?:[.,!?\n]|$)",
    r"(?:you mentioned|you said|you suggested|as you noted|as you pointed out)\s+(.{5,40}?)(?:[.,!?\n]|$)",
    r"(?:the|that|your)\s+([\w\s]{3,25}?)(?:thing|idea|point|tip|suggestion|advice|method|approach|technique|strategy|trick)",
]
SAFE = {"that makes sense", "that sounds good", "okay", "alright", "got it", "fair enough"}

import sys
target = sys.argv[1] if len(sys.argv) > 1 else "data/catd_test"

for fp in sorted(Path(target).rglob("*.json")):
    if "summary" in fp.name:
        continue
    data = json.load(open(fp))
    msgs = data.get("conversations", {}).get("en", [])
    users = [m for m in msgs if m["role"] == "user"]
    assts = [m for m in msgs if m["role"] == "assistant"]
    domain = data.get("domain", fp.stem)
    tier = fp.parent.parent.name

    violations = 0
    total = max(len(users) - 1, 0)

    print("=" * 60)
    print("  %s (%s) - %d user turns" % (domain, tier, len(users)))
    print("=" * 60)

    for i, u in enumerate(users):
        if i == 0:
            continue
        msg = u["content"]
        ul = msg.lower()
        if (i - 1) >= len(assts):
            continue
        prev = assts[i - 1]["content"].lower()
        issues = []

        for pattern in ECHO_PATTERNS:
            for match in re.findall(pattern, ul):
                clean = match.strip()
                if len(clean) < 4:
                    continue
                if any(s in clean for s in SAFE):
                    continue
                if clean in prev:
                    issues.append('ECHO: "%s"' % clean[:30])
                else:
                    words = clean.split()[:4]
                    phrase = " ".join(words)
                    if len(words) >= 3 and phrase in prev:
                        issues.append('PARTIAL: "%s"' % phrase[:30])

        if issues:
            violations += 1
            snippet = msg[:100].replace("\n", " ")
            print("  !! User %d: %s..." % (i + 1, snippet))
            for iss in issues:
                print("      -> %s" % iss)
        else:
            snippet = msg[:80].replace("\n", " ")
            print("  OK User %d: %s..." % (i + 1, snippet))

    pct = violations / total * 100 if total else 0
    print()
    print("  RESULT: %d/%d violations (%.1f%%)" % (violations, total, pct))
    if pct <= 5:
        print("  >>> PASS!")
    elif pct <= 20:
        print("  >>> WARN - moderate")
    else:
        print("  >>> FAIL")
    print()
