#!/usr/bin/env python3
"""Quick quality audit of generated conversations."""
import json, re, sys, glob

files = sorted(glob.glob("data_v2/generated/conv_*.json"))
if not files:
    files = sorted(glob.glob("data/data_v2/*/generated/conv_*.json"))
print(f"Found {len(files)} files\n")

for fname in files:
    with open(fname) as f:
        data = json.load(f)

    conv = data['conversations']['en']
    asst = [m for m in conv if m['role'] == 'assistant']
    user = [m for m in conv if m['role'] == 'user']
    n = len(asst)

    print(f"=== {data['id']} ===")
    print(f"Domain: {data['domain']} | Msgs: {len(conv)} | Quality: {data['quality']['overall']}/10 | DDM: {data['quality']['ddm_compliance']}/10")

    # DDM
    l1 = sum(1 for m in asst if '[SYS_ACK: ACTIVE]' in m['content'])
    l2 = sum(1 for m in asst if len(re.findall(r'^\s*\d+[\.)]\s', m['content'], re.MULTILINE)) >= 2)
    l3 = sum(1 for m in asst if not re.search(r'\bhowever\b', m['content'], re.IGNORECASE))
    l4 = sum(1 for m in asst if re.search(r'\[Source:', m['content']))
    at = sum(1 for m in asst if 'according to' in m['content'].lower())
    print(f"DDM: L1={l1}/{n} L2={l2}/{n} L3={l3}/{n} L4={l4}/{n} | AccordingTo={at}")

    # Failures
    for i, m in enumerate(asst):
        if '[SYS_ACK: ACTIVE]' not in m['content']:
            print(f"  L1 FAIL turn {i+1}")
        if not re.search(r'\[Source:', m['content']):
            print(f"  L4 FAIL turn {i+1}")

    # First user msg
    print(f"First user: \"{user[0]['content'][:120]}\"")

    # Phone greeting?
    bad = any(p in user[0]['content'].lower() for p in ['sorry to bother', 'got your number', 'this is sarah'])
    if bad:
        print("  ⚠️ PHONE GREETING DETECTED")

    # Phantom refs
    pcount = 0
    for ui, um in enumerate(conv):
        if um['role'] != 'user':
            continue
        refs = re.findall(r'(?:you\s+(?:mentioned|said|suggested|told\s+me|recommended))\s+(.{10,80}?)(?:\.|,|\?|!|$)', um['content'].lower())
        if refs:
            prior = ' '.join(c['content'].lower() for c in conv[:ui] if c['role'] == 'assistant')
            for ref in refs:
                terms = [w for w in re.findall(r'[a-z]{5,}', ref) if w not in ('about','which','there','their','these','those','would','could','should')]
                if terms and not any(t in prior for t in terms):
                    pcount += 1
                    print(f"  PHANTOM: user refs '{ref[:50]}' missing={terms[:3]}")
    if pcount == 0:
        print("Phantoms: none ✅")

    print()
