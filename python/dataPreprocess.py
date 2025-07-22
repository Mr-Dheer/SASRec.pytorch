#!/usr/bin/env python3
import os
import gzip
import json
from collections import defaultdict

def parse(path):
    """
    Yield one JSON record at a time from a gzip‐compressed JSONLines file.
    """
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            yield json.loads(line)

def main():
    # — CONFIGURATION — 
    dataset_name = 'Sports_and_Outdoors'
    raw_dir      = '/home/kavach/Dev/NewResearch/E2P/recommender-e2p/data/raw'
    out_dir      = '/home/kavach/Dev/NewResearch/E2P/recommender-e2p/data/processed'
    os.makedirs(out_dir, exist_ok=True)

    # — AUTO-DETECT INPUT FILE — 
    candidates = [
        f'{dataset_name}_5.json.gz',
        f'reviews_{dataset_name}_5.json.gz',
        f'{dataset_name}.json.gz',
        f'reviews_{dataset_name}.json.gz',
    ]
    for fn in candidates:
        path = os.path.join(raw_dir, fn)
        if os.path.exists(path):
            input_file = path
            break
    else:
        raise FileNotFoundError(
            f"None of {candidates!r} found in {raw_dir}"
        )
    print(f"Reading from: {input_file}")

    # — FIRST PASS: count total reviews per user/item, dump raw lines — 
    countU = defaultdict(int)
    countP = defaultdict(int)
    interm_file = os.path.join(out_dir, f'reviews_{dataset_name}.txt')
    with open(interm_file, 'w') as fout:
        for rec in parse(input_file):
            reviewer = rec['reviewerID']
            asin     = rec['asin']
            rating   = rec['overall']
            tstamp   = rec['unixReviewTime']
            fout.write(f"{reviewer} {asin} {rating} {tstamp}\n")
            countU[reviewer] += 1
            countP[asin]     += 1

    # — SECOND PASS: filter out cold users/items, remap IDs, collect sequences — 
    usermap = {}
    itemmap = {}
    User    = defaultdict(list)
    u_ctr   = 0
    i_ctr   = 0

    for rec in parse(input_file):
        reviewer = rec['reviewerID']
        asin     = rec['asin']
        tstamp   = rec['unixReviewTime']

        # Skip anyone with <5 reviews or any item with <5 reviews
        if countU[reviewer] < 5 or countP[asin] < 5:
            continue

        # assign a compact user ID
        if reviewer not in usermap:
            u_ctr += 1
            usermap[reviewer] = u_ctr
        uid = usermap[reviewer]

        # assign a compact item ID
        if asin not in itemmap:
            i_ctr += 1
            itemmap[asin] = i_ctr
        iid = itemmap[asin]

        User[uid].append((tstamp, iid))

    # — SORT each user’s interactions by timestamp — 
    for uid in User:
        User[uid].sort(key=lambda x: x[0])

    print(f"Kept {u_ctr} users and {i_ctr} items after filtering.")

    # — WRITE FINAL SEQUENCES — 
    final_file = os.path.join(out_dir, f'{dataset_name}.txt')
    with open(final_file, 'w') as fout:
        for uid, seq in User.items():
            for _, iid in seq:
                fout.write(f"{uid} {iid}\n")

    print(f"Intermediate file: {interm_file}")
    print(f"Final sequences file: {final_file}")

if __name__ == '__main__':
    main()
