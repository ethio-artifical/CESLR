"""Normalize CESLR annotation inconsistencies and regenerate derived files.

Fixes applied to every label across train/dev/test:
  - 'እወዳለሁማን' -> 'እወዳለሁ ማን'   (merged token, single occurrence)
  - 'ጽሎት'      -> 'ጸሎት'          (two spellings of the same sign)

Then rebuilds gloss_dict.npy from the corrected labels (same scheme as
dataset_preprocess.py: sorted glosses, index from 1, occurrence counts) and
rewrites the ground-truth STM files in BOTH preprocess/CESLR/ and
evaluation/slr_eval/ (the evaluator reads the latter).

Idempotent: safe to run multiple times, including after re-running
dataset_preprocess.py.
"""
import os
import numpy as np

REPLACEMENTS = [
    ('እወዳለሁማን', 'እወዳለሁ ማን'),
    ('ጽሎት', 'ጸሎት'),
]

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, 'CESLR')
EVAL = os.path.join(HERE, '..', 'evaluation', 'slr_eval')


def fix_label(label):
    for old, new in REPLACEMENTS:
        label = label.replace(old, new)
    return ' '.join(label.split())


def main():
    gloss_counts = {}
    for split in ['train', 'dev', 'test']:
        path = os.path.join(PREP, f'{split}_info.npy')
        info = np.load(path, allow_pickle=True).item()
        changed = 0
        for k, v in info.items():
            if not isinstance(v, dict):
                continue
            fixed = fix_label(v['label'])
            if fixed != v['label']:
                changed += 1
            v['label'] = fixed
            parts = v['original_info'].split('|')
            parts[-1] = fixed
            v['original_info'] = '|'.join(parts)
            for g in fixed.split():
                gloss_counts[g] = gloss_counts.get(g, 0) + 1
        np.save(path, info)

        stm_lines = [
            f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n"
            for k, v in info.items() if isinstance(v, dict)
        ]
        for out_dir in [PREP, EVAL]:
            with open(os.path.join(out_dir, f'CESLR-groundtruth-{split}.stm'),
                      'w', encoding='utf-8') as f:
                f.writelines(stm_lines)
        print(f'{split}: {len(stm_lines)} entries, {changed} labels corrected')

    gloss_dict = {g: [i + 1, c] for i, (g, c)
                  in enumerate(sorted(gloss_counts.items()))}
    np.save(os.path.join(PREP, 'gloss_dict.npy'), gloss_dict)
    print(f'gloss_dict rebuilt: {len(gloss_dict)} glosses '
          f'(num_classes = {len(gloss_dict) + 1} incl. blank)')


if __name__ == '__main__':
    main()
