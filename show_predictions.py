"""Show the model's predicted glosses next to the ground truth.

    python show_predictions.py                    # dev split, default work_dir
    python show_predictions.py --mode test
    python show_predictions.py --work-dir ./work_dir/local_test/ --all

Reads the hypothesis CTM written during evaluation (one row per predicted
gloss) and the ground-truth STM, and prints them aligned per clip.
"""
import argparse
import os
from collections import defaultdict


def read_ctm(path):
    """CTM row: <id> 1 <start> <end> <gloss>  ->  {id: [gloss, ...]}"""
    preds = defaultdict(list)
    if not os.path.exists(path):
        return preds
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                preds[parts[0]].append(parts[4])
    return preds


def read_stm(path):
    """STM row: <id> 1 <signer> <start> <end> <gloss...>  ->  {id: [gloss, ...]}"""
    truth = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                truth[parts[0]] = parts[5:]
    return truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', default='./work_dir/local_test/')
    p.add_argument('--mode', default='dev', choices=['dev', 'test', 'train'])
    p.add_argument('--eval-dir', default='./evaluation/slr_eval')
    p.add_argument('--all', action='store_true',
                   help='show every clip (default: one example per sentence)')
    args = p.parse_args()

    hyp = read_ctm(os.path.join(args.work_dir, f'output-hypothesis-{args.mode}.ctm'))
    ref = read_stm(os.path.join(args.eval_dir, f'CESLR-groundtruth-{args.mode}.stm'))
    if not hyp:
        raise SystemExit(
            f'No predictions in {args.work_dir}. Run training (which evaluates '
            f'each epoch) or `main.py --phase test` first.')

    seen, exact = set(), 0
    shown = 0
    for clip_id, truth in ref.items():
        key = ' '.join(truth)
        if not args.all and key in seen:
            continue
        seen.add(key)
        pred = hyp.get(clip_id, [])
        ok = pred == truth
        exact += ok
        shown += 1
        print(f"\n{clip_id}  {'MATCH' if ok else 'DIFF'}")
        print(f"  truth: {' '.join(truth)}")
        print(f"  pred : {' '.join(pred) if pred else '(empty)'}")

    # Exact-sentence accuracy over everything, not just what was printed.
    total_exact = sum(1 for cid, t in ref.items() if hyp.get(cid, []) == t)
    print(f"\n{'-' * 60}")
    print(f"showed {shown} clip(s); exact sentence matches over all "
          f"{len(ref)} {args.mode} clips: {total_exact} "
          f"({100.0 * total_exact / max(len(ref), 1):.1f}%)")
    print("WER per epoch is in", os.path.join(args.work_dir, f'{args.mode}.txt'))


if __name__ == '__main__':
    main()
