"""Break a split's WER down by signer and by sentence.

    python evaluation/fairness_report.py --work-dir ./work_dir/baseline_res18/ --mode dev
    python evaluation/fairness_report.py --work-dir ./work_dir/attn_pool/ --mode test --by sentence

A single WER hides the thing this corpus is most likely to get wrong. 27% mean
could be 12% on one signer and 42% on another, or an even 25-29 across all of
them, and those are different problems: the first says the model has learned a
few signers and not the rest, the second says it is uniformly mediocre. Only the
first is fixed by more signer diversity.

The same applies per sentence. 78% of the glosses in this corpus appear in
exactly one sentence, so a model can score well by classifying which of the 30
sentences it is looking at rather than by recognising glosses. A per-sentence
breakdown makes that visible: sentence classification gives a bimodal
distribution -- near-zero on sentences it identified, near-total on the ones it
did not -- while genuine gloss recognition degrades smoothly.

Reuses python_wer_evaluation's alignment, so the totals here agree with the WER
printed during training rather than being a second, subtly different metric.
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.slr_eval.python_wer_evaluation import sent_evaluation  # noqa: E402

PENALTY = {'ins': 3, 'del': 3, 'sub': 4}


def load_stm(path):
    """STM row: <id> 1 <signer> <start> <end> <gloss...>"""
    ref = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                ref[p[0]] = {'signer': p[2], 'glosses': p[5:]}
    return ref


def load_ctm(path):
    """CTM row: <id> 1 <start> <end> <gloss>"""
    hyp = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                hyp[p[0]].append(p[4])
    return hyp


def wer_of(group):
    """Aggregate errors/count the way wer_calculation does, then divide."""
    err = sum(g['wer_lstm'] for g in group)
    cnt = sum(g['cnt'] for g in group)
    return 100.0 * err / cnt if cnt else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', default='./work_dir/baseline_res18/')
    p.add_argument('--mode', default='dev', choices=['dev', 'test', 'train'])
    p.add_argument('--eval-dir', default='./evaluation/slr_eval')
    p.add_argument('--prefix', default='CESLR-groundtruth')
    p.add_argument('--by', default='both', choices=['signer', 'sentence', 'both'])
    p.add_argument('--ctm', default=None,
                   help='override the hypothesis file (default output-hypothesis-<mode>.ctm)')
    args = p.parse_args()

    stm = os.path.join(args.eval_dir, f'{args.prefix}-{args.mode}.stm')
    ctm = args.ctm or os.path.join(args.work_dir,
                                   f'output-hypothesis-{args.mode}.ctm')
    for f in (stm, ctm):
        if not os.path.exists(f):
            raise SystemExit(f'missing {f}')

    ref, hyp = load_stm(stm), load_ctm(ctm)
    by_signer, by_sentence, per_clip = defaultdict(list), defaultdict(list), []
    missing = 0

    for cid, r in ref.items():
        pred = hyp.get(cid, [])
        if not pred:
            missing += 1
            # Match mergectmstm.py, which the normal evaluation path runs before
            # scoring: a clip the model said nothing for gets an [EMPTY] token
            # rather than an empty list. Two reasons -- get_wer_delsubins indexes
            # an empty backtrace and raises on a truly empty hypothesis, and
            # dropping the clip instead would quietly flatter the average by
            # removing the model's worst cases.
            pred = ['[EMPTY]']
        stat = sent_evaluation(info=cid, gt=r['glosses'], merge_same=True,
                               lstm_prediction=pred, penalty=PENALTY)
        by_signer[r['signer']].append(stat)
        by_sentence[' '.join(r['glosses'])].append(stat)
        per_clip.append((cid, r['signer'], wer_of([stat])))

    overall = wer_of([s for g in by_signer.values() for s in g])
    print(f'{args.mode}: {len(ref)} clips, {len(by_signer)} signers, '
          f'{len(by_sentence)} sentences')
    if missing:
        print(f'  {missing} clips had no prediction (scored as all deletions)')
    print(f'  overall WER {overall:.2f}%\n')

    if args.by in ('signer', 'both'):
        rows = sorted(((s, wer_of(g), len(g)) for s, g in by_signer.items()),
                      key=lambda r: r[1])
        _table('signer', rows)

    if args.by in ('sentence', 'both'):
        rows = sorted(((s, wer_of(g), len(g)) for s, g in by_sentence.items()),
                      key=lambda r: r[1])
        _table('sentence', rows, width=42)

    worst = sorted(per_clip, key=lambda r: -r[2])[:5]
    print('worst clips')
    for cid, sg, w in worst:
        print(f'  {w:6.1f}%  {sg:<10} {cid}')


def _table(label, rows, width=12):
    vals = [w for _, w, _ in rows if w == w]
    print(f'{"WER by " + label:<{width}} {"WER":>8} {"clips":>7}')
    print('-' * (width + 17))
    for name, w, n in rows:
        print(f'{name[:width]:<{width}} {w:7.2f}% {n:7}')
    print('-' * (width + 17))
    if len(vals) > 1:
        lo, hi = min(vals), max(vals)
        print(f'{"mean ± std":<{width}} {statistics.mean(vals):7.2f}% '
              f'± {statistics.stdev(vals):.2f}')
        print(f'{"range":<{width}} {lo:7.2f}% .. {hi:.2f}%   '
              f'spread {hi - lo:.1f} points')
    print()


if __name__ == '__main__':
    main()
