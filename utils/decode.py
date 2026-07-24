import torch
import numpy as np
from itertools import groupby

try:
    from pyctcdecode import build_ctcdecoder
    _has_pyctc = True
except Exception:
    _has_pyctc = False


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode="max", blank_id=0):
        self.i2g_dict = {v[0]: k for k, v in gloss_dict.items()}
        self.num_classes = num_classes
        self.blank_id = blank_id
        self.search_mode = search_mode.lower()

        # "" at blank_id so pyctcdecode uses it as CTC blank instead of appending its own
        self.vocab = [chr(20000 + i) for i in range(num_classes)]
        self.vocab[blank_id] = ""

        if _has_pyctc and self.search_mode != "max":
            try:
                self.beam_decoder = build_ctcdecoder(self.vocab)
            except Exception as e:
                print("Beam decoder init failed, falling back to greedy:", e)
                self.beam_decoder = None
        else:
            self.beam_decoder = None

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max" or self.beam_decoder is None:
            return self._greedy(nn_output, vid_lgt)
        return self._beam(nn_output, vid_lgt, probs)

    def _greedy(self, logits, lengths):
        index = torch.argmax(logits, dim=2)
        results = []
        for b in range(index.size(0)):
            L = int(lengths[b])
            seq = index[b][:L].tolist()
            seq = [s for s, _ in groupby(seq) if s != self.blank_id]
            sent = [(self.i2g_dict.get(cid, "UNK"), i) for i, cid in enumerate(seq)]
            results.append(sent)
        return results

    def _beam(self, logits, lengths, probs=False):
        if not probs:
            logits = logits.softmax(dim=-1)
        logits = logits.detach().cpu().numpy()
        results = []
        for b in range(logits.shape[0]):
            L = int(lengths[b])
            logit = logits[b][:L]
            try:
                decoded = self.beam_decoder.decode(logit)
            except Exception as e:
                print("beam error, falling back to greedy:", e)
                return self._greedy(torch.tensor(logits), lengths)
            class_ids = [ord(ch) - 20000 for ch in decoded]
            sent = [(self.i2g_dict.get(cid, "UNK"), i)
                    for i, cid in enumerate(class_ids) if cid != self.blank_id]
            results.append(sent)
        return results
