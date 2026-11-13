"""Local, conservative target-ID recognition for the supplied 1080p radar UI."""
from pathlib import Path
import cv2
import numpy as np


def glyph(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.resize(crop.astype(np.uint8), (16, 24), interpolation=cv2.INTER_NEAREST) > 0


def text_mask(rgb):
    return (rgb.min(axis=2) >= 105) & ((rgb.max(axis=2).astype(int)-rgb.min(axis=2)) < 35)


def split_id_marker(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    digits = [(index, stat) for index, stat in enumerate(stats[1:], 1)
              if stat[3] >= 8 and stat[4] >= 8]
    if not 1 <= len(digits) <= 3:
        return [], False
    digits.sort(key=lambda item: item[1][0])
    index, (x, y, width, height, area) = digits[-1]
    starred = any(sx >= x + width and sy < y + height/2
                  and 3 <= sw <= 8 and 3 <= sh <= 7 and sa >= 6
                  for sx, sy, sw, sh, sa in stats[1:])
    return [glyph(labels == index) for index, _ in digits], starred


class TargetReader:
    def __init__(self, path=None):
        path = path or Path(__file__).parent / 'assets' / 'target_digits.npz'
        with np.load(path) as data:
            self.templates = {key: data[key] for key in data.files}

    def read(self, frame):
        # Refuse uncalibrated layouts rather than clicking guessed screen positions.
        if frame.shape[:2] != (1080, 1920):
            raise ValueError('Radar targeting currently requires 1920x1080 screenshots')
        targets = []
        for row in range(20):
            y = 152 + row * 32
            if y > 790:
                break
            mask = text_mask(frame[y-9:y+10, 1351:1385])
            candidates, starred = split_id_marker(mask)
            if not candidates:
                continue
            recognized = []
            for candidate in candidates:
                by_digit = {}
                for label, template in self.templates.items():
                    digit = label.split('_')[0]
                    by_digit[digit] = min(by_digit.get(digit, 1), np.mean(candidate != template))
                scores = sorted((score, digit) for digit, score in by_digit.items())
                if scores[0][0] > .18 or scores[1][0] - scores[0][0] < .06:
                    break
                recognized.append(scores[0][1])
            if len(recognized) != len(candidates):
                continue
            background = np.median(frame[y-9:y+10, 1390:1880], axis=(0, 1))
            selected = bool(np.all(np.abs(background - np.array([47, 54, 62])) <= 8))
            targets.append({'id': ''.join(recognized), 'row': row, 'x': 1450, 'y': y,
                            'starred': starred, 'selected': selected})
        # Duplicate IDs indicate ambiguity or a bad layout; never act on them.
        ids = [t['id'] for t in targets]
        return [t for t in targets if ids.count(t['id']) == 1]


def selected_target(rows):
    selected = [row for row in rows if row.get('selected', False)]
    return selected[0] if len(selected) == 1 else None


class TargetMemory:
    def __init__(self, ttl=3, retry_after=30):
        self.ttl = ttl
        self.retry_after = retry_after
        self.now = 0
        self.seen = {}
        self.pair_order = 0

    def update(self, rows, now):
        self.now = now
        self.seen = {key: value for key, value in self.seen.items()
                     if now-value['last'] <= self.ttl or
                     (value.get('completed_at') is not None and
                      now-value['completed_at'] <= self.retry_after)}
        for row in rows:
            state = self.seen.setdefault(row['id'], {'shots': 0, 'last': now,
                                                      'last_pair': -1, 'completed_at': None})
            if state['shots'] == 2 and (not row.get('starred', False) or
                                        now-state['completed_at'] >= self.retry_after):
                state['shots'] = 0
            state['last'] = now
            state['starred'] = row.get('starred', False)

    def choose(self, rows):
        # Finish an existing pair before switching to a newly appearing row.
        for count in (1, 0):
            candidates = [row for row in rows if self.eligible(row) and
                          self.seen[row['id']]['shots'] == count]
            # Round-robin fairness if the star has not appeared after a pair.
            target = min(candidates, key=lambda row: self.seen[row['id']]['last_pair'],
                         default=None)
            if target is not None:
                return target
        return None

    def eligible(self, row):
        state = self.seen[row['id']]
        shots = state['shots']
        retry_due = (state['completed_at'] is not None and
                     self.now-state['completed_at'] >= self.retry_after)
        return shots < 2 and (shots == 1 or retry_due or not row.get('starred', False))

    def fired(self, target_id, now=None):
        self.seen[target_id]['shots'] += 1
        if self.seen[target_id]['shots'] == 2:
            self.seen[target_id]['completed_at'] = self.now if now is None else now
            self.seen[target_id]['last_pair'] = self.pair_order
            self.pair_order += 1
