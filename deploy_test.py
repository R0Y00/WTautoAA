"""IRIS-T deployment and radar automation; optional single-match hangar entry."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import random
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import ImageGrab, Image
import pyautogui as pg
from radar_targets import TargetReader, TargetMemory, selected_target
from match_entry import EntryReader


def sequence(cooldown: float):
    return [
        ("place", "launcher_1", ("alt", "c")),
        ("place", "launcher_2", ("alt", "c")),
        ("wait", cooldown),
        ("key", "launcher_1", ("alt", "u")),
        ("place", "ammo_1", ("alt", "e")),
        ("key", "launcher_2", ("alt", "u")),
        ("place", "ammo_2", ("alt", "e")),
        ("key", "radar_vehicle", ("alt", "u")),
        ("key", "radar_screen", ("c", "r")),
    ]


def search_offsets(step=900, pitch_step=600, look_down=1200):
    """Aim at nearby ground first; subsequent rows only deepen the downward pitch.

    Positive mouse Y assumes the game's normal, non-inverted camera setting.
    Counts are configurable, not an estimate of world-space placement distance.
    """
    yield 0, look_down
    for _ in range(6):
        yield step, 0
    yield 0, pitch_step
    for _ in range(12):
        yield -step, 0
    yield 0, pitch_step
    for _ in range(12):
        yield step, 0


def placement_offsets(name, step=900, pitch_step=600, look_down=1200):
    if name == "launcher_1":
        yield from search_offsets(step, pitch_step, look_down)
        return
    # Second launcher preserves the successful pitch and moves sideways first.
    # Ammo starts after a vehicle switch, so it needs its own ground acquisition.
    yield (step, 0) if name == "launcher_2" else (0, look_down * 2)
    for _ in range(6):
        yield step, 0
    # Search both sides of the initial pitch instead of accumulating downward motion.
    yield 0, -pitch_step
    for _ in range(12):
        yield -step, 0
    yield 0, 2 * pitch_step
    for _ in range(12):
        yield step, 0


def smooth_motion(dx, dy):
    """Ease in/out without overshoot; integer deltas sum to the requested move."""
    count = max(1, math.ceil(max(abs(dx), abs(dy)) * 1.6 / 60))
    previous_x = previous_y = 0
    for index in range(1, count + 1):
        t = index / count
        progress = t * t * (3 - 2 * t)
        x, y = round(dx * progress), round(dy * progress)
        yield x - previous_x, y - previous_y
        previous_x, previous_y = x, y


def refined_offsets(offsets, rng):
    # Check partway through each horizontal sweep, then make the smaller
    # correction only if placement is still unavailable. Preserve total coverage.
    for dx, dy in offsets:
        if dx and not dy:
            first = round(dx * rng.uniform(.6, .75))
            yield first, 0
            yield dx - first, 0
        else:
            yield dx, dy


class Stopped(RuntimeError):
    pass


class RadarLost(Stopped):
    pass


class BattleEnded(Stopped):
    """A post-spawn selection/results screen interrupts the current operation."""
    pass


@lru_cache(maxsize=1)
def radar_panel_templates():
    root = Path(__file__).parent / 'assets' / 'entry'
    return tuple(np.asarray(Image.open(root / (name + '.png')).convert('L'))
                 for name in ('radar_panel_title', 'target_list_title'))


def radar_visible(frame):
    """Recognize the page even when the scanner is inactive or damaged."""
    if frame.shape[:2] == (1080, 1920):
        regions = ((605,70,755,100), (1348,70,1445,100))
        if all(cv2.matchTemplate(
                cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY),
                template, cv2.TM_CCOEFF_NORMED).max() >= .90
               for template, (x0,y0,x1,y1) in zip(radar_panel_templates(), regions)):
            return True
    # Retain the circle check for existing calibrated captures.
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (45, 150, 100), (90, 255, 255))
    hits = 0
    radius = h * .324
    margin = max(3, round(h * .006))
    for angle in np.linspace(0, 2*np.pi, 48, endpoint=False):
        x = round(w * .5 + radius * np.cos(angle))
        y = round(h * .422 + radius * np.sin(angle))
        patch = green[max(0,y-margin):min(h,y+margin+1),
                      max(0,x-margin):min(w,x+margin+1)]
        hits += bool(np.any(patch))
    return hits >= 38


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", INPUTUNION)]


class Runner:
    def __init__(self, args):
        self.args = args
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT
        self.target = None
        self.logdir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        self.logdir.mkdir(parents=True, exist_ok=True)
        pg.FAILSAFE = True
        pg.PAUSE = 0.1

    def log(self, event, **data):
        item = {"time": time.time(), "event": event, **data}
        print(json.dumps(item, ensure_ascii=False), flush=True)
        with (self.logdir / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    def down(self, vk):
        return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)

    def brief_focus_return(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.down(0x79):
                raise Stopped('F10 emergency stop')
            pg.failSafeCheck()
            if self.user32.GetForegroundWindow() == self.target:
                self.log('transient_focus_restored')
                return True
            time.sleep(.05)
        return False

    def record_focus_loss(self):
        hwnd = self.user32.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(1024)
        self.user32.GetWindowTextW(hwnd, title, len(title))
        self.log('paused', message='Return to game and press F8 to resume; F10 stops',
                 expected_hwnd=self.target, foreground_hwnd=hwnd, foreground_title=title.value)
        try:
            ImageGrab.grab().save(self.logdir / 'focus_lost.png')
        except OSError as exc:
            self.log('focus_capture_failed', reason=str(exc))

    def check(self):
        if self.down(0x79):  # F10
            raise Stopped("F10 emergency stop")
        pg.failSafeCheck()
        if self.target is not None and self.user32.GetForegroundWindow() != self.target:
            if getattr(self, "sending", False):
                raise Stopped("Focus lost during key chord; keys released, restart required")
            if self.brief_focus_return():
                return self.check()
            self.record_focus_loss()
            while self.down(0x77):
                if self.down(0x79):
                    raise Stopped("F10 emergency stop")
                time.sleep(.05)
            while self.user32.GetForegroundWindow() != self.target or not self.down(0x77):
                if self.down(0x79):
                    raise Stopped("F10 emergency stop")
                time.sleep(.05)
            while self.down(0x77):
                if self.down(0x79):
                    raise Stopped("F10 emergency stop")
                time.sleep(.05)
            self.log("resumed")
        if getattr(self, 'monitor_battle', False) and time.monotonic() >= getattr(self, 'next_battle_check', 0):
            self.next_battle_check = time.monotonic() + .3
            self.inspect_battle_frame(np.asarray(ImageGrab.grab().convert('RGB')))

    def inspect_battle_frame(self, frame):
        if not getattr(self, 'monitor_battle', False):
            return
        if not hasattr(self, 'battle_reader'):
            self.battle_reader = EntryReader()
        if (self.battle_reader.selection(frame) or self.battle_reader.results(frame) or
                self.battle_reader.destruction_countdown(frame) or
                self.battle_reader.destruction_replay(frame)):
            self.monitor_battle = False
            raise BattleEnded('Post-spawn selection/results detected; current operation interrupted')

    def recover_battle_end(self):
        self.monitor_battle = False
        self.log('battle_interrupted', message='Inputs stopped; waiting for destruction, selection or results transition')
        if not self.handle_radar_loss():
            raise Stopped('Battle-end screen was not confirmed; interrupted deployment will not resume')

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.05, max(0, end - time.monotonic())))

    def confirm(self, message):
        self.log("confirmation_required", message=message)
        # Require a new press, not a held key from the preceding step.
        while self.down(0x77):
            self.wait(0.05)
        deadline = time.monotonic() + self.args.confirm_timeout
        while not self.down(0x77):  # F8
            self.wait(0.05)
            if time.monotonic() > deadline:
                raise Stopped("Confirmation timed out")
        while self.down(0x77):
            self.wait(0.05)

    def hotkey(self, keys):
        self.check()
        codes = {"alt": 0x38, "ctrl": 0x1D, "c": 0x2E, "u": 0x16,
                 "e": 0x12, "r": 0x13, "enter": 0x1C, "esc": 0x01}
        pressed = []
        self.sending = True
        try:
            for key in keys:
                self.send_key(codes[key], False)
                pressed.append(key)
                self.wait(0.15)
            self.wait(.15)
        finally:
            for key in reversed(pressed):
                self.send_key(codes[key], True)
            self.sending = False

    def send_key(self, code, up):
        event = INPUT(type=1, ki=KEYBDINPUT(0, code, 0x0008 | (0x0002 if up else 0), 0, 0))
        if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
            raise Stopped("SendInput failed; check game/script privilege levels")

    def move_view(self, dx, dy):
        if getattr(self.args, 'natural_motion', False):
            interval = random.uniform(.012, .018)
            for x, y in smooth_motion(dx, dy):
                self.check()
                event = INPUT(type=0, mi=MOUSEINPUT(x, y, 0, 0x0001, 0, 0))
                if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
                    raise Stopped('Relative mouse input failed')
                self.wait(interval)
            return
        # Deliver movement over multiple game frames, rather than one large jump.
        count = max(1, (max(abs(dx), abs(dy)) + 59) // 60)
        sent_x = sent_y = 0
        for index in range(1, count + 1):
            self.check()
            x, y = round(dx * index / count), round(dy * index / count)
            event = INPUT(type=0, mi=MOUSEINPUT(x-sent_x, y-sent_y, 0, 0x0001, 0, 0))
            if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
                raise Stopped("Relative mouse input failed")
            sent_x, sent_y = x, y
            self.wait(.016)

    def click(self):
        self.check()
        hold = self.args.click_hold + (random.uniform(0, .04) if getattr(self.args, 'natural_motion', False) else 0)
        self.sending = True
        try:
            self.mouse_button(0x0002)  # LEFTDOWN
            self.wait(hold)
        finally:
            try:
                self.mouse_button(0x0004)  # Always release, including on F10/focus loss.
            finally:
                self.sending = False
        self.log("left_click_sent", hold_seconds=hold)

    def placement_search(self, name):
        offsets = placement_offsets(name, self.args.search_step, self.args.pitch_step, self.args.look_down)
        if getattr(self.args, 'natural_motion', False):
            return refined_offsets(offsets, random)
        return offsets

    def settle(self, minimum, spread=.3):
        seconds = minimum + (random.uniform(0, spread) if getattr(self.args, 'natural_motion', False) else 0)
        self.wait(seconds)

    def mouse_button(self, flag):
        event = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flag, 0, 0))
        if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
            raise Stopped("Mouse button input failed")

    def right_double_click(self):
        for index in range(2):
            self.check()
            self.sending = True
            try:
                self.mouse_button(0x0008)
                self.wait(.06)
            finally:
                try:
                    self.mouse_button(0x0010)
                finally:
                    self.sending = False
            if index == 0:
                self.wait(.08)

    def engage_targets(self):
        while True:
            try:
                return self.track_targets()
            except RadarLost:
                if self.handle_radar_loss():
                    return

    def handle_radar_loss(self):
        reader = EntryReader()
        self.log('radar_lost_waiting_for_selection')
        stable = 0
        for poll in range(180):
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            if poll % 10 == 0:
                Image.fromarray(frame).save(self.logdir / f'battle_recovery_{poll:03d}.png')
            if reader.results(frame):
                return True
            if reader.destruction_countdown(frame) or reader.destruction_replay(frame):
                stable = 0
                self.wait(.5)
                continue
            if radar_visible(frame):
                return False
            stable = stable + 1 if reader.selection(frame) else 0
            if stable >= 3:
                self.monitor_battle = False
                self.log('post_battle_selection_detected')
                self.return_to_base(reader)
                return True
            self.wait(.5)
        Image.fromarray(frame).save(self.logdir / 'battle_recovery_timeout.png')
        raise Stopped('Radar lost; no post-battle selection recognized; recovery screenshots saved')

    def return_to_base(self, reader):
        self.hotkey(('esc',))
        for name, predicate, position in (
                ('return_base', reader.return_menu, (960, 657)),
                ('return_yes', reader.return_confirmation, (894, 623))):
            for _ in range(20):
                self.wait(.5)
                frame = self.snapshot(name + '_check')
                if name == 'return_base' and reader.selection_card(frame):
                    self.check()
                    pg.moveTo(1800, 450)
                    self.click()
                    self.hotkey(('esc',))
                    continue
                if predicate(frame):
                    self.check()
                    pg.moveTo(*position)
                    self.click()
                    self.log(name + '_clicked')
                    break
            else:
                if name == 'return_yes' and not reader.return_menu(frame) and not reader.selection(frame):
                    self.log('return_to_base_requested',
                             message='No confirmation appeared; menu closed; hangar arrival not verified')
                    return
                raise Stopped(f'{name}: expected screen not recognized')
        self.wait(1)
        self.snapshot('after_return_confirmation')
        self.log('return_to_base_requested', message='Yes clicked; hangar arrival not verified')

    def track_targets(self):
        reader = TargetReader()
        memory = TargetMemory(self.args.target_memory)
        self.log('radar_tracking_started', calibrated_ids=sorted({k.split('_')[0] for k in reader.templates}),
                 message='Input attempts counted, not confirmed missile launches; F10 stops')
        next_shot = 0
        previous_ids = None
        capture_number = 0
        while True:
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            if not radar_visible(frame):
                raise RadarLost('Radar interface lost; target actions stopped')
            rows = reader.read(frame)
            visible_ids = tuple((r['id'], r['starred']) for r in rows)
            if visible_ids != previous_ids:
                self.log('visible_target_ids', ids=visible_ids)
                Image.fromarray(frame).save(self.logdir / f'target_list_{capture_number:04d}.png')
                capture_number += 1
                previous_ids = visible_ids
            memory.update(rows, time.monotonic())
            target = memory.choose(rows)
            if target is None or time.monotonic() < next_shot:
                self.wait(.08)
                continue
            # Reacquire by ID from a fresh frame; never reuse a disappeared row.
            self.check()
            fresh = np.asarray(ImageGrab.grab().convert('RGB'))
            if not radar_visible(fresh):
                raise RadarLost('Radar interface lost before target selection')
            current = next((r for r in reader.read(fresh) if r['id'] == target['id']), None)
            if current is None:
                self.log('target_skipped', target_id=target['id'], reason='not recognized in fresh frame')
                continue
            if not memory.eligible(current):
                self.log('target_skipped', target_id=current['id'], reason='already starred or pair finished')
                continue
            Image.fromarray(fresh).save(self.logdir / f'target_before_select_{capture_number:04d}.png')
            self.check()
            pg.moveTo(current['x'], current['y'])
            self.click()
            self.wait(.08)
            # Selection follows the target, not its former list position.
            check_frame = np.asarray(ImageGrab.grab().convert('RGB'))
            Image.fromarray(check_frame).save(self.logdir / f'target_after_select_{capture_number:04d}.png')
            capture_number += 1
            if not radar_visible(check_frame):
                raise RadarLost('Radar interface lost after selection')
            selected = selected_target(reader.read(check_frame))
            if selected is None or selected['id'] != current['id'] or not memory.eligible(selected):
                self.log('target_skipped', target_id=current['id'],
                         reason='expected target highlight not confirmed or already starred',
                         recognized_after=reader.read(check_frame))
                continue
            self.log('target_selection_verified', target_id=selected['id'], row=selected['row'])
            self.fire_selected_pair(current['id'], reader, memory)
            next_shot = time.monotonic() + .5
            self.wait(.15)
            self.snapshot(f'target_after_fire_{capture_number:04d}')

    def fire_selected_pair(self, target_id, reader, memory):
        while memory.seen[target_id]['shots'] < 2:
            self.check()
            self.right_double_click()
            memory.fired(target_id, time.monotonic())
            self.log('target_fire_input', target_id=target_id,
                     attempt=memory.seen[target_id]['shots'])
            if memory.seen[target_id]['shots'] >= 2:
                return
            self.wait(.5)
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            if not radar_visible(frame):
                raise RadarLost('Radar interface lost during selected target pair')
            selected = selected_target(reader.read(frame))
            if selected is not None and selected['id'] != target_id:
                raise Stopped('Selected target changed during pair; second input cancelled')
            # A brief list dropout does not clear the already verified selection.
            # No cursor movement or left click between the two right-double-clicks.

    def snapshot(self, name):
        self.check()
        shot = ImageGrab.grab()
        shot.save(self.logdir / f"{name}.png")
        frame = np.asarray(shot.convert("RGB"))
        self.inspect_battle_frame(frame)
        return frame

    def preview(self, frame, kind="launcher"):
        """Experimental color evidence only; never equates disappearance with success."""
        h, w = frame.shape[:2]
        # Excludes radar, minimap and bottom action bar in the supplied 1920x1080 UI.
        if kind == "ammo":
            # Ammo footprint surrounds the aiming ring (~50%, 27.5%), not the
            # unrelated green triangle above/left of it in the failed capture.
            crop = frame[int(h * .24):int(h * .33), int(w * .475):int(w * .525)]
        else:
            # Battle control-point diamonds occupy the top ~75px; they are
            # unrelated to placement validity and may be red over a green ghost.
            crop = frame[int(h * .08):int(h * .55), int(w * .44):int(w * .56)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        green = cv2.inRange(hsv, (50, 180, 150), (85, 255, 255))
        rgb = crop.astype(np.int16)
        synthetic_green = (rgb[:,:,1] - rgb[:,:,0] > 65) & (rgb[:,:,1] - rgb[:,:,2] > 45)
        green[~synthetic_green] = 0
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 150, 100), (10, 255, 255)),
                            cv2.inRange(hsv, (170, 150, 100), (179, 255, 255)))
        scale = w * h / (1920 * 1080)
        def evidence(mask, is_green=False):
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            # Require launcher-sized outline evidence for BOTH colors. Small
            # red tail lights must not veto an otherwise valid green outline.
            min_area = (100 if is_green else 20) if kind == "ammo" else 80
            min_height = (12 if is_green else 10) if kind == "ammo" else 70
            return any(area >= min_area * scale and height >= min_height * h / 1080
                       for x, y, width, height, area in stats[1:])
        if evidence(red):
            return "red"
        if evidence(green, True):
            return "green"
        if kind == 'ammo' and (h, w) == (1080, 1920):
            # Snow washes the ghost almost white. Only use weak chroma when
            # the nearby ground is bright/neutral and the placement UI is visible.
            background = np.median(rgb, axis=(0, 1))
            if background.min() >= 180 and np.ptp(background) <= 25:
                template = cv2.imread(str(Path(__file__).parent / 'assets' / 'ammo_placement_hint.png'), 0)
                hint = cv2.cvtColor(frame[750:940, 925:1115], cv2.COLOR_RGB2GRAY)
                if cv2.matchTemplate(hint, template, cv2.TM_CCOEFF_NORMED).max() >= .75:
                    def pale_component(channel):
                        other = [c for c in range(3) if c != channel]
                        mask = ((rgb[:,:,channel] - rgb[:,:,other[0]] >= 3) &
                                (rgb[:,:,channel] - rgb[:,:,other[1]] >= 3) &
                                (rgb.min(axis=2) >= 120)).astype(np.uint8)
                        stats = cv2.connectedComponentsWithStats(mask)[2][1:]
                        return any(area >= 70 and width >= 22 and height >= 5
                                   for x, y, width, height, area in stats)
                    if pale_component(0):
                        return 'red'
                    if pale_component(1):
                        return 'green'
        return 'unknown'

    def auto_attempt(self, name):
        if name.startswith('ammo'):
            return self.attempt_ammo(name)
        offsets = self.placement_search(name)
        kind = "ammo" if name.startswith("ammo") else "launcher"
        for index, (dx, dy) in enumerate(offsets):
            self.check()
            if dx or dy:
                self.move_view(dx, dy)
            self.settle(.4, .15)
            first = self.preview(self.snapshot(f"{name}_candidate_{index}"), kind)
            self.wait(.25)
            second = self.preview(np.asarray(ImageGrab.grab().convert("RGB")), kind)
            self.log("search_candidate", object=name, index=index, dx=dx, dy=dy,
                     first=first, second=second)
            if first == second == "green":
                self.check()
                self.click()
                self.log("placement_clicked_unverified", object=name)
                return True
        self.log("no_confident_preview", object=name)
        return False

    @staticmethod
    def ammo_hint(frame):
        if frame.shape[:2] != (1080, 1920):
            return False
        template = cv2.imread(str(Path(__file__).parent / 'assets' / 'ammo_placement_hint.png'), 0)
        # Capture-point notifications can move the placement prompt upward.
        hint = cv2.cvtColor(frame[750:940,925:1115], cv2.COLOR_RGB2GRAY)
        if cv2.matchTemplate(hint, template, cv2.TM_CCOEFF_NORMED).max() >= .75:
            return True
        dim = cv2.imread(str(Path(__file__).parent / 'assets' / 'ammo_placement_hint_dim.png'), 0)
        return bool(cv2.matchTemplate(hint, dim, cv2.TM_CCOEFF_NORMED).max() >= .85)

    def confirm_ammo_mode(self, frame, name):
        # Notifications can briefly dim/move the prompt. Recheck without input;
        # wait/snapshot still perform the global death and F10 checks.
        if self.ammo_hint(frame):
            return True
        self.log('ammo_mode_rechecking', object=name)
        for poll in range(6):
            self.wait(.2)
            frame = self.snapshot(f'{name}_mode_recheck_{poll}')
            if self.ammo_hint(frame):
                return True
        return False

    @staticmethod
    def ammo_icon(frame):
        if frame.shape[:2] != (1080, 1920):
            return None
        template_rgb = np.asarray(Image.open(Path(__file__).parent / 'assets' / 'ammo_key_label.png').convert('RGB'))
        def key_letters(rgb):
            return cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (15,80,80), (40,255,255))
        # Match yellow glyphs, not the changing selected/disabled button background.
        template = key_letters(template_rgb)
        band = key_letters(frame[975:1002,600:1350])
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(band, template, cv2.TM_CCOEFF_NORMED))
        if score < .8:
            return None
        center = 600 + location[0] + template.shape[1] // 2
        return frame[1005:1056, center-24:center+24].astype(float)

    @staticmethod
    def ammo_disabled(frame, baseline):
        if frame.shape[:2] != (1080, 1920) or baseline.shape[:2] != (1080, 1920):
            return False
        current = Runner.ammo_icon(frame)
        before = Runner.ammo_icon(baseline)
        if current is None or before is None:
            return False
        return bool(before.mean() >= 70 and current.mean() < before.mean() * .6 and
                    np.mean(np.ptp(current, axis=2)) < np.mean(np.ptp(before, axis=2)) * .7)

    def attempt_ammo(self, name):
        baseline = self.snapshot(name + '_icon_baseline')
        if not self.confirm_ammo_mode(baseline, name):
            raise BattleEnded(f'{name}: placement mode absent; wait for possible vehicle-loss transition')
        for index, (dx, dy) in enumerate(self.placement_search(name)):
            self.check()
            if dx or dy:
                self.move_view(dx, dy)
            self.settle(.4, .15)
            for attempt in range(2):
                frame = self.snapshot(f'{name}_attempt_{index}_{attempt}')
                if not self.confirm_ammo_mode(frame, name):
                    raise BattleEnded(f'{name}: placement mode lost; wait for possible vehicle-loss transition')
                self.click()
                self.log('ammo_placement_attempt', object=name, candidate=index, attempt=attempt + 1)
                stable = 0
                for poll in range(10):
                    self.wait(.1)
                    frame = self.snapshot(f'{name}_icon_check')
                    disabled = self.ammo_disabled(frame, baseline)
                    stable = stable + 1 if disabled and not self.ammo_hint(frame) else 0
                    if stable >= 2:
                        self.log('ammo_hud_change_detected', object=name)
                        return True
                    # No more clicks while the placement UI is absent; keep
                    # polling briefly to catch the transition into cooldown.
                if not self.confirm_ammo_mode(frame, name):
                    raise BattleEnded(f'{name}: placement closed without cooldown; wait for possible vehicle-loss transition')
        self.log('ammo_placement_unverified', object=name)
        return False

    def slot(self, frame, name):
        h, w = frame.shape[:2]
        # Launcher slots in the radar vehicle HUD, based on the supplied 1080p capture.
        left, right = (.575, .608) if name == "launcher_1" else (.611, .644)
        return frame[int(h*.932):int(h*.982), int(w*left):int(w*right)].astype(float)

    def verify_launcher(self, name, before):
        baseline = self.slot(before, name)
        stable = 0
        for attempt in range(20):
            self.wait(.5)
            frame = self.snapshot(f"{name}_verify_{attempt}")
            changed = np.abs(self.slot(frame, name) - baseline).mean() > 12
            stable = stable + 1 if changed and self.preview(frame) == "unknown" else 0
            if stable >= 3:
                self.log("launcher_hud_change_detected", object=name)
                return
        raise Stopped(f"{name}: cannot verify launcher HUD changed after click; screenshots saved")

    def place(self, name, keys):
        self.log("placement_started", object=name, keys=keys)
        before = self.snapshot(f"{name}_before")
        self.hotkey(keys)
        self.wait(.5)
        self.snapshot(f"{name}_opened")
        if self.args.mode == "preview":
            if not self.auto_attempt(name):
                raise Stopped(f"{name}: no valid central preview; not clicking or switching blindly")
            if name.startswith("launcher"):
                self.verify_launcher(name, before)
            else:
                self.log("building_wait", seconds=self.args.build_wait,
                         message="Timed wait; completion is not visually verified yet")
                self.wait(self.args.build_wait)
        else:
            self.confirm(f"{name}: finish placement/building manually if needed; press F8 ONLY after success")
        self.snapshot(f"{name}_confirmed")
        self.log("placement_step_finished", object=name, mode=self.args.mode)

    def run(self):
        self.log("armed", mode=self.args.mode,
                 message="F8 starts, F10 stops", from_hangar=self.args.from_hangar)
        self.await_start()
        hwnd = self.user32.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(1024)
        self.user32.GetWindowTextW(hwnd, title, len(title))
        if self.args.window_title.casefold() not in title.value.casefold():
            raise Stopped(f"Wrong foreground window: {title.value!r}")
        self.target = hwnd
        self.log("target_locked", title=title.value)
        if not self.args.auto_start:
            self.wait(self.args.start_delay)
        if self.args.from_hangar:
            self.run_match_loop()
            return
        if self.args.radar_only:
            self.engage_targets()
            self.log('finished')
            return
        self.deploy_and_engage()
        self.log('finished')

    def await_start(self):
        if self.args.auto_start:
            self.log('auto_start_countdown', seconds=5,
                     message='Switch to War Thunder within 5 seconds; F10 stops')
            self.wait(5)
        else:
            self.confirm('On hangar main screen with IRIS-T selected: F8 starts matchmaking'
                         if self.args.from_hangar else 'In radar vehicle or radar page: F8 starts')

    def run_match_loop(self):
        cycle = 0
        while True:
            self.check()
            cycle += 1
            self.log('match_cycle_started', cycle=cycle)
            try:
                self.enter_match()
                self.deploy_and_engage()
            except BattleEnded:
                self.recover_battle_end()
            finally:
                self.monitor_battle = False
            self.wait_for_results()

    def wait_for_results(self):
        reader = EntryReader()
        stable = 0
        for _ in range(240):
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            lobby_ready = (reader.results(frame) or reader.tier_complete(frame)
                           or reader.research(frame) or reader.hangar(frame))
            stable = stable + 1 if lobby_ready else 0
            if stable >= 3:
                self.snapshot('results_ready_for_next_match')
                self.log('results_ready_for_next_match')
                return
            self.wait(.5)
        raise Stopped('Return requested but results screen not recognized; no matchmaking input sent')

    def deploy_and_engage(self):
        self.monitor_battle = True
        try:
            self.perform_deployment()
        except BattleEnded:
            self.recover_battle_end()
        finally:
            self.monitor_battle = False

    def perform_deployment(self):
        for step in sequence(self.args.cooldown):
            if step[0] == "place":
                self.place(step[1], step[2])
            elif step[0] == "wait":
                self.log("cooldown", seconds=step[1])
                self.settle(step[1], .8)
            else:
                self.log("switch", target=step[1], keys=step[2])
                if step[1] == "radar_screen":
                    self.open_radar()
                    continue
                self.hotkey(step[2])
                self.settle(1)
                self.snapshot(step[1] + "_switch")
                if self.args.mode == "guided":
                    self.confirm(f"Confirm {step[1]} is active; F8 to continue, F10 if incorrect")
        if self.args.engage:
            self.engage_targets()
        self.log("finished")

    def enter_match(self):
        reader = EntryReader()
        frame = self.snapshot('entry_start')
        if frame.shape[:2] != (1080, 1920):
            raise Stopped('Match entry requires 1920x1080')
        if reader.selection(frame) or reader.spawned(frame) or radar_visible(frame):
            raise Stopped('--from-hangar expects the main hangar, not a battle screen')
        # The user's F8 on the hangar is the start signal; never repeatedly press
        # Enter while waiting for matchmaking/loading.
        lobby_actions = set()
        self.match_retry_count = 0
        self.last_match_error_click = float('-inf')
        if not self.advance_lobby(reader, frame, lobby_actions):
            raise Stopped('Expected hangar, results or research screen before matchmaking')
        deadline = time.monotonic() + self.args.match_timeout
        ready_since = None
        while time.monotonic() < deadline:
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            if self.advance_lobby(reader, frame, lobby_actions):
                if 'match_retry_reset' in lobby_actions:
                    lobby_actions.discard('match_retry_reset')
                    deadline = time.monotonic() + self.args.match_timeout
                self.wait(.5)
                continue
            if reader.spawned(frame):
                self.log('spawn_detected', automatic=True)
                break
            if reader.ready(frame):
                if ready_since is None:
                    ready_since = time.monotonic()
                    self.snapshot('selection_ready')
                    self.log('selection_ready', delay=self.args.selection_delay)
                if time.monotonic() - ready_since >= self.args.selection_delay:
                    self.hotkey(('enter',))
                    self.log('deployment_requested')
                    self.wait_for_spawn(reader)
                    break
            else:
                ready_since = None
            self.wait(.5)
        else:
            self.snapshot('matchmaking_timeout')
            raise Stopped('Timed out waiting for IRIS-T selection page; no repeated Enter sent')
        self.monitor_battle = True
        self.snapshot('spawn_confirmed')
        self.log('teammate_clearance_wait', seconds=self.args.spawn_wait)
        self.settle(self.args.spawn_wait, .8)
        if not reader.spawned(self.snapshot('before_deployment')):
            raise Stopped('Spawn HUD lost during clearance wait')

    def advance_lobby(self, reader, frame, actions):
        # One input per recognized page; never repeat Enter on a loading screen.
        if reader.tier_complete(frame):
            if 'tier_complete' not in actions:
                self.check()
                pg.moveTo(1301, 894)
                self.click()
                actions.add('tier_complete')
                actions.discard('research')
                actions.discard('hangar')
                self.log('tier_complete_dismissed')
            return True
        actions.discard('tier_complete')
        if reader.matchmaking_error(frame):
            now = time.monotonic()
            if now - getattr(self, 'last_match_error_click', float('-inf')) >= 3:
                self.check()
                pg.moveTo(1092, 581)
                self.click()
                self.last_match_error_click = time.monotonic()
                self.match_retry_count = getattr(self, 'match_retry_count', 0) + 1
                actions.discard('hangar')
                actions.add('match_retry_reset')
                self.log('matchmaking_error_dismissed', attempt=self.match_retry_count)
                self.wait(3)
            return True
        if reader.research(frame):
            if 'research' not in actions:
                self.check()
                pg.moveTo(1015, 770)
                self.click()
                actions.add('research')
                actions.discard('hangar')
                self.log('research_points_allocated_input')
            return True
        if reader.completed(frame):
            state, event = 'completed', 'results_confirmed_input'
        elif reader.results(frame):
            state, event = 'results', 'results_join_input'
        elif reader.hangar(frame):
            state, event = 'hangar', 'matchmaking_started'
        else:
            return False
        if state not in actions:
            self.hotkey(('enter',))
            actions.add(state)
            self.log(event)
        return True

    def wait_for_spawn(self, reader):
        deadline = time.monotonic() + self.args.spawn_timeout
        stable = 0
        while time.monotonic() < deadline:
            self.check()
            frame = np.asarray(ImageGrab.grab().convert('RGB'))
            stable = stable + 1 if reader.spawned(frame) else 0
            if stable >= 3:
                self.log('spawn_detected')
                return
            self.wait(.5)
        self.snapshot('spawn_timeout')
        raise Stopped('Deployment requested but spawn not confirmed; stopping')

    def open_radar(self):
        self.wait(1)
        for attempt in range(2):
            if radar_visible(self.snapshot(f"radar_before_{attempt}")):
                self.log("radar_screen_verified")
                return
            self.hotkey(("c", "r"))
            for poll in range(8):
                self.wait(.5)
                if radar_visible(self.snapshot(f"radar_attempt_{attempt}_{poll}")):
                    self.log("radar_screen_verified")
                    return
        raise Stopped("C+R sent twice but radar interface not detected; screenshots saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print sequence; send no input")
    parser.add_argument('--fixed-motion', dest='natural_motion', action='store_false',
                        help='Use original fixed movement and timing instead of smooth varied placement')
    parser.add_argument("--mode", choices=["guided", "preview"], default="guided")
    parser.add_argument('--engage', action='store_true', help='Run target loop after deployment in test drive')
    parser.add_argument('--radar-only', action='store_true', help='Start directly on the open radar screen')
    parser.add_argument('--from-hangar', action='store_true', help='Queue one match, deploy, then run radar actions')
    parser.add_argument('--match-timeout', type=float, default=600)
    parser.add_argument('--spawn-timeout', type=float, default=120)
    parser.add_argument('--selection-delay', type=float, default=3)
    parser.add_argument('--spawn-wait', type=float, default=10)
    parser.add_argument('--target-memory', type=float, default=3, help='Remember disappeared IDs for these seconds')
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--start-delay", type=float, default=2)
    parser.add_argument('--auto-start', action='store_true',
                        help='Start after 5 seconds without F8; ignores --start-delay')
    parser.add_argument("--build-wait", type=float, default=5,
                        help="Ammo building wait (default 5s, based on user test)")
    parser.add_argument("--click-hold", type=float, default=.20,
                        help="Seconds to hold LEFT mouse button before releasing")
    parser.add_argument("--search-step", type=int, default=900,
                        help="Horizontal mouse counts per search step (not screen pixels/degrees)")
    parser.add_argument("--pitch-step", type=int, default=600,
                        help="Additional downward mouse counts between ground sweeps")
    parser.add_argument("--look-down", type=int, default=1200,
                        help="Downward mouse counts BEFORE each placement search")
    parser.add_argument("--confirm-timeout", type=float, default=180)
    parser.add_argument("--window-title", default="War Thunder")
    args = parser.parse_args()
    if args.from_hangar and args.radar_only:
        parser.error('--from-hangar and --radar-only cannot be combined')
    if min(args.match_timeout, args.spawn_timeout) <= 0 or min(args.selection_delay, args.spawn_wait) < 0:
        parser.error('Entry timeouts must be positive and delays nonnegative')
    if args.from_hangar:
        args.mode = 'preview'
        args.engage = True
    if args.cooldown < 15 or args.start_delay < 0 or args.confirm_timeout <= 0 or args.build_wait < 1:
        parser.error("cooldown must be >=15, start-delay >=0, confirm-timeout >0, build-wait >=1")
    if not 100 <= args.search_step <= 3000 or not 0 <= args.pitch_step <= 1000:
        parser.error("search-step must be 100..3000 and pitch-step 0..1000")
    if not 0 <= args.look_down <= 6000:
        parser.error("look-down must be 0..6000")
    if not .05 <= args.click_hold <= 1:
        parser.error("click-hold must be 0.05..1 seconds")
    if args.target_memory <= 0:
        parser.error('target-memory must be positive')
    if args.dry_run:
        plan = sequence(args.cooldown)
        if args.from_hangar:
            plan = [('key', 'matchmaking', ('enter',)), ('wait_for', 'IRIS-T selection'),
                    ('wait', args.selection_delay), ('key', 'spawn', ('enter',)),
                    ('wait_for', 'spawn HUD'), ('wait', args.spawn_wait)] + plan
        if args.engage or args.radar_only:
            plan = ([('radar_actions', 'until F10 or error')] if args.radar_only else
                    plan + [('radar_actions', 'until F10 or error')])
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    runner = Runner(args)
    try:
        runner.run()
    except (Stopped, pg.FailSafeException, KeyboardInterrupt) as exc:
        runner.log("stopped", reason=str(exc) or type(exc).__name__)
    except Exception as exc:
        runner.log("error", reason=repr(exc))
        raise


if __name__ == "__main__":
    main()
