import tempfile
from capture_helpers import open_capture
import unittest
import random
from pathlib import Path
from PIL import Image
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from deploy_test import Runner, Stopped, sequence, search_offsets, placement_offsets, radar_visible
from deploy_test import smooth_motion, refined_offsets, BattleEnded


class DeploymentTests(unittest.TestCase):
    def test_terrain_rejection_keeps_ammo_placement_active(self):
        root = Path('runs/20261113-141249')
        for name in ('ammo_1_attempt_3_0', 'battle_recovery_timeout'):
            frame = np.array(open_capture(root / (name + '.png')).convert('RGB'))
            self.assertTrue(Runner.ammo_hint(frame))
        self.assertFalse(Runner.ammo_hint(np.zeros((1080,1920,3), dtype=np.uint8)))

    def test_ammo_mode_transient_miss_rechecks_without_click(self):
        runner = Runner.__new__(Runner)
        runner.ammo_hint = Mock(side_effect=[False, False, True])
        runner.log = Mock()
        runner.wait = Mock()
        runner.snapshot = Mock()
        runner.click = Mock()
        self.assertTrue(runner.confirm_ammo_mode(None, 'ammo_1'))
        self.assertEqual(runner.snapshot.call_count, 2)
        runner.click.assert_not_called()

    def test_radar_panel_without_scan_circle(self):
        frame = np.array(open_capture(
            'runs/20261113-104850/battle_recovery_timeout.png').convert('RGB'))
        self.assertTrue(radar_visible(frame))
        # Neither title alone is enough to establish the radar page.
        for rect in ((605,70,755,100), (1348,70,1445,100)):
            altered = frame.copy()
            x0,y0,x1,y1 = rect
            altered[y0:y1,x0:x1] = 0
            self.assertFalse(radar_visible(altered))

    def test_ammo_key_not_confused_with_alt_u(self):
        root = Path('runs/20261113-094712')
        if not root.exists():
            self.skipTest('Local capture unavailable')
        frames = [np.array(open_capture(root / (name+'.png')).convert('RGB')) for name in
                  ('ammo_1_icon_baseline', 'ammo_1_attempt_0_0', 'ammo_1_icon_check', 'battle_recovery_timeout')]
        for frame in frames:
            np.testing.assert_array_equal(Runner.ammo_icon(frame), frame[1005:1056,1005:1053])
        self.assertFalse(Runner.ammo_disabled(frames[1], frames[0]))
        self.assertTrue(Runner.ammo_disabled(frames[2], frames[0]))
        self.assertFalse(Runner.ammo_disabled(frames[3], frames[0]))

    def test_destroyed_vehicle_during_ammo_enters_recovery(self):
        root = Path('runs/20261113-093945')
        if not root.exists():
            self.skipTest('Local capture unavailable')
        before = np.array(open_capture(root / 'ammo_2_icon_baseline.png').convert('RGB'))
        after = np.array(open_capture(root / 'ammo_2_attempt_0_0.png').convert('RGB'))
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(search_step=900, pitch_step=600, look_down=1200)
        for name in ('check', 'wait', 'move_view', 'click', 'log'):
            setattr(runner, name, Mock())
        runner.snapshot = Mock(side_effect=[before] + [after] * 7)
        with self.assertRaises(BattleEnded):
            runner.attempt_ammo('ammo_2')
        runner.click.assert_not_called()

    def test_ammo_icon_shifted_by_extra_action_slot(self):
        root = Path('runs/20261113-042903')
        if not root.exists():
            self.skipTest('Local capture unavailable')
        before = np.array(open_capture(root / 'ammo_1_icon_baseline.png').convert('RGB'))
        after = np.array(open_capture(root / 'ammo_1_icon_check.png').convert('RGB'))
        self.assertFalse(Runner.ammo_disabled(before, before))
        self.assertTrue(Runner.ammo_disabled(after, before))
        after[975:1002,600:1350] = 0
        self.assertFalse(Runner.ammo_disabled(after, before))

    def test_smooth_motion_preserves_distance_and_bounds(self):
        for dx, dy in ((900, 0), (-900, 0), (0, 2400), (153, -91), (0, 0)):
            steps = list(smooth_motion(dx, dy))
            self.assertEqual(sum(x for x, _ in steps), dx)
            self.assertEqual(sum(y for _, y in steps), dy)
            self.assertTrue(all(max(abs(x), abs(y)) <= 60 for x, y in steps))
        steps = list(smooth_motion(900, 0))
        self.assertLess(steps[0][0], steps[len(steps)//2][0])
        self.assertLess(steps[-1][0], steps[len(steps)//2][0])

    def test_refined_search_preserves_pitch_and_coverage(self):
        for name in ('launcher_1', 'launcher_2', 'ammo_1'):
            original = list(placement_offsets(name))
            refined = list(refined_offsets(original, random.Random(7)))
            self.assertEqual(sum(x for x, _ in refined), sum(x for x, _ in original))
            self.assertEqual([y for _, y in refined if y], [y for _, y in original if y])
            self.assertLessEqual(max(abs(x) for x, _ in refined), 675)
            if name == 'launcher_2':
                self.assertEqual(refined[0][1], 0)

    def test_varied_wait_never_shortens_minimum(self):
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(natural_motion=True)
        runner.wait = Mock()
        for _ in range(20):
            runner.settle(15, .8)
        self.assertTrue(all(15 <= c.args[0] <= 15.8 for c in runner.wait.call_args_list))

    def test_single_line_ammo_prompt_near_bottom(self):
        path = Path('runs/20261113-020102/ammo_2_icon_baseline.png')
        if not path.exists():
            self.skipTest('Local capture unavailable')
        frame = np.array(open_capture(path).convert('RGB'))
        self.assertTrue(Runner.ammo_hint(frame))
        frame[750:940,925:1115] = 0
        self.assertFalse(Runner.ammo_hint(frame))

    def test_ammo_prompt_shifted_by_capture_notification(self):
        path = Path('runs/20261113-015038/ammo_1_icon_baseline.png')
        if not path.exists():
            self.skipTest('Local capture unavailable')
        frame = np.array(open_capture(path).convert('RGB'))
        self.assertTrue(Runner.ammo_hint(frame))
        frame[775:900,925:1115] = 0
        self.assertFalse(Runner.ammo_hint(frame))

    def test_ammo_cooldown_reference(self):
        root = Path(tempfile.gettempdir())
        frames = [np.array(open_capture(root / f'codex-clipboard-{key}.png').convert('RGB'))
                  for key in ('d7367cd3-2714-44e0-8529-7f19e4b6f5ec',
                              '9590d1bd-b902-4827-9128-534415bd9188',
                              '4ba0c2f6-3e61-4ba9-b967-a7366e63d3f7')]
        self.assertTrue(Runner.ammo_hint(frames[0]))
        self.assertTrue(Runner.ammo_hint(frames[1]))
        self.assertFalse(Runner.ammo_hint(frames[2]))
        self.assertFalse(Runner.ammo_disabled(frames[1], frames[0]))
        self.assertTrue(Runner.ammo_disabled(frames[2], frames[0]))

    def test_ammo_stops_clicking_after_gray_transition(self):
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(search_step=900, pitch_step=600, look_down=1200)
        for name in ('check', 'move_view', 'wait', 'log', 'click'):
            setattr(runner, name, Mock())
        baseline, gray = object(), object()
        runner.snapshot = Mock(side_effect=[baseline, baseline, gray, gray])
        runner.ammo_hint = lambda frame: frame is baseline
        runner.ammo_disabled = lambda frame, before: frame is gray
        self.assertTrue(runner.attempt_ammo('ammo_1'))
        runner.click.assert_called_once()

    def test_ammo_does_not_click_outside_placement(self):
        runner = Runner.__new__(Runner)
        runner.log = Mock()
        runner.wait = Mock()
        runner.snapshot = Mock(return_value=None)
        runner.ammo_hint = Mock(return_value=False)
        runner.click = Mock()
        with self.assertRaises(Stopped):
            runner.attempt_ammo('ammo_1')
        runner.click.assert_not_called()

    def test_snow_ammo_and_negative_variants(self):
        for key in ('74d2e731-5ecc-4c52-bcb7-dd6d41006565',
                    'a796c331-4742-4aa8-b8e3-3e3452442454'):
            path = Path(tempfile.gettempdir()) / f'codex-clipboard-{key}.png'
            if not path.exists():
                self.skipTest('User capture unavailable')
            frame = np.array(open_capture(path).convert('RGB'))
            self.assertEqual(Runner.preview(None, frame, 'ammo'), 'green')
            no_hint = frame.copy()
            no_hint[835:885,925:1115] = 0
            self.assertEqual(Runner.preview(None, no_hint, 'ammo'), 'unknown')
            neutral = frame.copy()
            crop = neutral[259:356,912:1008]
            crop[:] = crop.mean(axis=2, keepdims=True).astype(np.uint8)
            self.assertEqual(Runner.preview(None, neutral, 'ammo'), 'unknown')
            red = frame.copy()
            red[259:356,912:1008] = red[259:356,912:1008][:,:,[1,0,2]]
            self.assertEqual(Runner.preview(None, red, 'ammo'), 'red')

    def test_live_green_launcher_with_red_lights_and_objective_markers(self):
        path = (Path(tempfile.gettempdir()) / 'codex-clipboard-0323148a-5fe5-4349-9773-068e3ac9ddb2.png')
        if not path.exists():
            self.skipTest('User capture unavailable')
        frame = np.array(open_capture(path).convert('RGB'))
        self.assertEqual(Runner.preview(None, frame), 'green')

    def test_small_red_lights_do_not_override_outline(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[150:350, 950:953] = (0,255,0)
        frame[390:400, 970:988] = (255,0,0)
        frame[42:58, 930:946] = (255,0,0)
        self.assertEqual(Runner.preview(None, frame), 'green')
        frame[150:350, 950:953] = (255,0,0)
        self.assertEqual(Runner.preview(None, frame), 'red')

    def test_radar_binding_is_letters_c_r(self):
        self.assertEqual(sequence(30)[-1][2], ('c', 'r'))

    def test_radar_screen_captures(self):
        cases = [('runs/radar_reference.png', True),
                 ('runs/20261112-225759/radar_screen_switch.png', False)]
        for file, expected in cases:
            if not Path(file).exists():
                self.skipTest('Local capture not present')
            with self.subTest(file=file):
                self.assertEqual(radar_visible(np.asarray(open_capture(file).convert('RGB'))), expected)

    def test_second_launcher_preserves_pitch(self):
        offsets = list(placement_offsets('launcher_2'))
        self.assertTrue(all(dy == 0 and dx > 0 for dx, dy in offsets[:7]))
        self.assertEqual(list(placement_offsets('ammo_1'))[0], (0, 2400))

    def test_ammo_false_positive_capture(self):
        path = Path('runs/20261112-225216/ammo_1_candidate_19.png')
        if not path.exists():
            self.skipTest('Local capture not present')
        frame = np.asarray(open_capture(path).convert('RGB'))
        self.assertNotEqual(Runner.preview(None, frame, 'ammo'), 'green')

    def test_real_launcher_green_still_recognized(self):
        path = Path('runs/20261112-224846/launcher_1_candidate_4.png')
        if not path.exists():
            self.skipTest('Local capture not present')
        self.assertEqual(Runner.preview(None, np.asarray(open_capture(path).convert('RGB'))), 'green')

    def test_left_click_holds_and_releases(self):
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(click_hold=.2)
        events = []
        runner.check = Mock()
        runner.log = Mock()
        runner.mouse_button = lambda flag: events.append(('button', flag))
        runner.wait = lambda seconds: events.append(('wait', seconds))
        runner.click()
        self.assertEqual(events, [('button', 2), ('wait', .2), ('button', 4)])
        self.assertFalse(runner.sending)

    def test_interrupted_click_still_releases(self):
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(click_hold=.2)
        runner.check = Mock()
        runner.mouse_button = Mock()
        runner.wait = Mock(side_effect=Stopped('F10'))
        with self.assertRaises(Stopped):
            runner.click()
        self.assertEqual([call.args[0] for call in runner.mouse_button.call_args_list], [2, 4])
        self.assertFalse(runner.sending)

    def test_sweep_is_wide_and_bounded(self):
        offsets = list(search_offsets())
        self.assertEqual(len(offsets), 33)
        x = y = 0
        positions = []
        for dx, dy in offsets:
            x += dx
            y += dy
            positions.append((x, y))
        self.assertEqual(max(p[0] for p in positions), 5400)
        self.assertEqual(min(p[0] for p in positions), -5400)
        self.assertEqual(offsets[0], (0, 1200))
        self.assertTrue(all(dy >= 0 for dx, dy in offsets))
        self.assertEqual({p[1] for p in positions}, {1200, 1800, 2400})

    def test_latest_blocked_previews_are_red(self):
        paths = list(Path('runs/20261112-224028').glob('launcher_1_candidate_*.png'))
        if not paths:
            self.skipTest('Local captures not present')
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(Runner.preview(None, np.asarray(open_capture(path).convert('RGB'))), 'red')

    def test_sequence_and_cooldown(self):
        steps = sequence(15)
        self.assertEqual([s[1] for s in steps if s[0] == "place"],
                         ["launcher_1", "launcher_2", "ammo_1", "ammo_2"])
        self.assertEqual(steps[2], ("wait", 15))
        self.assertEqual([s[1] for s in steps if s[0] == "key"],
                         ["launcher_1", "launcher_2", "radar_vehicle", "radar_screen"])

    def test_radar_green_does_not_trigger(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[10:500, 10:500] = (0, 255, 0)
        self.assertEqual(Runner.preview(None, frame), "unknown")

    def test_red_veto_and_green_candidate(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[200:350, 950:953] = (0, 255, 0)
        self.assertEqual(Runner.preview(None, frame), "green")
        frame[200:350, 970:973] = (255, 0, 0)
        self.assertEqual(Runner.preview(None, frame), "red")

    def test_left_hand_marker_is_not_preview(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[190:238, 620:650] = (0, 255, 0)
        self.assertEqual(Runner.preview(None, frame), "unknown")

    def test_actual_failed_run_does_not_trigger(self):
        path = Path('runs/20261112-223726/launcher_1_candidate_0.png')
        if not path.exists():
            self.skipTest('Local failure capture not present')
        frame = np.asarray(open_capture(path).convert('RGB'))
        self.assertEqual(Runner.preview(None, frame), "unknown")


if __name__ == "__main__":
    unittest.main()
