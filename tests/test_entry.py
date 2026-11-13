import tempfile
from capture_helpers import open_capture
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
from types import SimpleNamespace
from match_entry import EntryReader
from deploy_test import Runner, Stopped, BattleEnded


class EntryTests(unittest.TestCase):
    def test_tier_reward_then_research_then_matchmaking(self):
        root = Path(tempfile.gettempdir())
        keys = ['442927e4-5d21-42f7-8f0c-cee7c62ee75a',
                'f48e5d51-a62d-4c2b-a273-9407f6d82122',
                '685d5fde-d01b-4167-9014-d26629cb0712']
        frames = [np.array(open_capture(root / f'codex-clipboard-{key}.png').convert('RGB'))
                  for key in keys]
        reader = EntryReader()
        self.assertTrue(reader.tier_complete(frames[0]))
        self.assertTrue(reader.research(frames[1]))
        for frame in frames[1:]:
            self.assertFalse(reader.tier_complete(frame))
        runner = Runner.__new__(Runner)
        for name in ('check', 'log', 'click', 'hotkey'):
            setattr(runner, name, Mock())
        actions = {'hangar', 'research'}
        with patch('deploy_test.pg.moveTo') as move:
            for index, frame in enumerate(frames):
                self.assertTrue(runner.advance_lobby(reader, frame, actions))
                self.assertTrue(runner.advance_lobby(reader, frame, actions))
                if index < 2:
                    runner.hotkey.assert_not_called()
            self.assertEqual([c.args for c in move.call_args_list],
                             [(1301, 894), (1015, 770)])
            self.assertEqual(runner.click.call_count, 2)
            runner.hotkey.assert_called_once_with(('enter',))

    def test_vehicle_loss_transition_waits_before_selection(self):
        runner = Runner.__new__(Runner)
        runner.check = Mock()
        runner.wait = Mock()
        runner.log = Mock()
        runner.return_to_base = Mock()
        runner.logdir = Path('runs')
        reader = Mock()
        reader.results.return_value = False
        reader.destruction_countdown.return_value = False
        reader.selection.side_effect = [False, False, True, True, True]
        with patch('deploy_test.EntryReader', return_value=reader), \
             patch('deploy_test.ImageGrab.grab', return_value=Image.new('RGB', (1920,1080))), \
             patch('deploy_test.Image.Image.save'), patch('deploy_test.radar_visible', return_value=False):
            self.assertTrue(runner.handle_radar_loss())
        runner.return_to_base.assert_called_once_with(reader)

    def test_matchmaking_error_rearms_hangar_repeatedly(self):
        root = Path(tempfile.gettempdir())
        error = np.array(open_capture(root / 'codex-clipboard-5c883dee-ed60-4d7e-a028-79404bb433e7.png').convert('RGB'))
        hangar = np.array(open_capture(root / 'codex-clipboard-685d5fde-d01b-4167-9014-d26629cb0712.png').convert('RGB'))
        reader = EntryReader()
        self.assertTrue(reader.matchmaking_error(error))
        self.assertFalse(reader.matchmaking_error(hangar))
        runner = Runner.__new__(Runner)
        for name in ('check', 'click', 'wait', 'hotkey', 'log'):
            setattr(runner, name, Mock())
        actions = {'hangar'}
        with patch('deploy_test.pg.moveTo') as move, patch('deploy_test.time.monotonic') as clock:
            for index in range(5):
                clock.return_value = index * 4
                runner.advance_lobby(reader, error, actions)
                self.assertNotIn('hangar', actions)
                self.assertIn('match_retry_reset', actions)
                runner.advance_lobby(reader, error, actions)  # Same instant: no double click.
                runner.advance_lobby(reader, hangar, actions)
            self.assertEqual(move.call_count, 5)
        self.assertEqual(runner.click.call_count, 5)
        self.assertEqual(runner.hotkey.call_count, 5)
        self.assertEqual(runner.match_retry_count, 5)

    def test_optional_auto_start(self):
        for enabled in (True, False):
            runner = Runner.__new__(Runner)
            runner.args = SimpleNamespace(auto_start=enabled, from_hangar=True)
            runner.log = Mock()
            runner.wait = Mock()
            runner.confirm = Mock()
            runner.await_start()
            if enabled:
                runner.wait.assert_called_once_with(5)
                runner.confirm.assert_not_called()
            else:
                runner.confirm.assert_called_once()
                runner.wait.assert_not_called()
    def test_transient_focus_recovers_without_input(self):
        runner = Runner.__new__(Runner)
        runner.target = 123
        runner.user32 = Mock()
        runner.user32.GetForegroundWindow.side_effect = [0, 123]
        runner.down = Mock(return_value=False)
        runner.log = Mock()
        with patch('deploy_test.pg.failSafeCheck'), patch('deploy_test.time.sleep'):
            self.assertTrue(runner.brief_focus_return())
        runner.user32.SendInput.assert_not_called()

    def test_focus_grace_still_honors_f10(self):
        runner = Runner.__new__(Runner)
        runner.down = Mock(return_value=True)
        with self.assertRaises(Stopped):
            runner.brief_focus_return()

    def test_failed_results_restarts_once(self):
        path = (Path(tempfile.gettempdir()) / 'codex-clipboard-6a4541ac-7edf-4763-a329-cf22381013fd.png')
        if not path.exists():
            self.skipTest('User capture unavailable')
        frame = np.array(open_capture(path).convert('RGB'))
        reader = EntryReader()
        self.assertTrue(reader.results(frame))
        self.assertFalse(reader.completed(frame))
        runner = Runner.__new__(Runner)
        runner.hotkey = Mock()
        runner.log = Mock()
        actions = set()
        self.assertTrue(runner.advance_lobby(reader, frame, actions))
        self.assertTrue(runner.advance_lobby(reader, frame, actions))
        runner.hotkey.assert_called_once_with(('enter',))
        runner.monitor_battle = True
        with self.assertRaises(BattleEnded):
            runner.inspect_battle_frame(frame)
        frame[965:1005,880:1040] = 0
        self.assertFalse(reader.results(frame))

    def test_destroyed_launcher_countdown_interrupts_deployment(self):
        root = Path('runs/20261113-015619')
        if not root.exists():
            self.skipTest('Local capture unavailable')
        reader = EntryReader()
        alive = np.array(open_capture(root / 'ammo_2_icon_baseline.png').convert('RGB'))
        dead = np.array(open_capture(root / 'ammo_2_attempt_0_0.png').convert('RGB'))
        self.assertFalse(reader.destruction_countdown(alive))
        self.assertTrue(reader.destruction_countdown(dead))
        dead[825:890,1190:1280] = 0
        self.assertTrue(reader.destruction_countdown(dead))
        runner = Runner.__new__(Runner)
        runner.monitor_battle = True
        runner.battle_reader = reader
        with self.assertRaises(BattleEnded):
            runner.inspect_battle_frame(dead)
        self.assertFalse(runner.monitor_battle)

    def test_completed_research_hangar_flow(self):
        reader = EntryReader()
        root = Path(tempfile.gettempdir())
        keys = ['9ee710d5-d909-4028-8d37-10a63a46fd65',
                '69a77ca9-f95a-4804-a0e5-5b34f6d010d6',
                '685d5fde-d01b-4167-9014-d26629cb0712']
        frames = [np.array(open_capture(root / f'codex-clipboard-{key}.png').convert('RGB')) for key in keys]
        self.assertTrue(reader.results(frames[0]))
        self.assertTrue(reader.completed(frames[0]))
        self.assertFalse(reader.hangar(frames[1]))
        self.assertTrue(reader.research(frames[1]))
        self.assertTrue(reader.hangar(frames[2]))
        self.assertFalse(reader.results(frames[2]))
        runner = Runner.__new__(Runner)
        events = []
        runner.check = Mock()
        runner.log = Mock()
        runner.hotkey = lambda keys: events.append(keys)
        runner.click = lambda: events.append('allocate')
        with patch('deploy_test.pg.moveTo') as move:
            actions = set()
            for frame in frames:
                self.assertTrue(runner.advance_lobby(reader, frame, actions))
                runner.advance_lobby(reader, frame, actions)
            move.assert_called_once_with(1015, 770)
            self.assertEqual(events, [('enter',), 'allocate', ('enter',)])
            self.assertFalse(runner.advance_lobby(reader, np.zeros_like(frames[0]), actions))

    def test_death_during_deployment_with_card(self):
        path = (Path(tempfile.gettempdir()) / 'codex-clipboard-dcbb2dd3-bd1c-453d-940c-fdf999880698.png')
        if not path.exists():
            self.skipTest('User capture unavailable')
        frame = np.array(open_capture(path).convert('RGB'))
        runner = Runner.__new__(Runner)
        runner.monitor_battle = False
        runner.inspect_battle_frame(frame)  # Initial selection must not mean death.
        runner.monitor_battle = True
        with self.assertRaises(BattleEnded):
            runner.inspect_battle_frame(frame)
        self.assertFalse(runner.monitor_battle)

    def test_deployment_death_recovers_but_f10_does_not(self):
        for error in (BattleEnded('death'), Stopped('F10 emergency stop')):
            runner = Runner.__new__(Runner)
            runner.perform_deployment = Mock(side_effect=error)
            runner.recover_battle_end = Mock()
            if isinstance(error, BattleEnded):
                runner.deploy_and_engage()
                runner.recover_battle_end.assert_called_once()
            else:
                with self.assertRaises(Stopped):
                    runner.deploy_and_engage()
                runner.recover_battle_end.assert_not_called()
            self.assertFalse(runner.monitor_battle)

    def test_death_during_spawn_clearance_continues_cycle(self):
        runner = Runner.__new__(Runner)
        runner.check = Mock()
        runner.log = Mock()
        runner.enter_match = Mock(side_effect=BattleEnded('clearance death'))
        runner.deploy_and_engage = Mock()
        runner.recover_battle_end = Mock()
        runner.wait_for_results = Mock(side_effect=Stopped('F10'))
        with self.assertRaises(Stopped):
            runner.run_match_loop()
        runner.recover_battle_end.assert_called_once()
        runner.deploy_and_engage.assert_not_called()
        runner.wait_for_results.assert_called_once()

    def test_card_and_results_references(self):
        reader = EntryReader()
        root = Path(tempfile.gettempdir())
        card = np.array(open_capture(root / 'codex-clipboard-a313232d-c5a4-4779-b96b-69ae67208013.png').convert('RGB'))
        results = np.array(open_capture(root / 'codex-clipboard-56b99769-e4fb-4d7f-987c-66536aefaad4.png').convert('RGB'))
        self.assertTrue(reader.selection(card))
        self.assertTrue(reader.selection_card(card))
        self.assertFalse(reader.results(card))
        self.assertTrue(reader.results(results))
        self.assertFalse(reader.selection(results))

    def test_next_match_requires_results_before_entry(self):
        runner = Runner.__new__(Runner)
        events = []
        runner.check = Mock()
        runner.log = Mock()
        runner.enter_match = lambda: events.append('enter')
        runner.deploy_and_engage = lambda: events.append('deploy')
        def result():
            events.append('results')
            if events.count('results') == 2:
                raise Stopped('F10')
        runner.wait_for_results = result
        with self.assertRaises(Stopped):
            runner.run_match_loop()
        self.assertEqual(events, ['enter', 'deploy', 'results'] * 2)

    def test_return_screens(self):
        reader = EntryReader()
        root = Path(tempfile.gettempdir())
        cases = [('c7069e84-48aa-406a-9fb5-b20a33f463b6', True, False, False),
                 ('b6ce7f9f-3909-4c1e-9dda-2868f576318c', False, True, False),
                 ('0b3ccdb7-3893-4c3a-84ef-23972c31f779', False, False, True)]
        for key, selection, menu, confirmation in cases:
            path = root / f'codex-clipboard-{key}.png'
            if not path.exists():
                self.skipTest('User capture unavailable')
            frame = np.array(open_capture(path).convert('RGB'))
            self.assertEqual(reader.selection(frame), selection)
            self.assertEqual(reader.return_menu(frame), menu)
            self.assertEqual(reader.return_confirmation(frame), confirmation)

    def test_return_clicks_only_verified_screens(self):
        for confirmed in (False, True):
            runner = Runner.__new__(Runner)
            for name in ('hotkey', 'wait', 'snapshot', 'check', 'click', 'log'):
                setattr(runner, name, Mock())
            reader = Mock()
            reader.selection_card.return_value = False
            reader.return_menu.return_value = True
            reader.return_confirmation.return_value = confirmed
            with patch('deploy_test.pg.moveTo'):
                if confirmed:
                    runner.return_to_base(reader)
                else:
                    with self.assertRaises(Stopped):
                        runner.return_to_base(reader)
            runner.hotkey.assert_called_once_with(('esc',))
            self.assertEqual(runner.click.call_count, 2 if confirmed else 1)

    def test_return_without_optional_confirmation(self):
        runner = Runner.__new__(Runner)
        for name in ('hotkey', 'wait', 'snapshot', 'check', 'click', 'log'):
            setattr(runner, name, Mock())
        reader = Mock()
        reader.selection_card.return_value = False
        reader.return_menu.side_effect = [True, False]
        reader.return_confirmation.return_value = False
        reader.selection.return_value = False
        with patch('deploy_test.pg.moveTo'):
            runner.return_to_base(reader)
        self.assertEqual(runner.click.call_count, 1)

    def test_live_countdown_selection(self):
        path = (Path(tempfile.gettempdir()) / 'codex-clipboard-1aa8f7fe-887e-41c2-9ff1-455eb06e0a48.png')
        if not path.exists():
            self.skipTest('User capture unavailable')
        reader = EntryReader()
        frame = np.array(open_capture(path).convert('RGB'))
        self.assertTrue(reader.selection(frame))
        self.assertTrue(reader.ready(frame))
        self.assertFalse(reader.spawned(frame))
        # The countdown and map can change without changing entry readiness.
        frame[195:1020, 720:1545] = 0
        self.assertTrue(reader.ready(frame))

    def test_recorded_entry_states(self):
        reader = EntryReader()
        cases = [('runs/selection_ready.png', True, True, False),
                 ('runs/selection_reference.png', True, False, False),
                 ('runs/20261112-224028/launcher_1_before.png', False, False, True),
                 ('runs/radar_reference.png', False, False, False)]
        for path, selection, ready, spawned in cases:
            if not Path(path).exists():
                self.skipTest('Local recording unavailable')
            with self.subTest(path=path):
                frame = np.array(open_capture(path).convert('RGB'))
                self.assertEqual(reader.selection(frame), selection)
                self.assertEqual(reader.ready(frame), ready)
                self.assertEqual(reader.spawned(frame), spawned)

    def test_blank_screen_not_ready(self):
        reader = EntryReader()
        frame = np.zeros((1080,1920,3), dtype=np.uint8)
        self.assertFalse(reader.ready(frame))
        self.assertFalse(reader.spawned(frame))
