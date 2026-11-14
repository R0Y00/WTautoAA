import tempfile
from capture_helpers import open_capture
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
from radar_targets import TargetReader, TargetMemory, selected_target
from deploy_test import Runner


class TargetTests(unittest.TestCase):
    def test_changed_selection_cancels_pair_then_allows_reacquisition(self):
        runner = Runner.__new__(Runner)
        for name in ('check', 'right_double_click', 'wait', 'log'):
            setattr(runner, name, Mock())
        runner.logdir = Path('runs')
        memory = TargetMemory()
        memory.update([{'id': '15', 'starred': False}], 0)
        reader = Mock()
        reader.read.return_value = [{'id': '24', 'selected': True}]
        with patch('deploy_test.ImageGrab.grab', return_value=Image.new('RGB', (1920,1080))), \
             patch('deploy_test.radar_visible', return_value=True), \
             patch('deploy_test.Image.Image.save') as save:
            runner.fire_selected_pair('15', reader, memory)
        self.assertEqual(runner.right_double_click.call_count, 1)
        self.assertEqual(memory.seen['15']['shots'], 1)
        save.assert_called_once()
        # After the outer loop verifies selection again, finish the pending pair.
        runner.fire_selected_pair('15', reader, memory)
        self.assertEqual(runner.right_double_click.call_count, 2)
        self.assertEqual(memory.seen['15']['shots'], 2)

    def test_unstarred_retries_immediately_but_fresh_star_blocks(self):
        memory = TargetMemory()
        row = {'id': '3', 'starred': False}
        memory.update([row], 0)
        memory.fired('3', 0)
        memory.fired('3', .5)
        row['starred'] = True
        memory.update([row], 1)
        self.assertIsNone(memory.choose([row]))
        row['starred'] = False
        memory.update([row], 2)
        self.assertEqual(memory.choose([row])['id'], '3')
        self.assertEqual(memory.seen['3']['shots'], 0)
        self.assertFalse(memory.eligible({'id': '3', 'starred': True}))
        memory.fired('3', 2)
        self.assertTrue(memory.eligible({'id': '3', 'starred': True}))

    def test_expanded_digit_capture(self):
        reader = TargetReader()
        self.assertEqual({key.split('_')[0] for key in reader.templates}, set('0123456789'))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fixture = Path(__file__).resolve().parents[1] / 'assets' / 'target_list_reference.png'
        frame[137:393, 1351:1885] = np.array(open_capture(fixture).convert('RGB'))
        rows = reader.read(frame)
        self.assertEqual([r['id'] for r in rows], ['9', '12', '21', '18', '0', '24', '14', '13'])
        self.assertEqual(selected_target(rows)['id'], '12')
        self.assertTrue(all(not r['starred'] for r in rows))

    def test_selected_nineteen_capture(self):
        p = (Path(tempfile.gettempdir()) / 'codex-clipboard-dec3088e-f8a2-4931-8a62-5a366b45b51a.png')
        if not p.exists():
            self.skipTest('User reference unavailable')
        frame = np.array(open_capture(p).convert('RGB'))
        reader = TargetReader()
        self.assertEqual([(r['id'], r['starred']) for r in reader.read(frame)], [('19', False)])
        # Add a superscript marker beside the full ID, not between its digits.
        frame[145:149, 1378:1382] = 180
        self.assertEqual([(r['id'], r['starred']) for r in reader.read(frame)], [('19', True)])

    def test_retry_after_thirty_seconds_even_with_star(self):
        memory = TargetMemory()
        rows = [{'id': '19', 'starred': False}]
        memory.update(rows, 0)
        memory.fired('19', 0)
        memory.fired('19', .5)
        for starred in (True,):
            rows[0]['starred'] = starred
            memory.update(rows, 29)
            self.assertIsNone(memory.choose(rows))
        memory.update([], 30)
        self.assertIsNone(memory.choose([]))
        memory.update(rows, 30.5)
        self.assertEqual(memory.choose(rows)['id'], '19')
        memory.fired('19', 30.5)
        memory.fired('19', 31)
        memory.update(rows, 59)
        self.assertIsNone(memory.choose(rows))
        memory.update(rows, 61)
        self.assertEqual(memory.choose(rows)['id'], '19')

    def test_selected_zero_capture(self):
        p = Path('runs/20261113-002247/target_after_select_0023.png')
        if not p.exists():
            self.skipTest('Local capture unavailable')
        rows = TargetReader().read(np.array(open_capture(p).convert('RGB')))
        self.assertEqual(selected_target(rows)['id'], '0')
        self.assertFalse(selected_target(rows)['starred'])

    def test_pair_survives_reordering_and_dropout_without_reselect(self):
        for rows in ([{'id': '0', 'row': 3, 'selected': True, 'starred': True}], []):
            with self.subTest(rows=rows):
                runner = Runner.__new__(Runner)
                runner.check = Mock()
                runner.right_double_click = Mock()
                runner.wait = Mock()
                runner.log = Mock()
                memory = TargetMemory()
                memory.update([{'id': '0', 'row': 0}], 0)
                reader = Mock()
                reader.read.return_value = rows
                with patch('deploy_test.ImageGrab.grab', return_value=Image.new('RGB', (1,1))), patch('deploy_test.radar_visible', return_value=True):
                    runner.fire_selected_pair('0', reader, memory)
                self.assertEqual(runner.right_double_click.call_count, 2)
                runner.wait.assert_called_once_with(.5)
                self.assertEqual(memory.seen['0']['shots'], 2)

    def test_starred_reference_and_selection(self):
        p = (Path(tempfile.gettempdir()) / 'codex-clipboard-31debb59-ee11-41eb-964b-eefedfd602f8.png')
        if not p.exists():
            self.skipTest('User reference unavailable')
        rows = TargetReader().read(np.array(open_capture(p).convert('RGB')))
        self.assertEqual([(r['id'], r['starred']) for r in rows],
                         [('3', True), ('0', False), ('1', False), ('2', True)])
        memory = TargetMemory()
        memory.update(rows, 0)
        self.assertEqual(memory.choose(rows)['id'], '0')
        memory.fired('0')
        rows[1]['starred'] = True
        memory.update(rows, 1)
        self.assertEqual(memory.choose(rows)['id'], '0')
        memory.fired('0')
        self.assertEqual(memory.choose(rows)['id'], '1')
        memory.fired('1')
        memory.fired('1')
        self.assertIsNone(memory.choose(rows))

    def test_fleeting_target_keeps_shot_count_but_cannot_be_selected_absent(self):
        memory = TargetMemory(3)
        row = {'id': '1', 'row': 0}
        memory.update([row], 0)
        memory.fired('1')
        memory.update([], 1)
        self.assertIsNone(memory.choose([]))
        memory.update([{'id': '1', 'row': 3}], 2)
        self.assertEqual(memory.seen['1']['shots'], 1)
        memory.fired('1')
        self.assertIsNone(memory.choose([row]))
        memory.update([], 33)
        self.assertNotIn('1', memory.seen)
        memory.update([row], 34)
        self.assertEqual(memory.choose([row])['id'], '1')

    def test_pair_cooldown_and_same_model_is_independent(self):
        memory = TargetMemory()
        rows = [{'id': '0', 'starred': False, 'model': 'same'},
                {'id': '1', 'starred': False, 'model': 'same'}]
        memory.update(rows, 0)
        memory.fired('0')
        memory.fired('0')
        memory.update(rows, 1)
        self.assertEqual(memory.choose(rows)['id'], '1')
        rows[0]['starred'] = True
        rows[1]['starred'] = True
        memory.update(rows, 2)
        self.assertIsNone(memory.choose(rows))
        rows[0]['starred'] = False
        memory.update(rows, 3)
        self.assertEqual(memory.choose(rows)['id'], '0')
        memory.update(rows, 30)
        self.assertEqual(memory.choose(rows)['id'], '0')

    def test_reference_ids_and_rows(self):
        p = (Path(tempfile.gettempdir()) / 'codex-clipboard-f97690c5-9857-45bf-91d6-b11e335fb7ba.png')
        if not p.exists():
            self.skipTest('User reference unavailable')
        frame = np.array(open_capture(p).convert('RGB'))
        reader = TargetReader()
        self.assertEqual([r['id'] for r in reader.read(frame)], ['3', '0', '1', '2'])
        # Remove first row: following targets must keep their own IDs.
        frame[140:166, 1358:1380] = 0
        self.assertEqual([r['id'] for r in reader.read(frame)], ['0', '1', '2'])

    def test_empty_list_and_wrong_resolution(self):
        reader = TargetReader()
        self.assertEqual(reader.read(np.zeros((1080,1920,3), dtype=np.uint8)), [])
        with self.assertRaises(ValueError):
            reader.read(np.zeros((720,1280,3), dtype=np.uint8))
