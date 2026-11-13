"""Entry UI checks calibrated from the user's Chinese 1920x1080 recording."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


class EntryReader:
    def __init__(self):
        root = Path(__file__).parent / 'assets' / 'entry'
        self.templates = {name: cv2.cvtColor(np.array(Image.open(root / (name+'.png')).convert('RGB')),
                                             cv2.COLOR_RGB2GRAY)
                          for name in ('spawn_label', 'radar_mode_label', 'iris_label', 'join_button', 'deploy_key',
                                       'return_base', 'menu_controls', 'return_question', 'return_yes',
                                       'card_hint', 'radar_scale', 'results_tab', 'results_base', 'results_join',
                                       'completed_title', 'completed_ok', 'research_title', 'research_allocate',
                                       'hangar_join', 'hangar_tree', 'launch_system_destroyed', 'evacuation_countdown',
                                       'missing_vehicle', 'missing_vehicle_ok',
                                       'tier_complete_title', 'tier_complete_ok',
                                       'destroyed_banner', 'death_score')}

    def destruction_replay(self, frame):
        # Static labels only: exclude attacker identity, vehicle and score.
        return (self.matches(frame, 'destroyed_banner', (720,30,970,80), .90) and
                self.matches(frame, 'death_score', (890,975,1030,1030)))

    def tier_complete(self, frame):
        return (self.matches(frame, 'tier_complete_title', (550,168,890,208)) and
                self.matches(frame, 'tier_complete_ok', (1265,875,1335,912)))

    def matchmaking_error(self, frame):
        return (self.matches(frame, 'missing_vehicle', (815,475,1105,545)) and
                self.matches(frame, 'missing_vehicle_ok', (1060,560,1125,602)))

    def destruction_countdown(self, frame):
        # Match static words, excluding the changing countdown number.
        return (self.matches(frame, 'launch_system_destroyed', (620,760,965,910)) and
                self.matches(frame, 'evacuation_countdown', (940,760,1190,910)))

    def completed(self, frame):
        # Outcome titles vary (completed/failed); the summary tab and action
        # button identify the page. Final results shift the tab down by 23px.
        return (self.matches(frame, 'results_tab', (390,73,490,135)) and
                self.matches(frame, 'completed_ok', (925,965,995,1005)))

    def research(self, frame):
        return (self.matches(frame, 'research_title', (515,210,665,250)) and
                self.matches(frame, 'research_allocate', (950,750,1080,786)))

    def hangar(self, frame):
        return (self.matches(frame, 'hangar_join', (865,15,1055,62)) and
                self.matches(frame, 'hangar_tree', (230,882,310,918)))

    def results(self, frame):
        return self.completed(frame) or (self.matches(frame, 'results_tab', (390,73,490,135)) and
                self.matches(frame, 'results_base', (1385,915,1518,955)) and
                self.matches(frame, 'results_join', (880,965,1040,1005)))

    def selection_card(self, frame):
        return (self.matches(frame, 'card_hint', (190,168,390,197)) and
                self.matches(frame, 'radar_scale', (507,375,600,405)) and
                self.matches(frame, 'iris_label', (215,108,375,145)))

    def return_confirmation(self, frame):
        return (self.matches(frame, 'return_question', (850,518,1070,560)) and
                self.matches(frame, 'return_yes', (875,605,915,642)))

    def return_menu(self, frame):
        return (self.matches(frame, 'return_base', (900,635,1020,680)) and
                self.matches(frame, 'menu_controls', (900,520,1020,565)))

    def matches(self, frame, name, rect, threshold=.86):
        if frame.shape[:2] != (1080, 1920):
            raise ValueError('Match entry requires calibrated 1920x1080 UI')
        x0,y0,x1,y1 = rect
        crop = cv2.cvtColor(frame[y0:y1,x0:x1], cv2.COLOR_RGB2GRAY)
        score = cv2.matchTemplate(crop, self.templates[name], cv2.TM_CCOEFF_NORMED).max()
        return bool(score >= threshold)

    def selection(self, frame):
        # These configuration labels are independent of the map, spawn cost,
        # and the footer's auto-start countdown. Do not match dynamic footer text.
        return ((self.matches(frame, 'spawn_label', (270,325,345,360)) and
                 self.matches(frame, 'radar_mode_label', (490,325,610,360))) or
                self.selection_card(frame))

    def ready(self, frame):
        return (self.selection(frame) and
                self.matches(frame, 'iris_label', (215,108,375,145)) and
                self.matches(frame, 'join_button', (215,60,370,95)))

    def spawned(self, frame):
        if self.selection(frame):
            return False
        hsv = cv2.cvtColor(frame[50:410,45:400], cv2.COLOR_RGB2HSV)
        green = cv2.inRange(hsv, (45,150,100), (90,255,255))
        return (np.count_nonzero(green) > 1500 and
                self.matches(frame, 'deploy_key', (650,965,1250,1008)))
