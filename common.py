# ============================== IMPORTS ==============================
import pygame
import sys
import math
import random
import pickle
import os
import csv
import numpy as np
from datetime import datetime
from collections import deque, defaultdict

# ============================== CONFIGURATION ==============================
os.environ["SDL_VIDEO_CENTERED"] = "1"
WIDTH, HEIGHT = 1400, 900
FPS = 60
LIDAR_COUNT = 8
LIDAR_RANGE = 300
WHEELBASE = 70

# ============================== COLORS ==============================
WHITE = (255, 255, 255)
GREY = (129, 129, 129)
GREEN = (0, 255, 0)
RED = (255, 50, 50)
CLR_ACCENT = (0, 255, 170)
CLR_OBSTACLE = (220, 40, 60)
CLR_TEXT = (240, 240, 240)
CLR_BG = (45, 48, 52)
CLR_BUTTON = (60, 63, 68)
CLR_BUTTON_HOVER = (80, 83, 88)
PARKING_AREA_LINE_WIDTH = 5

# ============================== SIM STATES ==============================
STATE_MENU = 0
STATE_TRAIN = 1
STATE_DEMO = 2


# ============================== LOGGING SYSTEM ==============================
class DetailedLogger:
    def __init__(self, prefix="training"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f"{prefix}_log_{timestamp}.csv"
        self.reward_history = deque(maxlen=100)
        self.results_history = deque(maxlen=1000)
        self.best_avg_reward = float("-inf")
        try:
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                header = [
                    "Episode",
                    "Steps",
                    "Result",
                    "Dist_Error",
                    "Angle_Error",
                    "Reward",
                ]
                if prefix == "training":
                    header.append("Epsilon")
                csv.writer(f).writerow(header)
        except (IOError, OSError) as e:
            print(f"Warning: Could not create log file: {e}")

    def log_episode(self, ep, steps, result, dist, ang, reward, eps=None):
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                row = [
                    ep,
                    steps,
                    result,
                    round(dist, 2),
                    round(ang, 2),
                    round(reward, 2),
                ]
                if eps is not None:
                    row.append(round(eps, 4))
                csv.writer(f).writerow(row)
        except (IOError, OSError) as e:
            print(f"Warning: Could not write to log: {e}")
        self.reward_history.append(reward)
        self.results_history.append(1 if result == "Success" else 0)

    def should_save_best_model(self):
        if len(self.reward_history) < 20:
            return False
        current_avg = sum(self.reward_history) / len(self.reward_history)
        if self.best_avg_reward == float("-inf") or current_avg > self.best_avg_reward:
            self.best_avg_reward = current_avg
            return True
        return False


class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text, self.callback = text, callback
        self.hovered = False
        self.font = pygame.font.SysFont("Segoe UI", 14, bold=True)

    def update(self, mouse_pos, mouse_pressed):
        self.hovered = self.rect.collidepoint(mouse_pos)
        if self.hovered and mouse_pressed:
            self.callback()

    def draw(self, surface):
        color = CLR_BUTTON_HOVER if self.hovered else CLR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, CLR_ACCENT, self.rect, 2, border_radius=5)
        text_surf = self.font.render(self.text, True, CLR_TEXT)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))


class TitleBar:
    def __init__(self, width):
        self.height, self.width = 35, width
        self.font = pygame.font.SysFont("Segoe UI", 12, bold=True)

    def draw(self, surface):
        pygame.draw.rect(surface, CLR_BG, (0, 0, self.width, self.height))
        pygame.draw.line(
            surface, CLR_ACCENT, (0, self.height), (self.width, self.height), 2
        )
        title = self.font.render("SELF PARKING CAR SIMULATOR", True, CLR_TEXT)
        surface.blit(title, title.get_rect(center=(self.width // 2, 18)))


def generate_parking_area(surface, center_x, center_y, width, height):
    x, y = center_x - width // 2, center_y - height // 2
    entrance = 200
    walls = []
    walls.append((x, y, x + width, y))
    walls.append((x + width, y, x + width, y + height))
    walls.append((x, y + height, x + width, y + height))
    y_top_end = y + (height - entrance) / 2
    y_bot_start = y_top_end + entrance
    walls.append((x, y, x, y_top_end))
    walls.append((x, y_bot_start, x, y + height))
    if surface:
        for wall in walls:
            pygame.draw.line(
                surface,
                WHITE,
                (wall[0], wall[1]),
                (wall[2], wall[3]),
                PARKING_AREA_LINE_WIDTH,
            )
    return x, y, walls


class AutonomousVehicle:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.angle, self.speed, self.steer = 0.0, 0.0, 0.0
        self.width, self.length = 44, 88
        self.max_steer, self.max_speed = 40, 4.0
        self.accel, self.brake = 0.6, 0.5

    def get_corners(self):
        corners = []
        for dx in [-self.length / 2, self.length / 2]:
            for dy in [-self.width / 2, self.width / 2]:
                corners.append(self.pos + pygame.Vector2(dx, dy).rotate(self.angle))
        return corners

    def drive(self, action):
        # Action Map (7 core moves for stability):
        # 0: Idle, 1: Fwd, 2: Fwd-Left, 3: Fwd-Right, 
        # 4: Bwd, 5: Bwd-Left, 6: Bwd-Right
        
        throttle = 0
        steer_target = 0
        
        if action == 1: throttle = self.accel
        elif action == 2: throttle, steer_target = self.accel, -self.max_steer
        elif action == 3: throttle, steer_target = self.accel, self.max_steer
        elif action == 4: throttle = -self.brake
        elif action == 5: throttle, steer_target = -self.brake, -self.max_steer
        elif action == 6: throttle, steer_target = -self.brake, self.max_steer
        
        self.steer += (steer_target - self.steer) * 0.4
        self.speed = (self.speed + throttle) * 0.92
        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))

        if abs(self.speed) < 0.1:
            self.speed = 0.0

        if abs(self.speed) > 0.0:
            if math.isclose(self.steer, 0, abs_tol=1e-5):
                turn_radius = float("inf")
            else:
                turn_radius = WHEELBASE / math.tan(math.radians(self.steer))
            
            ang_vel = self.speed / turn_radius if not math.isinf(turn_radius) else 0
            self.angle = (self.angle + math.degrees(ang_vel)) % 360
            self.pos += (
                pygame.Vector2(
                    math.cos(math.radians(self.angle)),
                    math.sin(math.radians(self.angle)),
                )
                * self.speed
            )


class QBrain:
    def __init__(self, model_path="Q_parking_model.pkl"):
        self.path, self.best_path = model_path, "Q_parking_model_best.pkl"
        self.table = defaultdict(lambda: np.zeros(7))
        self.eps, self.eps_min, self.gamma = (
            1.0,
            0.01,
            0.995,
        )
        self.alpha, self.alpha_min, self.alpha_decay = (
            0.5,
            0.05,
            0.9999,
        )

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.replay_batch_size = 32
        self.replay_frequency = 10
        self.step_counter = 0

    def load(self, path=None, verbose=True):
        load_path = path or self.path
        if os.path.exists(load_path):
            try:
                with open(load_path, "rb") as f:
                    loaded_table = pickle.load(f)
                    self.table.update(loaded_table)
                if verbose:
                    print(f"✓ Model loaded: {load_path} ({len(self.table)} states)")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load model from {load_path}: {e}")

    def get_action(self, state, demo=False):
        if not demo and random.random() < self.eps:
            return random.randint(0, 6)
        return int(np.argmax(self.table[state]))

    def learn(self, s, a, r, sn):
        # Traditional Q-Learning update
        self.table[s][a] += self.alpha * (
            r + self.gamma * np.max(self.table[sn]) - self.table[s][a]
        )
        
        # Experience Replay
        self.replay_buffer.append((s, a, r, sn))
        self.step_counter += 1
        
        if len(self.replay_buffer) >= self.replay_batch_size and self.step_counter % self.replay_frequency == 0:
            batch = random.sample(self.replay_buffer, self.replay_batch_size)
            for bs, ba, br, bsn in batch:
                self.table[bs][ba] += self.alpha * (
                    br + self.gamma * np.max(self.table[bsn]) - self.table[bs][ba]
                )

    def decay_params(self, success_rate):
        self.alpha = max(self.alpha_min, self.alpha * 0.9995) # Slower decay
        # Smarter epsilon decay based on success
        if success_rate > 70:
            decay = 0.99
        elif success_rate > 40:
            decay = 0.995
        elif success_rate > 20:
            decay = 0.998
        else:
            decay = 0.9992
        self.eps = max(self.eps_min, self.eps * decay)

    def save(self, path=None, verbose=True):
        try:
            with open(path or self.path, "wb") as f:
                # Convert back to regular dict for pickling
                pickle.dump(dict(self.table), f)
            if verbose:
                print(f"✓ Model saved: {path or self.path} ({len(self.table)} states)")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save model: {e}")

    def save_best(self, verbose=True):
        self.save(self.best_path, verbose=verbose)


class Simulator:
    def __init__(self, screen, mode=STATE_TRAIN, headless=False):
        self.screen, self.mode, self.headless = screen, mode, headless
        if not headless:
            self.clock = pygame.time.Clock()
            self.f_ui = pygame.font.SysFont("Consolas", 13)
            self.f_msg = pygame.font.SysFont("Segoe UI", 60, bold=True)
            self.parking_surf = self.f_ui.render("PARKING", True, (0, 255, 120))
            self.title_bar = TitleBar(WIDTH)
            self.buttons = []
            self.create_buttons()
        else:
            self.clock = None
            self.f_ui = None
            self.f_msg = None

        self.brain = QBrain()
        self.brain.load(verbose=not headless)
        self.logger = DetailedLogger("training" if mode == STATE_TRAIN else "demo")
        self.state, self.ep_count, self.paused = mode, 0, False
        self.p_area_w, self.p_area_h = 1000, 650
        self.slot_w, self.slot_h = 110, 170
        self.recent_successes = []
        self.base_step_limit = 2000 
        self.msg_text, self.msg_timer, self.msg_color = "", 0, CLR_ACCENT
        self.init_parking_area()
        self.reset_env()

    def create_buttons(self):
        w, h, x, s = 120, 35, WIDTH - 140, 45
        self.buttons = [
            Button(x, 50, w, h, "PAUSE", self.toggle_pause),
            Button(
                x,
                50 + s,
                w,
                h,
                "SAVE",
                lambda: self.brain.save(verbose=not self.headless),
            ),
            Button(x, 50 + s * 2, w, h, "EXIT", self.exit_simulation),
            Button(
                x,
                50 + s * 3,
                w,
                h,
                "SAVE BEST",
                lambda: self.brain.save_best(verbose=not self.headless),
            ),
        ]

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.headless:
            self.buttons[0].text = "RESUME" if self.paused else "PAUSE"

    def exit_simulation(self):
        if self.mode == STATE_TRAIN:
            self.brain.save(verbose=not self.headless)
        pygame.quit()
        sys.exit()

    def init_parking_area(self):
        tb_h = 35 if self.headless else self.title_bar.height
        cx, cy = WIDTH / 2, (HEIGHT + tb_h) / 2 + 20
        self.p_area_x, self.p_area_y, self.walls = generate_parking_area(
            None, cx, cy, self.p_area_w, self.p_area_h
        )
        entrance = 200
        self.entrance_top = self.p_area_y + (self.p_area_h - entrance) / 2
        self.entrance_bottom = self.entrance_top + entrance
        self.slots = []
        max_slots = (self.p_area_w + 15) // (self.slot_w + 15)
        gap = (self.p_area_w - max_slots * self.slot_w) // (max_slots + 1)
        for y_mult in [0, 1]:
            y = self.p_area_y + (self.p_area_h - self.slot_h - gap) * y_mult + gap
            for i in range(max_slots):
                self.slots.append(
                    pygame.Rect(
                        self.p_area_x + gap + i * (self.slot_w + gap),
                        y,
                        self.slot_w,
                        self.slot_h,
                    )
                )

    def reset_env(self, outcome=None):
        if outcome:
            dist_err = self.car.pos.distance_to(self.target.center)
            ang_err = self.calculate_angle_error()
            self.logger.log_episode(
                self.ep_count,
                self.steps,
                outcome,
                dist_err,
                ang_err,
                self.ep_reward,
                self.brain.eps if self.state == STATE_TRAIN else None,
            )

            self.recent_successes.append(1 if outcome == "Success" else 0)
            if len(self.recent_successes) > 100:
                self.recent_successes.pop(0)

            if self.state == STATE_TRAIN:
                rate = sum(self.recent_successes) / len(self.recent_successes) * 100
                self.brain.decay_params(rate)
                if self.logger.should_save_best_model():
                    self.brain.save_best(verbose=not self.headless)

            self.msg_text, self.msg_timer = outcome.upper(), 60
            self.msg_color = GREEN if outcome == "Success" else RED

        # Curriculum learning: progressive difficulty based on performance
        success_rate = (
            sum(self.recent_successes) / len(self.recent_successes) * 100
            if self.recent_successes
            else 0
        )

        if self.mode == STATE_DEMO:
            obstacle_prob = 0.2
            start_dist_range = (50, 300)
            angle_variation = 15
        elif success_rate < 50:  # Bootstrap
            obstacle_prob = 0.0
            start_dist_range = (30, 50)
            angle_variation = 0
        elif success_rate < 80:  # Intermediate
            obstacle_prob = 0.05
            start_dist_range = (40, 150)
            angle_variation = 10
        elif success_rate < 95:  # Advanced
            obstacle_prob = 0.1
            start_dist_range = (50, 250)
            angle_variation = 20
        else:  # Mastery
            obstacle_prob = 0.25
            start_dist_range = (50, 400)
            angle_variation = 45

        self.target = random.choice(self.slots)
        self.obstacles = [
            s
            for s in self.slots
            if s != self.target and random.random() < obstacle_prob
        ]

        # Progressive starting positions
        start_dist = random.randint(*start_dist_range)
        start_x = self.p_area_x + start_dist
        start_y = (self.entrance_top + self.entrance_bottom) / 2 + random.randint(
            -100, 100
        )

        # Progressive starting angles
        base_angles = [0, 90, 180, 270]
        self.car = AutonomousVehicle(start_x, start_y)
        self.car.angle = random.choice(base_angles) + random.randint(
            -angle_variation, angle_variation
        )
        self.prev_dist = self.car.pos.distance_to(self.target.center)
        self.prev_ang_err = self.calculate_angle_error()
        self.prev_speed = 0.0
        self.steps, self.ep_reward, self.collision_counter, self.ep_count = (
            0,
            0,
            0,
            self.ep_count + 1,
        )

    def calculate_angle_error(self):
        ang = self.car.angle % 360
        # Optimal vertical parking angles are 90 or 270
        return min(
            abs(ang - 90), 360 - abs(ang - 90), abs(ang - 270), 360 - abs(ang - 270)
        )

    def get_observation(self):
        # Target Relative Position (Enhanced Cartesian Bins)
        dx = self.target.centerx - self.car.pos.x
        dy = self.target.centery - self.car.pos.y
        
        # X Bins: 7 zones for smoother approach
        if dx < -250: xb = 0
        elif dx < -100: xb = 1
        elif dx < -30: xb = 2
        elif dx < 30: xb = 3
        elif dx < 100: xb = 4
        elif dx < 250: xb = 5
        else: xb = 6
        
        # Y Bins: 7 zones
        if dy < -250: yb = 0
        elif dy < -100: yb = 1
        elif dy < -30: yb = 2
        elif dy < 30: yb = 3
        elif dy < 100: yb = 4
        elif dy < 250: yb = 5
        else: yb = 6

        # Angle Bins: 12 directions (30 deg each)
        ang = self.car.angle % 360
        ab = int(ang // 30)

        # Alignment Error: 2 bins (Aligned < 10, Not)
        ang_err = self.calculate_angle_error()
        aeb = 0 if ang_err < 10 else 1

        # Speed Bins: 3 states
        if self.car.speed < -0.2: vb = 0
        elif abs(self.car.speed) < 0.2: vb = 1
        else: vb = 2

        # Surroundings: 4-ray focus (Front, Rear, Left, Right)
        obs = []
        for i in [0, 4, 2, 6]:
            angle = math.radians(self.car.angle + (i * 360 / LIDAR_COUNT))
            p = self.car.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * 65
            hit = 1 if not pygame.Rect(self.p_area_x, self.p_area_y, self.p_area_w, self.p_area_h).collidepoint(p) or \
                      any(o.collidepoint(p) for o in self.obstacles) else 0
            obs.append(hit)

        # Total States: 7 * 7 * 12 * 2 * 3 * 2^4 = 18,816 states
        return (xb, yb, ab, aeb, vb, tuple(obs)), None

    def calculate_reward(self, dist, collided, parked, timeout):
        reward = 0

        # Potential-based Distance Reward (Closing distance is key)
        reward += (self.prev_dist - dist) * 250.0

        # Alignment Reward near target
        ang_err = self.calculate_angle_error()
        if dist < 200:
            reward += (self.prev_ang_err - ang_err) * 150.0
        
        # Major outcomes
        if collided:
            reward -= 2000000 
        elif parked:
            reward += 20000000 
        elif timeout:
            reward -= 500000

        # Small Living penalty
        reward -= 15.0

        return reward, ang_err

    def step(self):
        s, _ = self.get_observation()
        a = self.brain.get_action(s, self.mode == STATE_DEMO)
        self.car.drive(a)
        dist = self.car.pos.distance_to(self.target.center)
        car_corners = self.car.get_corners()
        collided = any(
            obs.collidepoint(c) for obs in self.obstacles for c in car_corners
        ) or any(
            not pygame.Rect(
                self.p_area_x, self.p_area_y, self.p_area_w, self.p_area_h
            ).collidepoint(c)
            for c in car_corners
        )

        parked = (
            self.target.collidepoint(self.car.pos)
            and abs(self.car.speed) < 0.2
            and self.calculate_angle_error() < 15
        )

        timeout = self.steps > self.base_step_limit

        if self.mode == STATE_TRAIN:
            reward, new_ang_err = self.calculate_reward(dist, collided, parked, timeout)
            self.ep_reward += reward
            sn, _ = self.get_observation()
            self.brain.learn(s, a, reward, sn)
            self.prev_ang_err = new_ang_err

        self.prev_dist, self.prev_speed, self.steps = dist, self.car.speed, self.steps + 1
        if collided:
            self.collision_counter += 1
        else:
            self.collision_counter = 0

        result = (
            "Collision"
            if self.collision_counter > 2
            else (
                "Success"
                if parked
                else ("Timeout" if timeout else None)
            )
        )
        if result:
            self.reset_env(result)
        return result

    def run(self):
        if self.headless:
            return self._run_headless()
        self._run_visual()

    def _run_visual(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.exit_simulation()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    self.toggle_pause()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            for btn in self.buttons:
                btn.update(pygame.mouse.get_pos(), mouse_pressed)
            if not self.paused:
                self.step()
            self.draw_world()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _run_headless(self):
        while True:
            res = self.step()
            if res:
                yield res

    def draw_world(self):
        self.screen.fill(GREY)
        # Draw pre-calculated walls
        for wall in self.walls:
            pygame.draw.line(
                self.screen,
                WHITE,
                (wall[0], wall[1]),
                (wall[2], wall[3]),
                PARKING_AREA_LINE_WIDTH,
            )
        
        for slot in self.slots:
            is_target = slot == self.target
            color = (
                (0, 255, 120)
                if is_target
                else (CLR_OBSTACLE if slot in self.obstacles else WHITE)
            )
            pygame.draw.rect(
                self.screen,
                color,
                slot,
                5 if is_target else (0 if slot in self.obstacles else 3),
                border_radius=5,
            )
            if is_target:
                self.screen.blit(
                    self.parking_surf, self.parking_surf.get_rect(center=slot.center)
                )

        # Draw LIDAR rays
        for i in range(LIDAR_COUNT):
            angle = math.radians(self.car.angle + (i * 360 / LIDAR_COUNT))
            ray_dir = pygame.Vector2(math.cos(angle), math.sin(angle))
            closest_dist = LIDAR_RANGE
            for d in range(20, LIDAR_RANGE, 10):
                p = self.car.pos + ray_dir * d
                if not (0 <= p.x <= WIDTH and 0 <= p.y <= HEIGHT) or \
                   not pygame.Rect(self.p_area_x, self.p_area_y, self.p_area_w, self.p_area_h).collidepoint(p) or \
                   any(obs.collidepoint(p) for obs in self.obstacles):
                    closest_dist = d
                    break
            
            # Color based on distance
            color = RED if closest_dist < 60 else (CLR_ACCENT if closest_dist < 120 else (100, 100, 100))
            pygame.draw.line(self.screen, color, self.car.pos, self.car.pos + ray_dir * closest_dist, 1)

        # Car body
        c_s = pygame.Surface((self.car.length, self.car.width), pygame.SRCALPHA)
        pygame.draw.rect(
            c_s, CLR_ACCENT, (0, 0, self.car.length, self.car.width), border_radius=8
        )
        # Headlights (Front)
        pygame.draw.circle(c_s, (255, 255, 200), (self.car.length - 10, 10), 5)
        pygame.draw.circle(
            c_s, (255, 255, 200), (self.car.length - 10, self.car.width - 10), 5
        )
        # Taillights (Rear)
        pygame.draw.rect(c_s, (200, 0, 0), (5, 5, 5, 10))
        pygame.draw.rect(c_s, (200, 0, 0), (5, self.car.width - 15, 5, 10))

        rot_car = pygame.transform.rotate(c_s, -self.car.angle)
        self.screen.blit(rot_car, rot_car.get_rect(center=self.car.pos))

        self.title_bar.draw(self.screen)
        self.draw_hud()
        if self.msg_timer > 0:
            msg = self.f_msg.render(self.msg_text, True, self.msg_color)
            pygame.draw.rect(
                self.screen,
                CLR_BG,
                msg.get_rect(center=(WIDTH / 2, HEIGHT / 2)).inflate(40, 20),
                border_radius=10,
            )
            self.screen.blit(msg, msg.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
            self.msg_timer -= 1
        for btn in self.buttons:
            btn.draw(self.screen)

    def draw_hud(self):
        rate = (
            sum(self.recent_successes) / len(self.recent_successes) * 100
            if self.recent_successes
            else 0
        )
        avg_rew = (
            sum(self.logger.reward_history) / len(self.logger.reward_history)
            if self.logger.reward_history
            else 0
        )
        stats = [
            f"Episode: {self.ep_count}",
            f"Success Rate: {rate:.1f}%",
            f"Avg Reward: {avg_rew:.2f}",
            f"Epsilon: {self.brain.eps:.4f}",
            f"Q-Table Size: {len(self.brain.table)}",
            f"Steps: {self.steps}",
            f"Velocity: {self.car.speed:.2f}",
            f"Dist to Target: {int(self.car.pos.distance_to(self.target.center))}",
            f"Mode: {'DEMO' if self.mode == STATE_DEMO else 'TRAINING'}",
        ]
        for i, text in enumerate(stats):
            self.screen.blit(
                self.f_ui.render(text, True, CLR_ACCENT), (20, 50 + i * 20)
            )
