import pygame
import time
import random
import math


class DrivingSimulator:
    def __init__(self):
        pygame.init()
        self.width, self.height = 400, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Safety Auto-Brake Monitor Simulation")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("arial", 26, bold=True)
        self.font_med   = pygame.font.SysFont("arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("arial", 16)

        # Physics
        self.velocity       = 5.0
        self.lane_y_offset  = 0.0
        self.tree_y_offset  = 0.0

        # Car geometry
        self.car_w = 60
        self.car_h = 110
        self.car_x = (self.width - self.car_w) // 2
        self.car_y = self.height - 150

        # State
        self.unconscious_timer_start = None
        self.ignition_lock_start     = None
        self.unc_stage               = 0
        self.particles               = []
        self.hazard_blink            = False
        self.last_blink_time         = time.time()
        self._quit_requested         = False

        # Parallax tree positions (fixed x, scroll y)
        self.trees_left  = [{"x": 30,  "y": random.randint(0, 600)} for _ in range(6)]
        self.trees_right = [{"x": 355, "y": random.randint(0, 600)} for _ in range(6)]

        # Streak particles for speed blur
        self.streaks = []

    # ── Quit handling ─────────────────────────────────────────────────────────
    def should_quit(self):
        """Consume the full pygame event queue each frame."""
        self.pressed_key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN:
                k = getattr(event, 'unicode', '').lower()
                
                if event.key == pygame.K_q or k == 'q': self._quit_requested = True
                elif event.key in (pygame.K_1, pygame.K_KP1) or k == '1': self.pressed_key = '1'
                elif event.key in (pygame.K_2, pygame.K_KP2) or k == '2': self.pressed_key = '2'
                elif event.key in (pygame.K_0, pygame.K_KP0) or k == '0': self.pressed_key = '0'
                elif event.key == pygame.K_l or k == 'l': self.pressed_key = 'l'
                elif event.key == pygame.K_k or k == 'k': self.pressed_key = 'k'
                elif event.key == pygame.K_d or k == 'd': self.pressed_key = 'd'
        return self._quit_requested

    # ── Physics update ────────────────────────────────────────────────────────
    def update(self, driver_state, drowsy_count):
        current_time  = time.time()
        should_hazard = False

        # Hazard blink toggle
        if current_time - self.last_blink_time > 0.5:
            self.hazard_blink    = not self.hazard_blink
            self.last_blink_time = current_time

        # Ignition lock check
        ignition_locked = False
        if self.ignition_lock_start is not None:
            if current_time - self.ignition_lock_start < 30.0:
                ignition_locked = True
                self.velocity   = 0.0
            else:
                self.ignition_lock_start = None

        if ignition_locked:
            return True

        # State-based velocity control
        if driver_state in ("ALERT", "DISTRACTED"):
            self.velocity                = min(5.0, self.velocity + 0.05)
            self.unconscious_timer_start = None
            self.ignition_lock_start     = None
            self.unc_stage               = 0

        elif driver_state == "DROWSY":
            self.velocity                = 3.0
            should_hazard                = True
            self.unconscious_timer_start = None

        elif driver_state == "UNCONSCIOUS":
            should_hazard = True
            if self.unconscious_timer_start is None:
                self.unconscious_timer_start = current_time
                self.unc_stage               = 1

            elapsed = current_time - self.unconscious_timer_start
            if elapsed < 15.0:
                self.unc_stage = 1
                self.velocity  = max(2.5, self.velocity - 0.01)
            elif elapsed < 25.0:
                self.unc_stage = 2
                self.velocity  = max(1.0, self.velocity - 0.03)
            elif elapsed < 30.0:
                self.unc_stage = 3
                self.velocity  = max(0.2, self.velocity - 0.10)
            else:
                self.unc_stage = 4

        elif driver_state == "EMERGENCY":
            should_hazard = True
            self.velocity  = 0.0
            if self.ignition_lock_start is None:
                self.ignition_lock_start     = current_time
            self.unconscious_timer_start = None

        # Smoke particles when braking hard
        if (driver_state in ("UNCONSCIOUS", "EMERGENCY") and
                self.velocity < 4.8 and random.random() > 0.4):
            self.particles.append({
                "x":     self.car_x + self.car_w // 2 + random.randint(-15, 15),
                "y":     self.car_y + self.car_h,
                "r":     5,
                "alpha": 200
            })

        # Speed streaks
        if self.velocity > 3.5 and random.random() > 0.6:
            self.streaks.append({
                "x": random.randint(self.car_x - 30, self.car_x + self.car_w + 30),
                "y": random.randint(self.car_y - 80, self.car_y),
                "life": 6
            })

        # Scroll offsets
        self.lane_y_offset += self.velocity
        if self.lane_y_offset > 100:
            self.lane_y_offset = 0

        self.tree_y_offset += self.velocity * 1.5   # parallax faster than road
        if self.tree_y_offset > 100:
            self.tree_y_offset = 0

        return should_hazard

    # ── Renderer ──────────────────────────────────────────────────────────────
    def render(self, driver_state, should_hazard, drowsy_count):
        # Road background
        self.screen.fill((35, 35, 35))

        # Road surface
        pygame.draw.rect(self.screen, (55, 55, 55),
                         (80, 0, self.width - 160, self.height))

        # ── Parallax trees ──
        for tree in self.trees_left + self.trees_right:
            ty = (tree["y"] + self.tree_y_offset) % 650 - 50
            # Trunk
            pygame.draw.rect(self.screen, (100, 70, 40),
                             (tree["x"] - 4, int(ty) + 30, 8, 30))
            # Canopy (3 circles for a fuller look)
            pygame.draw.circle(self.screen, (34, 100, 34),
                               (tree["x"], int(ty) + 20), 22)
            pygame.draw.circle(self.screen, (28, 85, 28),
                               (tree["x"] - 10, int(ty) + 30), 16)
            pygame.draw.circle(self.screen, (28, 85, 28),
                               (tree["x"] + 10, int(ty) + 30), 16)

        # ── Lane markings ──
        for y in range(-100, self.height, 100):
            # Centre dashes
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (self.width // 2 - 5,
                              y + int(self.lane_y_offset), 10, 50))
            # Edge lines
            pygame.draw.rect(self.screen, (200, 200, 20),
                             (82, y + int(self.lane_y_offset), 5, 80))
            pygame.draw.rect(self.screen, (200, 200, 20),
                             (self.width - 87, y + int(self.lane_y_offset), 5, 80))

        # ── Speed streaks ──
        for s in self.streaks[:]:
            alpha = int(s["life"] / 6 * 180)
            surf  = pygame.Surface((18, 2), pygame.SRCALPHA)
            surf.fill((255, 255, 255, alpha))
            self.screen.blit(surf, (s["x"], s["y"]))
            s["life"] -= 1
            if s["life"] <= 0:
                self.streaks.remove(s)

        # ── Smoke particles ──
        for p in self.particles[:]:
            s = pygame.Surface((int(p["r"] * 2), int(p["r"] * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, (150, 150, 150, max(0, int(p["alpha"]))),
                               (int(p["r"]), int(p["r"])), int(p["r"]))
            self.screen.blit(s, (p["x"] - int(p["r"]), p["y"] - int(p["r"])))
            p["r"]     += 0.8
            p["alpha"] -= 4
            p["y"]     += self.velocity
            if p["alpha"] <= 0:
                self.particles.remove(p)

        # ── Car ──
        car_color = (0, 150, 255)
        if should_hazard and self.hazard_blink:
            car_color = (255, 200, 0)
        pygame.draw.rect(self.screen, car_color,
                         (self.car_x, self.car_y, self.car_w, self.car_h),
                         border_radius=8)
        # Windscreen
        pygame.draw.rect(self.screen, (10, 10, 10),
                         (self.car_x + 8, self.car_y + 20,
                          self.car_w - 16, 25), border_radius=4)
        # Headlights
        pygame.draw.rect(self.screen, (255, 255, 180),
                         (self.car_x + 5, self.car_y + 5, 12, 8),
                         border_radius=3)
        pygame.draw.rect(self.screen, (255, 255, 180),
                         (self.car_x + self.car_w - 17, self.car_y + 5, 12, 8),
                         border_radius=3)

        # ── Speedometer arc (bottom right) ──
        self._draw_speedometer(driver_state)

        # ── HUD header panel ──
        state_color = (0, 255, 0)
        if driver_state in ("DROWSY", "DISTRACTED"):
            state_color = (255, 200, 0)
        elif driver_state == "UNCONSCIOUS":
            state_color = (255, 50, 50)
        elif driver_state == "EMERGENCY":
            state_color = (255, 0, 255)

        pygame.draw.rect(self.screen, (15, 15, 15),
                         (0, 0, self.width, 110))
        pygame.draw.line(self.screen, (100, 100, 100),
                         (0, 110), (self.width, 110), 2)

        kmh      = int(self.velocity * 12)
        state_s  = self.font_large.render(f"STATE: {driver_state}", True, state_color)
        spd_s    = self.font_small.render(f"Speed: {kmh} km/h", True, (200, 200, 200))
        d_col    = (255, 200, 0) if drowsy_count >= 3 else (200, 200, 200)
        dsw_s    = self.font_small.render(f"Drowsy Events: {drowsy_count}/3",
                                          True, d_col)

        self.screen.blit(state_s, (15, 12))
        self.screen.blit(spd_s,   (15, 52))
        self.screen.blit(dsw_s,   (15, 78))

        # Ignition status
        if self.ignition_lock_start is not None:
            ig_rem = max(0.0, 30.0 - (time.time() - self.ignition_lock_start))
            ig_s   = self.font_small.render(f"IGNITION: LOCKED ({ig_rem:.1f}s)",
                                            True, (255, 50, 50))
            self.screen.blit(ig_s, (self.width - 230, 78))
        else:
            ig_s = self.font_small.render("IGNITION: ACTIVE", True, (50, 255, 50))
            self.screen.blit(ig_s, (self.width - 185, 78))

        if (driver_state == "UNCONSCIOUS" and
                self.unconscious_timer_start is not None):
            elapsed  = time.time() - self.unconscious_timer_start
            timer_s  = self.font_small.render(
                f"Unresponsive: {elapsed:.1f}s / 30.0s", True, (255, 50, 50))
            self.screen.blit(timer_s, (self.width - 245, 52))

        # ── Emergency / unconscious overlays ──
        if self.ignition_lock_start is not None:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 80))
            pygame.draw.rect(overlay, (0, 0, 0, 210),
                             (0, self.height // 2 - 65, self.width, 130))
            self.screen.blit(overlay, (0, 0))

            w1 = self.font_large.render("DRIVER UNRESPONSIVE", True, (255, 50, 50))
            w2 = self.font_med.render("IGNITION LOCKED", True, (255, 50, 50))
            w3 = self.font_small.render("VEHICLE SECURED — HELP EN ROUTE",
                                        True, (255, 255, 255))
            self.screen.blit(w1, (self.width // 2 - w1.get_width() // 2,
                                  self.height // 2 - 45))
            self.screen.blit(w2, (self.width // 2 - w2.get_width() // 2,
                                  self.height // 2 - 10))
            self.screen.blit(w3, (self.width // 2 - w3.get_width() // 2,
                                  self.height // 2 + 22))

        elif driver_state == "UNCONSCIOUS" and self.unc_stage > 0:
            stage_s = self.font_large.render(
                f"[ EMERGENCY BRAKING: STAGE {self.unc_stage} ]",
                True, (255, 255, 0))
            self.screen.blit(stage_s,
                             (self.width // 2 - stage_s.get_width() // 2,
                              self.height // 2 - 40))

        pygame.display.flip()
        self.clock.tick(60)

    # ── Speedometer helper ────────────────────────────────────────────────────
    def _draw_speedometer(self, driver_state):
        cx, cy, r = self.width - 48, self.height - 48, 36
        # Background arc
        pygame.draw.circle(self.screen, (30, 30, 30), (cx, cy), r)
        pygame.draw.circle(self.screen, (80, 80, 80), (cx, cy), r, 2)

        # Speed arc — 210° sweep, start at 210° (bottom-left)
        max_vel   = 6.0
        fraction  = min(self.velocity / max_vel, 1.0)
        sweep_deg = 210
        start_a   = 210   # degrees from 3-o'clock, pygame uses anti-clockwise
        end_a     = start_a - int(fraction * sweep_deg)

        arc_col = ((0, 220, 0)   if fraction < 0.5
                   else (255, 200, 0) if fraction < 0.8
                   else (255, 50, 50))

        # Draw arc as polyline of small segments
        steps = max(2, int(fraction * sweep_deg))
        pts   = []
        for i in range(steps + 1):
            a_deg = start_a - i * (fraction * sweep_deg / max(steps, 1))
            a_rad = math.radians(a_deg)
            pts.append((
                cx + int((r - 4) * math.cos(a_rad)),
                cy - int((r - 4) * math.sin(a_rad))
            ))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, arc_col, False, pts, 4)

        # Needle
        needle_a = math.radians(start_a - fraction * sweep_deg)
        nx = cx + int((r - 8) * math.cos(needle_a))
        ny = cy - int((r - 8) * math.sin(needle_a))
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (nx, ny), 2)
        pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), 4)

        # Speed label
        kmh   = int(self.velocity * 12)
        lbl   = self.font_small.render(f"{kmh}", True, (220, 220, 220))
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy + r + 4))
