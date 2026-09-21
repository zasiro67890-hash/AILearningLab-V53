import re
import asyncio
import hashlib
import json
import math
import os
import random
import time
import threading
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.async_api import async_playwright


# ============================================================
# AI LEARNING AGENT V33 - TACTICAL WORLD UNDERSTANDING + SAFE ROUTE SELECTION + V32
# ============================================================

GAME_URL = "https://adlearningsite.onrender.com/game"

MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "learning_memory_v31.json"
)

ROUTE_MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "route_memory_v31.json"
)

MEMORY_B_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "learning_memory_v31_b.json"
)

# Shared persistent data directory for V31 replay/analysis artifacts.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "backend", "data")
os.makedirs(DATA_DIR, exist_ok=True)


DIRECTOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_director_v31.json")
TRAJECTORY_FILE = os.path.join(DATA_DIR, "failure_trajectories_v31.json")
MAX_TRAJECTORIES = 2000
REPLAY_WINDOW = 12

# V31 self-directed learning
FAILURE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failure_analysis_v31.json")
FAILURE_FOCUS_BONUS = 0.25

# V31 predictive world model
WORLD_MODEL_FILE = os.path.join(DATA_DIR, "world_model_v31.json")
PHYSICS_GRAVITY = 0.6
PHYSICS_MOVE_SPEED = 5.0
PHYSICS_JUMP_POWER = -12.0
MODEL_HORIZON_FRAMES = 36
MODEL_LEARNING_RATE = 0.08



class PredictiveWorldModel:
    """Small empirical model layered over the known game physics.
    It predicts short-horizon landing position for each candidate action and
    slowly learns systematic prediction error from observed transitions.
    """
    def __init__(self):
        self.bias = {a: {"x": 0.0, "y": 0.0} for a in ACTIONS}
        self.samples = {a: 0 for a in ACTIONS}
        self.load()

    def load(self):
        try:
            with open(WORLD_MODEL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a in ACTIONS:
                if a in data.get("bias", {}):
                    self.bias[a] = data["bias"][a]
                self.samples[a] = int(data.get("samples", {}).get(a, 0))
        except Exception:
            pass

    def save(self):
        try:
            with open(WORLD_MODEL_FILE, "w", encoding="utf-8") as f:
                json.dump({"bias": self.bias, "samples": self.samples}, f, indent=2)
        except Exception:
            pass

    def predict(self, state, action):
        p = state.get("player", {})
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        vx = float(p.get("velocityX", 0.0))
        vy = float(p.get("velocityY", 0.0))
        grounded = bool(p.get("grounded", False))
        frames = MODEL_HORIZON_FRAMES
        if action in ("JUMP_RIGHT", "JUMP_LEFT") and grounded:
            vy = PHYSICS_JUMP_POWER
        direction = 1 if action in ("RIGHT", "JUMP_RIGHT") else -1 if action in ("LEFT", "JUMP_LEFT") else 0
        landing_y = None
        landing_x = x
        for _ in range(frames):
            if direction:
                vx = direction * PHYSICS_MOVE_SPEED
            else:
                vx *= 0.82
            x += vx
            y += vy
            vy += PHYSICS_GRAVITY
            if vy >= 0 and landing_y is None and not grounded:
                landing_y = y
        return {
            "x": landing_x + self.bias.get(action, {}).get("x", 0.0),
            "y": (landing_y if landing_y is not None else y) + self.bias.get(action, {}).get("y", 0.0),
            "error_bias": self.bias.get(action, {}).copy(),
        }

    def update(self, state_before, action, state_after):
        try:
            pred = self.predict(state_before, action)
            actual = state_after.get("player", {})
            ax = float(actual.get("x", 0.0))
            ay = float(actual.get("y", 0.0))
            ex = ax - pred["x"]
            ey = ay - pred["y"]
            b = self.bias[action]
            b["x"] += MODEL_LEARNING_RATE * ex
            b["y"] += MODEL_LEARNING_RATE * ey
            self.samples[action] += 1
            if self.samples[action] % 50 == 0:
                self.save()
        except Exception:
            pass

    def action_score(self, state, action):
        target = get_target(state, None)
        pred = self.predict(state, action)
        if not target:
            return 0.0
        left = float(target["x"])
        right = left + float(target["width"])
        px = pred["x"]
        if px < left:
            distance = left - px
        elif px > right:
            distance = px - right
        else:
            distance = 0.0
        center = (left + right) / 2.0
        center_error = abs(px - center)
        # Strongly reward predicted landing inside the target, then prefer
        # the center for a safer landing.
        return (260.0 if distance == 0 else -2.4 * distance) - 0.7 * center_error


class ModelBasedPlanner:
    """Bounded model-predictive planner using the learned world model."""
    def __init__(self, world_model, depth=3):
        self.world_model = world_model
        self.depth = depth
        self.last_plan = []
        self.last_score = 0.0

    def _score_x(self, x, target):
        if not target:
            return 0.0
        left = float(target.get("x", 0.0))
        right = left + float(target.get("width", 0.0))
        center = (left + right) / 2.0
        if left <= x <= right:
            return 300.0 - 0.45 * abs(x - center)
        return -2.8 * min(abs(x-left), abs(x-right))

    def plan(self, state, target):
        allowed = valid_actions(state)
        if not allowed or not target:
            return None, [], 0.0
        # Depth 3 keeps the central brain cheap with many workers.
        sequences = [[]]
        for _ in range(self.depth):
            sequences = [seq + [a] for seq in sequences for a in allowed]
        best_seq, best_score = [allowed[0]], -1e18
        for seq in sequences:
            virtual = {"player": dict(state.get("player", {})),
                       "currentLevel": state.get("currentLevel", state.get("level", 1)),
                       "level": state.get("level", state.get("currentLevel", 1))}
            score = 0.0
            for i, action in enumerate(seq):
                pred = self.world_model.predict(virtual, action)
                score += self._score_x(pred["x"], target)
                if action.startswith("JUMP"):
                    score -= 4.0
                if i and action != seq[i-1]:
                    score -= 0.5
                virtual["player"].update({"x": pred["x"], "y": pred["y"],
                                          "velocityX": PHYSICS_MOVE_SPEED if action in ("RIGHT", "JUMP_RIGHT") else -PHYSICS_MOVE_SPEED if action in ("LEFT", "JUMP_LEFT") else 0.0,
                                          "velocityY": 0.0, "grounded": not action.startswith("JUMP")})
            score += 0.12 * self.world_model.action_score(state, seq[0])
            if score > best_score:
                best_score, best_seq = score, seq
        self.last_plan = list(best_seq)
        self.last_score = float(best_score)
        return best_seq[0], best_seq, best_score


class PrecisionLandingController:
    """V32: convert the predicted landing error into precise action + hold time.

    The game uses a simple deterministic physics model. Instead of always
    holding an action for the same duration, V32 changes the key-hold window
    according to distance, horizontal velocity, flight phase and target width.
    It also learns a small timing bias from observed landing errors.
    """
    def __init__(self):
        self.timing_bias = {}
        self.landings = 0
        self.last = {}
        self.load()

    def _regime(self, player):
        return f"{velocity_bucket(float(player.get('velocityX', 0)))}|{vertical_velocity_bucket(float(player.get('velocityY', 0)))}"

    def _frames_to_contact(self, player):
        vy = float(player.get("velocityY", 0.0))
        # Estimate until the descending phase reaches a useful contact window.
        if vy <= 0:
            return max(5, min(36, int((-vy) * 1.6 + 10)))
        return max(5, min(28, int(vy * 1.35 + 6)))

    def decide(self, state, target, fallback_action=None):
        if not state or not target:
            return fallback_action, ACTION_TIME_MS, {"mode": "fallback"}
        p = state["player"]
        px = float(p.get("x", 0.0)) + 10.0
        vx = float(p.get("velocityX", 0.0))
        left = float(target.get("x", 0.0)) + 8.0
        right = left + max(4.0, float(target.get("width", 0.0)) - 16.0)
        center = (left + right) / 2.0
        predicted = predicted_landing_x(p)
        error = center - predicted
        inside = left <= predicted <= right
        regime = self._regime(p)
        bias = float(self.timing_bias.get(regime, 0.0))
        frames = self._frames_to_contact(p)

        if p.get("grounded", False):
            direction = "RIGHT" if center > px + 10 else "LEFT" if center < px - 10 else "WAIT"
            if direction == "WAIT":
                return fallback_action or "WAIT", V32_MIN_HOLD_MS, {"mode":"centered", "error":error, "predicted":predicted}
            # Longer holds when the target is far away, shorter near the edge.
            distance = abs(center - px)
            hold = min(V32_GROUNDED_HOLD_MS, 42 + int(min(55, distance * 0.12)))
            action = direction
            # If the target is reachable by a jump, let the high-level controller
            # keep the jump decision; V32 mainly controls its horizontal duration.
            if target.get("y", p.get("y", 0)) < float(p.get("y", 0)) - 18:
                action = "JUMP_RIGHT" if direction == "RIGHT" else "JUMP_LEFT"
                hold = min(V32_MAX_HOLD_MS, hold + 8)
            return action, max(V32_MIN_HOLD_MS, hold), {"mode":"grounded", "error":error, "frames":frames}

        # Airborne: steer based on predicted landing interval, not current X.
        if predicted < left:
            action = "RIGHT"
        elif predicted > right:
            action = "LEFT"
        elif abs(vx) > 2.0:
            action = "LEFT" if vx > 0 else "RIGHT"
        else:
            action = "WAIT"

        miss = abs(error)
        # Larger correction for large miss; tiny correction near the center.
        hold = V32_AIR_HOLD_MS + int(min(55, miss * 0.22))
        if inside:
            hold = max(V32_MIN_HOLD_MS, hold - 12)
        hold += int(bias)
        hold = max(V32_MIN_HOLD_MS, min(V32_MAX_HOLD_MS, hold))
        self.last = {"regime":regime, "predicted":predicted, "target_center":center,
                     "error":error, "frames":frames, "action":action, "hold_ms":hold}
        return action, hold, self.last

    def learn_landing(self, state_before, target, actual_x):
        if not state_before or not target:
            return
        p = state_before.get("player", {})
        predicted = predicted_landing_x(p)
        center = float(target.get("x", 0.0)) + float(target.get("width", 0.0)) / 2.0
        # Positive error means we predicted/ended too far left; negative means too far right.
        error = center - float(actual_x)
        regime = self._regime(p)
        old = float(self.timing_bias.get(regime, 0.0))
        # Keep the learned correction intentionally small so it cannot destabilize RL.
        self.timing_bias[regime] = max(-25.0, min(25.0, old + V32_TIMING_LEARNING_RATE * error))
        self.landings += 1
        self.save()

    def save(self):
        try:
            with open(V32_TIMING_FILE, "w", encoding="utf-8") as f:
                json.dump({"timing_bias": self.timing_bias, "landings": self.landings}, f, indent=2)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(V32_TIMING_FILE):
                with open(V32_TIMING_FILE, "r", encoding="utf-8") as f:
                    d=json.load(f)
                self.timing_bias=d.get("timing_bias", {})
                self.landings=int(d.get("landings", 0))
        except Exception:
            self.timing_bias={}


class TacticalWorldModel:
    """V33: turn the raw game state into tactical decisions.

    The controller evaluates several reachable platforms, marks dangerous
    landing zones near spikes/enemies, rewards wider/successful routes, and
    gives the planner a safer target instead of blindly selecting the first
    platform candidate. It learns a tiny per-route hazard penalty so repeated
    near-misses make the corresponding route less attractive.
    """
    def __init__(self):
        self.route_hazard_memory = {}
        self.decisions = 0
        self.safe_decisions = 0
        self.last = {}
        self.load()

    def load(self):
        try:
            with open(V33_TACTICAL_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.route_hazard_memory = d.get("route_hazard_memory", {})
            self.decisions = int(d.get("decisions", 0))
            self.safe_decisions = int(d.get("safe_decisions", 0))
        except Exception:
            pass

    def save(self):
        try:
            with open(V33_TACTICAL_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "route_hazard_memory": self.route_hazard_memory,
                    "decisions": self.decisions,
                    "safe_decisions": self.safe_decisions,
                }, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _center(obj):
        return (float(obj.get("x", 0)) + float(obj.get("width", 0)) / 2.0,
                float(obj.get("y", 0)) + float(obj.get("height", 0)) / 2.0)

    def _hazard_score(self, platform, state):
        plx = float(platform.get("x", 0))
        pry = float(platform.get("y", 0))
        pright = plx + float(platform.get("width", 0))
        score = 0.0
        hazards = []
        for spike in state.get("spikes", []):
            sx, sy = self._center(spike)
            if sx < plx - V33_HAZARD_RADIUS or sx > pright + V33_HAZARD_RADIUS:
                continue
            vertical = abs(sy - pry)
            if vertical <= 45 + float(spike.get("height", 0)):
                overlap = max(0.0, min(pright, sx + 8) - max(plx, sx - 8))
                proximity = max(0.0, V33_HAZARD_RADIUS - abs(sx - (plx + pright) / 2.0))
                score += V33_HAZARD_PENALTY * (0.65 + proximity / V33_HAZARD_RADIUS * 0.35)
                score += overlap * 1.5
                hazards.append("spike")
        for enemy in state.get("enemies", []):
            ex, ey = self._center(enemy)
            if abs(ey - pry) <= 55 and plx - V33_ENEMY_RADIUS <= ex <= pright + V33_ENEMY_RADIUS:
                proximity = max(0.0, V33_ENEMY_RADIUS - abs(ex - (plx + pright) / 2.0))
                score += V33_ENEMY_PENALTY * (0.45 + proximity / V33_ENEMY_RADIUS * 0.55)
                hazards.append("enemy")
        return score, hazards

    def rank_targets(self, state, route_memory=None):
        candidates = platform_candidates(state, route_memory)
        if not candidates:
            return []
        p = state.get("player", {})
        px = float(p.get("x", 0)) + 10.0
        py = float(p.get("y", 0))
        goal = state.get("goal") or {}
        gx = float(goal.get("x", 0)) + float(goal.get("width", 0)) / 2.0 if goal else None
        ranked = []
        for item in candidates:
            platform = item["platform"]
            rid = route_id(platform)
            hazard, hazard_types = self._hazard_score(platform, state)
            memory_penalty = float(self.route_hazard_memory.get(rid, 0.0))
            center = float(platform["x"]) + float(platform["width"]) / 2.0
            distance = abs(center - px)
            vertical_gap = abs(float(platform["y"]) - py)
            score = float(item["score"]) - hazard - memory_penalty
            if hazard == 0:
                score += V33_SAFE_ROUTE_BONUS
            if gx is not None and abs(center - gx) < 150:
                score += V33_GOAL_ROUTE_BONUS
            if vertical_gap > 220:
                score -= (vertical_gap - 220) * V33_REACHABILITY_PENALTY
            # Mildly prefer routes that make forward progress without requiring
            # a huge horizontal correction.
            if center > px:
                score += min(80.0, max(0.0, center - px) * 0.08)
            ranked.append({
                "platform": platform, "score": score, "base_score": item["score"],
                "hazard_score": hazard, "hazards": sorted(set(hazard_types)),
                "hazard_memory": memory_penalty, "distance": distance,
                "vertical_gap": vertical_gap,
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def choose_target(self, state, route_memory=None):
        ranked = self.rank_targets(state, route_memory)
        if not ranked:
            self.last = {"target": None, "ranked": []}
            return None
        self.decisions += 1
        best = ranked[0]
        if best["hazard_score"] <= 0:
            self.safe_decisions += 1
        self.last = {
            "target": best["platform"],
            "score": best["score"],
            "hazard_score": best["hazard_score"],
            "hazards": best["hazards"],
            "alternatives": [
                {"x": x["platform"]["x"], "y": x["platform"]["y"],
                 "width": x["platform"]["width"], "score": round(x["score"], 2),
                 "hazards": x["hazards"]}
                for x in ranked[:4]
            ],
        }
        return best["platform"]

    def record_outcome(self, target, success, events=None):
        if not target:
            return
        rid = route_id(target)
        events_text = " ".join(events or [])
        hazard_event = any(x in events_text for x in ("SPIKE", "ENEMY", "DEATH", "FAIL"))
        old = float(self.route_hazard_memory.get(rid, 0.0))
        if success:
            old *= 0.92
        elif hazard_event:
            old = min(180.0, old * 0.96 + 12.0)
        self.route_hazard_memory[rid] = old
        if self.decisions % 50 == 0:
            self.save()


class EvolutionManager:
    """V31: real strategy evolution layered on the shared learning brain.

    Each worker is assigned a genome containing action biases. The genome changes
    which actions the shared brain prefers, so candidates genuinely behave
    differently. Episode results rank the candidates; elites survive and the rest
    are mutated clones for the next generation.
    """
    def __init__(self):
        self.generation = 0
        self.population = []
        self.next_worker_index = 0
        self.episodes_seen = 0
        self.best_score = -1e18
        self.best_policy = {}
        self.load()
        self._ensure_population()

    def _random_genome(self, gid):
        return {"id": gid, "score": 0.0, "episodes": 0, "goals": 0, "landings": 0,
                "mutation": V31_MUTATION_RATE,
                "bias": {a: random.uniform(-1.0, 1.0) for a in ACTIONS}}

    def _ensure_population(self):
        while len(self.population) < V31_POPULATION_SIZE:
            self.population.append(self._random_genome(len(self.population)))

    def assign_worker(self, worker_id):
        gid = self.next_worker_index % V31_POPULATION_SIZE
        self.next_worker_index += 1
        return gid

    def action_bias(self, genome_id, action):
        try:
            g = self.population[int(genome_id) % len(self.population)]
            return float(g.get("bias", {}).get(action, 0.0))
        except Exception:
            return 0.0

    def record_episode(self, genome_id, score, goals=0, landings=0):
        if not self.population:
            self._ensure_population()
        g = self.population[int(genome_id) % len(self.population)]
        score = float(score)
        g["score"] = 0.8 * float(g.get("score", 0.0)) + 0.2 * score
        g["episodes"] = int(g.get("episodes", 0)) + 1
        g["goals"] = int(g.get("goals", 0)) + int(goals)
        g["landings"] = int(g.get("landings", 0)) + int(landings)
        self.episodes_seen += 1
        if g["score"] > self.best_score:
            self.best_score = g["score"]
            self.best_policy = json.loads(json.dumps(g))
        return self.episodes_seen % V31_EVOLVE_EVERY_EPISODES == 0

    def evolve(self):
        ranked = sorted(self.population, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        elites = ranked[:V31_ELITES]
        new_pop = []
        for i, elite in enumerate(elites):
            e = json.loads(json.dumps(elite)); e["id"] = i
            new_pop.append(e)
        while len(new_pop) < V31_POPULATION_SIZE:
            parent = random.choice(elites)
            child = json.loads(json.dumps(parent))
            child["id"] = len(new_pop)
            child["score"] = 0.0; child["episodes"] = 0; child["goals"] = 0; child["landings"] = 0
            child["mutation"] = min(0.5, max(0.02, float(parent.get("mutation", V31_MUTATION_RATE)) * random.uniform(0.9, 1.15)))
            for a in ACTIONS:
                if random.random() < 0.85:
                    child["bias"][a] = max(-2.5, min(2.5, float(child["bias"].get(a, 0.0)) + random.gauss(0, V31_MUTATION_SCALE)))
            new_pop.append(child)
        self.population = new_pop
        self.generation += 1
        self.next_worker_index = 0
        self.save()
        return ranked[:V31_ELITES]

    def save(self):
        try:
            with open(EVOLUTION_POP_FILE, "w", encoding="utf-8") as f:
                json.dump({"generation": self.generation, "population": self.population,
                           "episodes_seen": self.episodes_seen, "best_score": self.best_score,
                           "best_policy": self.best_policy}, f, indent=2)
        except Exception as e:
            print(f"⚠️ Strategy population save failed: {e}")

    def load(self):
        try:
            if os.path.exists(EVOLUTION_POP_FILE):
                with open(EVOLUTION_POP_FILE, "r", encoding="utf-8") as f:
                    d=json.load(f)
                self.generation=int(d.get("generation",0)); self.population=d.get("population",[])
                self.episodes_seen=int(d.get("episodes_seen",0)); self.best_score=float(d.get("best_score",-1e18))
                self.best_policy=d.get("best_policy",{})
        except Exception as e:
            print(f"⚠️ Strategy population load failed: {e}")
            self.population=[]

    def summary(self):
        ranked=sorted(self.population,key=lambda x: float(x.get("score",0.0)),reverse=True)
        return {"generation":self.generation,"episodes_seen":self.episodes_seen,
                "best_score":self.best_score,"population":ranked}


class TrainingDirector:
    """V31 curriculum director: weak routes get higher action-selection pressure."""
    def __init__(self):
        self.stats = {}
        self.focus_route = None
        self.focus_level = None
        self.load()

    def record(self, key, success):
        if key is None:
            return
        key = str(key)
        s = self.stats.setdefault(key, {"attempts": 0, "successes": 0})
        s["attempts"] += 1
        s["successes"] += int(bool(success))
        self.recompute()

    def priority(self, key):
        s = self.stats.get(str(key), {"attempts": 0, "successes": 0})
        a = s["attempts"]
        if a < 3:
            return 1.0
        rate = s["successes"] / a
        deficit = max(0.0, CURRICULUM_SUCCESS_TARGET - rate)
        return max(CURRICULUM_MIN_WEIGHT, min(CURRICULUM_MAX_WEIGHT, 1.0 + deficit / CURRICULUM_SUCCESS_TARGET * 4.0))

    def focus(self, keys):
        return max(keys, key=self.priority) if keys else None

    def recompute(self):
        candidates = [(k, self.priority(k)) for k, s in self.stats.items() if s.get("attempts", 0) >= 3]
        self.focus_route = max(candidates, key=lambda x: x[1])[0] if candidates else None

    def save(self):
        try:
            with open(DIRECTOR_FILE, "w", encoding="utf-8") as f:
                json.dump({"stats": self.stats, "focus_route": self.focus_route, "saved_at": time.time()}, f, indent=2)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(DIRECTOR_FILE):
                with open(DIRECTOR_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.stats = data.get("stats", {})
                self.focus_route = data.get("focus_route")
        except Exception:
            self.stats = {}

    def summary(self):
        out = []
        for k, s in self.stats.items():
            a = s.get("attempts", 0)
            rate = s.get("successes", 0) / a if a else 0.0
            out.append({"key": k, "attempts": a, "success_rate": round(rate, 3), "priority": round(self.priority(k), 3)})
        return sorted(out, key=lambda x: -x["priority"])[:20]


ACTIONS = [
    "RIGHT",
    "LEFT",
    "JUMP_RIGHT",
    "JUMP_LEFT",
    "WAIT",
]

MAX_STEPS = max(1, int(os.getenv("TRAIN_STEPS", "10000")))
AUTO_TRAIN_CYCLES = 5
AUTO_EVAL_WORKERS = 2
AUTO_MIN_IMPROVEMENT = 1.0
NUM_ENVS = 1
BRAIN_HOST = os.getenv("BRAIN_HOST", "0.0.0.0")
BRAIN_PORT = int(os.getenv("PORT", os.getenv("BRAIN_PORT", "8090")))
BRAIN_URL = os.getenv("BRAIN_URL", f"http://127.0.0.1:{BRAIN_PORT}").rstrip("/")
BRAIN_API_KEY = os.getenv("BRAIN_API_KEY", "")

# V53 REGIONAL WORKER POOLS
# Supported deployment regions: US, GB/UK, EU.  These identify the
# worker deployment pool only; they are NOT user/device location telemetry.
_REGION_ALIASES = {
    "US": "US", "VA": "US", "OH": "OH",
    "EU": "EU", "GB": "EU", "UK": "EU",
    "SG": "SG", "OR": "OR",
}

REGIONAL_RELAY_URLS = {
    "US": "https://v53-us-relay.onrender.com",
    "OH": "https://v53-oh-relay.onrender.com",
    "EU": "https://v53-gb-relay.onrender.com",
    "SG": "https://ailearninglab-v53-1.onrender.com",
    "OR": "https://ailearninglab-v53-2.onrender.com",
}

# Relay overrides can still be supplied explicitly, but default to the
# selected regional pool. Support both the new names and the existing
# environment names already used by the Render relays.
REGIONAL_RELAY_KEY_ENVS = {
    "US": ("RELAY_KEY_US", "RELAY_KEY"),
    "OH": ("RELAY_KEY_OH", "OH_RELAY_KEY"),
    "EU": ("RELAY_KEY_EU", "EU_RELAY_KEY"),
    "SG": ("RELAY_KEY_SG", "SG_RELAY_KEY"),
    "OR": ("RELAY_KEY_OR", "OR_RELAY_KEY"),
}

BRAIN_RELAY_URL = os.getenv("BRAIN_RELAY_URL", "").rstrip("/")
BRAIN_RELAY_API_KEY = os.getenv("BRAIN_RELAY_API_KEY", "")

WORKER_REGION = _REGION_ALIASES.get(
    os.getenv("WORKER_REGION", "US").strip().upper(), "US"
)

if not BRAIN_RELAY_URL:
    BRAIN_RELAY_URL = REGIONAL_RELAY_URLS[WORKER_REGION]

if not BRAIN_RELAY_API_KEY:
    for _key_env in REGIONAL_RELAY_KEY_ENVS[WORKER_REGION]:
        _key_value = os.getenv(_key_env, "")
        if _key_value:
            BRAIN_RELAY_API_KEY = _key_value
            break

REPLAY_CAPACITY = 10000
REPLAY_BATCH_SIZE = 64
REPLAY_START_SIZE = 128
REPLAY_EVERY = 4
BENCHMARK_WINDOW = 100
WEAK_ROUTE_THRESHOLD = 0.35
FOCUS_BONUS = 0.75
METRICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_metrics_v33.json")
CHAMPION_A_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_memory_v33.json")
CHAMPION_B_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion_memory_v33_b.json")
EVAL_INTERVAL = 25
BEST_POLICY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_policy_v33.json")
ACTION_TIME_MS = 60

# V31 actual evolving strategy population
EVOLUTION_POP_FILE = os.path.join(DATA_DIR, "strategy_population_v33.json")
V31_POPULATION_SIZE = 8
V31_ELITES = 3
V31_EVOLVE_EVERY_EPISODES = 8
V32_TIMING_FILE = os.path.join(DATA_DIR, "jump_timing_v32.json")
V33_TACTICAL_FILE = os.path.join(DATA_DIR, "tactical_world_v33.json")
V31_MUTATION_RATE = 0.18
V31_MUTATION_SCALE = 0.35

# V32 precision landing controller
V32_MIN_HOLD_MS = 25
V32_MAX_HOLD_MS = 115
V32_GROUNDED_HOLD_MS = 85
V32_AIR_HOLD_MS = 42
V32_TIMING_LEARNING_RATE = 0.08

# V33 tactical scene understanding
V33_HAZARD_RADIUS = 55.0
V33_ENEMY_RADIUS = 85.0
V33_SAFE_ROUTE_BONUS = 90.0
V33_GOAL_ROUTE_BONUS = 120.0
V33_HAZARD_PENALTY = 160.0
V33_ENEMY_PENALTY = 75.0
V33_REACHABILITY_PENALTY = 2.0

# V31 AUTONOMOUS TRAINING MANAGER
MANAGER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_manager_v33.json")
RUN_SUMMARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_summary_v33.json")


class AutonomousTrainingManager:
    """Coordinates train -> evaluate -> champion/rollback cycles across processes."""
    def __init__(self):
        self.generation = 0
        self.best_eval_reward = -1e18
        self.best_eval_level = 0
        self.best_eval_goals = 0
        self.stagnation = 0
        self.load()

    def load(self):
        try:
            if os.path.exists(MANAGER_FILE):
                with open(MANAGER_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.generation = int(d.get("generation", 0))
                self.best_eval_reward = float(d.get("best_eval_reward", -1e18))
                self.best_eval_level = int(d.get("best_eval_level", 0))
                self.best_eval_goals = int(d.get("best_eval_goals", 0))
                self.stagnation = int(d.get("stagnation", 0))
        except Exception as e:
            print(f"⚠️ Manager load failed: {e}")

    def save(self):
        try:
            with open(MANAGER_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "generation": self.generation,
                    "best_eval_reward": self.best_eval_reward,
                    "best_eval_level": self.best_eval_level,
                    "best_eval_goals": self.best_eval_goals,
                    "stagnation": self.stagnation,
                    "updated_at": time.time(),
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Manager save failed: {e}")

    def accept(self, reward, level, goals):
        improved = reward >= self.best_eval_reward + AUTO_MIN_IMPROVEMENT
        if improved:
            self.best_eval_reward = float(reward)
            self.best_eval_level = int(level)
            self.best_eval_goals = int(goals)
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.generation += 1
        self.save()
        return improved

    def summary(self):
        return {
            "generation": self.generation,
            "best_eval_reward": self.best_eval_reward,
            "best_eval_level": self.best_eval_level,
            "best_eval_goals": self.best_eval_goals,
            "stagnation": self.stagnation,
        }


# V31 REAL CURRICULUM CONTROL
CURRICULUM_SUCCESS_TARGET = 0.70
CURRICULUM_MIN_WEIGHT = 0.35
CURRICULUM_MAX_WEIGHT = 5.0
CURRICULUM_FOCUS_BONUS = 35.0

# V31 evolutionary training
POPULATION_SIZE = 6
ELITE_COUNT = 2
MUTATION_RATE = 0.12
MUTATION_SCALE = 0.15
EVOLUTION_INTERVAL = 5000
EVOLUTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolution_v29.json")
DIRECTOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_director_v31.json")

ALPHA = 0.20
GAMMA = 0.95

# V6 telemetry / reward tuning
LANDING_REWARD = 150.0
LEVEL_REWARD = 250.0
GOAL_REWARD = 1000.0
DEATH_PENALTY = -150.0

EPSILON_START = 0.80
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999


# ============================================================
# MEMORY
# ============================================================

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        print("🧠 Starting fresh V31 memory")
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)

        print(f"🧠 Loaded {len(memory)} learned states")
        return memory

    except Exception as e:
        print(f"⚠️ Memory load failed: {e}")
        return {}


def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)

        print(f"💾 Memory saved: {len(memory)} states")

    except Exception as e:
        print(f"⚠️ Memory save failed: {e}")

def load_memory_b():
    if not os.path.exists(MEMORY_B_FILE):
        print("🧠 Starting fresh V31 Double-Q table B")
        return {}
    try:
        with open(MEMORY_B_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"🧠 Loaded {len(data)} states from Q-table B")
        return data
    except Exception as e:
        print(f"⚠️ Q-table B load failed: {e}")
        return {}


def save_memory_b(memory):
    try:
        with open(MEMORY_B_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        print(f"💾 Q-table B saved: {len(memory)} states")
    except Exception as e:
        print(f"⚠️ Q-table B save failed: {e}")


def load_route_memory():
    if not os.path.exists(ROUTE_MEMORY_FILE):
        print("🗺️ Starting fresh V31 route memory")
        return {}
    try:
        with open(ROUTE_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"🗺️ Loaded {len(data)} route entries")
        return data
    except Exception as e:
        print(f"⚠️ Route memory load failed: {e}")
        return {}


def save_route_memory(route_memory):
    try:
        with open(ROUTE_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(route_memory, f, indent=2)
        print(f"💾 Route memory saved: {len(route_memory)} entries")
    except Exception as e:
        print(f"⚠️ Route memory save failed: {e}")


def route_id(platform):
    if not platform:
        return None
    return f"{round(platform.get('x', 0)/20)*20}:{round(platform.get('y', 0)/20)*20}:{round(platform.get('width', 0)/20)*20}"


def update_route_memory(route_memory, platform, success):
    rid = route_id(platform)
    if not rid:
        return
    item = route_memory.setdefault(rid, {"attempts": 0, "successes": 0})
    item["attempts"] += 1
    if success:
        item["successes"] += 1


def route_success_rate(route_memory, platform):
    rid = route_id(platform)
    if not rid:
        return 0.0
    item = route_memory.get(rid, {})
    attempts = item.get("attempts", 0)
    return item.get("successes", 0) / attempts if attempts else 0.0


# ============================================================
# BROWSER
# ============================================================

async def create_browser(playwright):
    browser = await playwright.chromium.launch(
        headless=os.getenv("CHROMIUM_HEADLESS", "1").strip().lower() not in {"0", "false", "no"},
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    )

    context = await browser.new_context(
        viewport={
            "width": 1000,
            "height": 650,
        }
    )

    page = await context.new_page()

    await page.goto(
        GAME_URL,
        wait_until="networkidle",
        timeout=60000
    )

    print("✅ Game page loaded")

    await page.wait_for_timeout(5000)

    # V32 diagnostic: report frame structure before training starts.
    try:
        print(f"🔎 Frames detected: {len(page.frames)}")
        for i, frame in enumerate(page.frames):
            print(f"   FRAME {i}: {frame.url}")
    except Exception as e:
        print(f"⚠️ Frame diagnostic failed: {e}")

    return browser, context, page


# ============================================================
# CDP GAME STATE
# ============================================================

async def get_state(page, cdp):
    """
    V32 robust game-state bridge.

    The Render game can keep its variables in a different JS execution
    context (for example an iframe/module), so assuming bare globals such
    as `player` are always visible to CDP is unsafe.

    We try:
      1) the CDP main context
      2) Playwright main-world evaluation
      3) every iframe/frame
      4) an optional window.__AI_GAME_STATE__ bridge if the game exposes one

    The returned structure is unchanged for the AI.
    """

    state_expression = r"""
    (() => {
        try {
            const P = (typeof player !== "undefined")
                ? player
                : (window.player || window.__player || null);

            const PL = (typeof platforms !== "undefined")
                ? platforms
                : (window.platforms || window.__platforms || []);

            const MP = (typeof movingPlatforms !== "undefined")
                ? movingPlatforms
                : (window.movingPlatforms || window.__movingPlatforms || []);

            const G = (typeof goal !== "undefined")
                ? goal
                : (window.goal || window.__goal || null);

            const LV = (typeof currentLevel !== "undefined")
                ? currentLevel
                : (window.currentLevel ?? window.__currentLevel ?? 1);

            const SC = (typeof score !== "undefined")
                ? score
                : (window.score ?? window.__score ?? 0);

            const DE = (typeof deaths !== "undefined")
                ? deaths
                : (window.deaths ?? window.__deaths ?? 0);

            const SP = (typeof spikes !== "undefined")
                ? spikes
                : (window.spikes || window.__spikes || []);

            const EN = (typeof enemies !== "undefined")
                ? enemies
                : (window.enemies || window.__enemies || []);

            // Optional explicit bridge supplied by a patched game.
            if (!P && typeof window.__AI_GAME_STATE__ === "function") {
                const s = window.__AI_GAME_STATE__();
                if (s && s.player) return s;
            }

            if (!P) {
                return {
                    __error:
                        "Game state object `player` is not visible in this execution context",
                    __diagnostics: {
                        href: location.href,
                        readyState: document.readyState,
                        frames: window.frames.length,
                        playerType: typeof player,
                        windowPlayer: !!window.player,
                        platformType: typeof platforms
                    }
                };
            }

            return {
                player: {
                    x: Number(P.x) || 0,
                    y: Number(P.y) || 0,
                    velocityX: Number(P.velocityX) || 0,
                    velocityY: Number(P.velocityY) || 0,
                    grounded: !!P.grounded
                },

                platforms: Array.isArray(PL) ? PL.map(p => ({
                    x: Number(p.x) || 0,
                    y: Number(p.y) || 0,
                    width: Number(p.width) || 0,
                    height: Number(p.height) || 0
                })) : [],

                movingPlatforms: Array.isArray(MP) ? MP.map(p => ({
                    x: Number(p.x) || 0,
                    y: Number(p.y) || 0,
                    width: Number(p.width) || 0,
                    height: Number(p.height) || 0
                })) : [],

                spikes: Array.isArray(SP) ? SP.map(o => ({
                    x: Number(o.x) || 0, y: Number(o.y) || 0,
                    width: Number(o.width) || 0, height: Number(o.height) || 0
                })) : [],

                enemies: Array.isArray(EN) ? EN.map(o => ({
                    x: Number(o.x) || 0, y: Number(o.y) || 0,
                    width: Number(o.width) || 0, height: Number(o.height) || 0,
                    velocityX: Number(o.velocityX) || 0
                })) : [],

                goal: G ? {
                    x: Number(G.x) || 0,
                    y: Number(G.y) || 0,
                    width: Number(G.width) || 0,
                    height: Number(G.height) || 0
                } : null,

                level: Number(LV) || 1,
                currentLevel: Number(LV) || 1,
                score: Number(SC) || 0,
                deaths: Number(DE) || 0
            };
        } catch (e) {
            return {
                __error: String(e),
                __stack: e && e.stack ? String(e.stack) : ""
            };
        }
    })()
    """

    async def normalize(value, source):
        if not value:
            return None
        if "__error" in value:
            diagnostics = value.get("__diagnostics")
            if diagnostics:
                print(f"⚠️ {source}: {value['__error']} | {diagnostics}")
            else:
                print(f"⚠️ {source}: {value['__error']}")
            return None
        if isinstance(value, dict) and value.get("player"):
            return value
        return None

    # --------------------------------------------------------
    # 1) CDP
    # --------------------------------------------------------
    try:
        result = await cdp.send(
            "Runtime.evaluate",
            {
                "expression": state_expression,
                "returnByValue": True,
                "awaitPromise": True
            }
        )

        value = result.get("result", {}).get("value")
        state = await normalize(value, "CDP main context")
        if state:
            return state
    except Exception as e:
        print(f"⚠️ CDP state read failed: {e}")

    # --------------------------------------------------------
    # 2) Playwright main world
    # --------------------------------------------------------
    try:
        value = await page.evaluate(state_expression)
        state = await normalize(value, "Playwright main context")
        if state:
            return state
    except Exception as e:
        print(f"⚠️ Playwright state read failed: {e}")

    # --------------------------------------------------------
    # 3) Every frame
    # --------------------------------------------------------
    for index, frame in enumerate(page.frames):
        try:
            value = await frame.evaluate(state_expression)
            state = await normalize(value, f"frame {index}")
            if state:
                print(f"🧠 Game state found in frame {index}")
                return state
        except Exception:
            continue

    print("❌ V32 could not access game state in any page/frame")
    return None

# ============================================================
# STATE DISCRETIZATION
# ============================================================

def velocity_bucket(vx):
    if vx < -4:
        return "LEFT_FAST"

    if vx < -1:
        return "LEFT"

    if vx <= 1:
        return "STOP"

    if vx <= 4:
        return "RIGHT"

    return "RIGHT_FAST"


def vertical_velocity_bucket(vy):
    if vy < -6:
        return "RISING_FAST"
    if vy < -1:
        return "RISING"
    if vy <= 1:
        return "APEX"
    if vy <= 6:
        return "FALLING"
    return "FALLING_FAST"


def predicted_landing_x(player):
    vy = player["velocityY"]
    vx = player["velocityX"]
    frames = min(35, max(6, int(abs(vy) * 2) if vy < 0 else int(vy * 1.5) + 8))
    return player["x"] + vx * frames


def distance_bucket(distance):
    if distance < -250:
        return "BEHIND"

    if distance < -100:
        return "LEFT_FAR"

    if distance < -40:
        return "LEFT"

    if distance < 40:
        return "CENTER"

    if distance < 100:
        return "RIGHT"

    if distance < 250:
        return "RIGHT_FAR"

    return "FAR_RIGHT"


def height_bucket(player_y, target_y):
    difference = target_y - player_y

    if difference < -80:
        return "TARGET_HIGH"

    if difference < -25:
        return "TARGET_UP"

    if difference <= 25:
        return "SAME_HEIGHT"

    if difference <= 80:
        return "TARGET_DOWN"

    return "TARGET_LOW"


def state_key(state):
    if not state:
        return "UNKNOWN"

    player = state["player"]

    target = get_target(state)

    if target:
        target_center = (
            target["x"] +
            target["width"] / 2
        )

        player_center = (
            player["x"] + 10
        )

        distance = (
            target_center -
            player_center
        )

        distance_state = distance_bucket(
            distance
        )

        height_state = height_bucket(
            player["y"],
            target["y"]
        )

    else:
        distance_state = "NO_TARGET"
        height_state = "NO_TARGET"

    predicted = predicted_landing_x(player)
    if target:
        landing_error = target_center - predicted
        if landing_error < -100:
            landing_state = "OVERSHOOT_LEFT"
        elif landing_error < -35:
            landing_state = "LEFT_OF_CENTER"
        elif landing_error <= 35:
            landing_state = "GOOD"
        elif landing_error <= 100:
            landing_state = "RIGHT_OF_CENTER"
        else:
            landing_state = "UNDERSHOOT"
        width = target["width"]
        width_state = "NARROW" if width < 140 else ("MEDIUM" if width < 220 else "WIDE")
    else:
        landing_state = "NO_TARGET"
        width_state = "NO_TARGET"

    return "|".join([
        distance_state,
        velocity_bucket(player["velocityX"]),
        vertical_velocity_bucket(player["velocityY"]),
        height_state,
        width_state,
        landing_state,
        "GROUND" if player["grounded"] else "AIR",
    ])


# ============================================================
# Q TABLE
# ============================================================

def get_q_values(memory, key):
    if key not in memory:
        memory[key] = {
            action: 0.0
            for action in ACTIONS
        }

    return memory[key]


# ============================================================
# V8 EXPERIENCE REPLAY
# ============================================================

def add_experience(replay, state_key_value, action, reward, next_key, done):
    replay.append({"state": state_key_value, "action": action, "reward": float(reward), "next_state": next_key, "done": bool(done)})
    if len(replay) > REPLAY_CAPACITY:
        del replay[:len(replay) - REPLAY_CAPACITY]

def replay_learn(memory, replay):
    if len(replay) < REPLAY_START_SIZE:
        return 0
    count = min(REPLAY_BATCH_SIZE, len(replay))
    pool = list(replay)
    batch = []
    weights = [max(0.001, float(e.get("priority", 1.0))) for e in pool]
    for _ in range(count):
        if not pool:
            break
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        batch.append(pool.pop(idx))
        weights.pop(idx)
    updates = 0
    for e in batch:
        q = get_q_values(memory, e["state"])
        current = q.get(e["action"], 0.0)
        if e["done"]:
            target = e["reward"]
        else:
            nq = get_q_values(memory, e["next_state"])
            best_action = max(nq, key=nq.get) if nq else WAIT
            target = e["reward"] + GAMMA * nq.get(best_action, 0.0)
        td_error = target - current
        q[e["action"]] = current + ALPHA * td_error
        e["priority"] = max(0.1, abs(td_error) + 0.1)
        updates += 1
    return updates


def update_benchmark(reward, level=0, goal=False, death=False, landing=False):
    """Persist lightweight rolling training metrics and best-policy score."""
    try:
        data = {"rewards": [], "levels": [], "goals": 0, "deaths": 0, "landings": 0, "best_reward": None}
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r", encoding="utf-8") as f: data.update(json.load(f))
        data["rewards"].append(float(reward)); data["levels"].append(int(level))
        data["rewards"] = data["rewards"][-1000:]; data["levels"] = data["levels"][-1000:]
        data["goals"] += int(bool(goal)); data["deaths"] += int(bool(death)); data["landings"] += int(bool(landing))
        recent=data["rewards"][-BENCHMARK_WINDOW:]
        data["rolling_average_reward"] = sum(recent)/len(recent) if recent else 0.0
        if data["best_reward"] is None or float(reward) > float(data["best_reward"]):
            data["best_reward"] = float(reward)
            with open(BEST_POLICY_FILE, "w", encoding="utf-8") as f: json.dump({"saved_at": time.time(), "best_reward": data["best_reward"]}, f, indent=2)
        with open(METRICS_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        return data
    except Exception:
        return {}


class FocusTrainer:
    """Identifies weak routes/levels and supplies a lightweight training focus bonus."""
    def __init__(self):
        self.route_attempts = {}
        self.route_successes = {}
        self.level_attempts = {}
        self.level_successes = {}

    def record_route(self, route_id, success):
        rid = str(route_id)
        self.route_attempts[rid] = self.route_attempts.get(rid, 0) + 1
        if success:
            self.route_successes[rid] = self.route_successes.get(rid, 0) + 1

    def record_level(self, level, success):
        lid = str(level)
        self.level_attempts[lid] = self.level_attempts.get(lid, 0) + 1
        if success:
            self.level_successes[lid] = self.level_successes.get(lid, 0) + 1

    def weak_routes(self):
        out = []
        for rid, attempts in self.route_attempts.items():
            if attempts >= 3:
                rate = self.route_successes.get(rid, 0) / attempts
                if rate < WEAK_ROUTE_THRESHOLD:
                    out.append((rid, rate, attempts))
        return sorted(out, key=lambda x: (x[1], -x[2]))

    def summary(self):
        return {
            "weak_routes": self.weak_routes()[:10],
            "levels": {
                lid: {
                    "attempts": a,
                    "success_rate": round(self.level_successes.get(lid, 0) / a, 3),
                }
                for lid, a in self.level_attempts.items()
            },
        }


def platform_candidates(state, route_memory=None):
    if not state:
        return []

    route_memory = route_memory or {}
    p = state["player"]
    px = p["x"] + 10
    py = p["y"]
    out = []

    plats = list(state.get("platforms", [])) + list(state.get("movingPlatforms", []))

    for platform in plats:
        left = platform["x"]
        right = left + platform["width"]
        center = (left + right) / 2

        # Ignore platforms clearly behind us or implausibly far away.
        if right < px - 45 or abs(center - px) > 600:
            continue

        dx = center - px
        dy = platform["y"] - py

        # Prefer forward platforms that are wide and reachable.
        forward = max(0.0, dx)
        backward = max(0.0, -dx)
        width_bonus = min(platform["width"], 260) * 0.45
        forward_bonus = min(forward, 420) * 0.28
        distance_penalty = abs(dx) * 0.18
        vertical_penalty = max(0.0, -dy - 170) * 1.7
        huge_jump_penalty = max(0.0, abs(dx) - 360) * 0.35
        backward_penalty = backward * 0.55

        success_bonus = route_success_rate(route_memory, platform) * 80.0
        attempts = route_memory.get(route_id(platform), {}).get("attempts", 0)
        exploration_bonus = 12.0 if attempts == 0 else 0.0

        score = (
            width_bonus
            + forward_bonus
            - distance_penalty
            - vertical_penalty
            - huge_jump_penalty
            - backward_penalty
            + success_bonus
            + exploration_bonus
        )

        out.append({
            "platform": platform,
            "score": score,
            "dx": dx,
            "dy": dy,
            "success_rate": route_success_rate(route_memory, platform),
        })

    out.sort(key=lambda z: z["score"], reverse=True)
    return out


def get_target(state, route_memory=None):
    candidates = platform_candidates(state, route_memory)
    if not candidates:
        return None

    p = state["player"]
    px = p["x"] + 10
    py = p["y"]

    # Never target the platform we are already standing on.
    usable = []
    for item in candidates:
        platform = item["platform"]
        same_floor = (
            platform["x"] <= px <= platform["x"] + platform["width"]
            and abs(platform["y"] - py) < 20
        )
        if not same_floor:
            usable.append(item)

    if not usable:
        usable = candidates

    return usable[0]["platform"]


def route_summary(state, route_memory=None):
    candidates = platform_candidates(state, route_memory)
    if not candidates:
        return "NO_PLATFORMS"

    return " ".join(
        f"P{i}:{z['platform']['x']:.0f}-{z['platform']['x'] + z['platform']['width']:.0f}"
        f"/S{z['score']:.0f}/R{z['success_rate']:.0%}"
        for i, z in enumerate(candidates[:4], 1)
    )


# ============================================================
# ACTION FILTERING
# ============================================================

def valid_actions(state):
    if not state:
        return ACTIONS

    player = state["player"]
    target = get_target(state)

    if not target:
        return ACTIONS

    target_center = (
        target["x"] +
        target["width"] / 2
    )

    player_center = (
        player["x"] + 10
    )

    distance = (
        target_center -
        player_center
    )

    actions = list(ACTIONS)

    # If target is strongly to the right,
    # discourage moving left.
    if distance > 100:
        actions = [
            "RIGHT",
            "JUMP_RIGHT",
            "WAIT",
        ]

    # If target is strongly to the left.
    elif distance < -100:
        actions = [
            "LEFT",
            "JUMP_LEFT",
            "WAIT",
        ]

    # If close to target, jumping straight
    # is generally more useful than huge movement.
    elif abs(distance) < 50:
        actions = [
            "JUMP_RIGHT",
            "JUMP_LEFT",
            "WAIT",
            "RIGHT",
            "LEFT",
        ]

    return actions


# ============================================================
# CHOOSE ACTION
# ============================================================

def landing_guidance(state):
    """Return a short-horizon steering action based on predicted landing X."""
    if not state:
        return None

    player = state["player"]
    target = get_target(state)

    if not target or player["grounded"]:
        return None

    predicted = predicted_landing_x(player)
    left = target["x"] + 12
    right = target["x"] + target["width"] - 12

    # Strong correction while airborne.
    if predicted < left:
        return "RIGHT"
    if predicted > right:
        return "LEFT"

    # Already predicted to land safely: release horizontal pressure.
    if abs(player["velocityX"]) > 2.5:
        return "LEFT" if player["velocityX"] > 0 else "RIGHT"

    return "WAIT"


def landing_error_value(state):
    if not state:
        return None
    player = state["player"]
    target = get_target(state)
    if not target:
        return None
    predicted = predicted_landing_x(player)
    center = target["x"] + target["width"] / 2
    return center - predicted


def choose_action(memory, state, epsilon, world_model=None):
    key = state_key(state)
    q = get_q_values(memory, key)
    allowed = valid_actions(state)

    guidance = landing_guidance(state)
    if guidance and guidance in allowed and random.random() < 0.70:
        return guidance, key

    # V31 model-based lookahead: estimate where each action will put the
    # player, then blend that prediction with the learned Q-value.
    if world_model is not None and allowed:
        scores = {a: world_model.action_score(state, a) for a in allowed}
        model_best = max(scores.values())
        candidates = [a for a in allowed if scores[a] >= model_best - 35.0]
        if random.random() > epsilon * 0.55:
            # Q remains important, but a strong predicted landing can override
            # a stale Q-value.
            blended = {a: q.get(a, 0.0) + 0.12 * scores[a] for a in candidates}
            best = max(blended.values())
            best_actions = [a for a in candidates if blended[a] == best]
            return random.choice(best_actions), key

    if random.random() < epsilon:
        return random.choice(allowed), key

    best_value = max(q.get(a, 0.0) for a in allowed)
    best_actions = [a for a in allowed if q.get(a, 0.0) == best_value]
    return random.choice(best_actions), key


# ============================================================
# EXECUTE ACTION
# ============================================================

async def execute_action(page, action, state=None, hold_ms=None):
    try:
        # Shorter corrections in the air prevent overshooting; grounded
        # movement gets a little more time to build momentum.
        action_time = ACTION_TIME_MS
        if state:
            player = state.get("player", {})
            if not player.get("grounded", False):
                action_time = 45
            else:
                action_time = 70
        if hold_ms is not None:
            try:
                action_time = max(V32_MIN_HOLD_MS, min(V32_MAX_HOLD_MS, int(hold_ms)))
            except Exception:
                pass

        if action == "RIGHT":

            await page.keyboard.down("d")

            await page.wait_for_timeout(
                action_time
            )

            await page.keyboard.up("d")

        elif action == "LEFT":

            await page.keyboard.down("a")

            await page.wait_for_timeout(
                action_time
            )

            await page.keyboard.up("a")

        elif action == "JUMP_RIGHT":

            await page.keyboard.down("d")

            await page.keyboard.press("w")

            await page.wait_for_timeout(
                action_time
            )

            await page.keyboard.up("d")

        elif action == "JUMP_LEFT":

            await page.keyboard.down("a")

            await page.keyboard.press("w")

            await page.wait_for_timeout(
                action_time
            )

            await page.keyboard.up("a")

        else:

            await page.wait_for_timeout(
                action_time
            )

        return True

    except Exception as e:

        print(
            f"\n⚠️ Action failed: {e}"
        )

        return False


# ============================================================
# LANDING DETECTION
# ============================================================

def landed_on_target(
    previous,
    current,
    route_memory=None
):
    """
    V53 robust landing detector.

    Uses:
        - current target first
        - horizontal overlap
        - grounded transition
        - vertical crossing
        - downward motion
        - tolerant platform-top distance

    This avoids relying on one exact browser frame.
    """

    if not previous or not current:
        return False

    try:
        old_player = previous.get("player", {})
        new_player = current.get("player", {})

        if not old_player or not new_player:
            return False

        # Prefer the CURRENT target because the CURRENT player
        # position is what we are evaluating.
        target = get_target(
            current,
            route_memory
        )

        if not target:
            target = get_target(
                previous,
                route_memory
            )

        if not target:
            return False

        # --------------------------------------------------------
        # PLAYER GEOMETRY
        # --------------------------------------------------------

        PLAYER_WIDTH = 20.0
        PLAYER_HEIGHT = 50.0

        old_x = float(old_player.get("x", 0.0))
        old_y = float(old_player.get("y", 0.0))

        new_x = float(new_player.get("x", 0.0))
        new_y = float(new_player.get("y", 0.0))

        old_grounded = bool(
            old_player.get("grounded", False)
        )

        new_grounded = bool(
            new_player.get("grounded", False)
        )

        old_vy = float(
            old_player.get("velocityY", 0.0)
        )

        new_vy = float(
            new_player.get("velocityY", 0.0)
        )

        target_x = float(
            target.get("x", 0.0)
        )

        target_y = float(
            target.get("y", 0.0)
        )

        target_width = float(
            target.get("width", 0.0)
        )

        if target_width <= 0:
            return False

        # --------------------------------------------------------
        # HORIZONTAL OVERLAP
        # --------------------------------------------------------

        target_left = target_x
        target_right = target_x + target_width

        player_left = new_x
        player_right = new_x + PLAYER_WIDTH

        horizontal_overlap = (
            player_right >= target_left
            and
            player_left <= target_right
        )

        if not horizontal_overlap:
            return False

        # --------------------------------------------------------
        # VERTICAL GEOMETRY
        # --------------------------------------------------------

        old_bottom = old_y + PLAYER_HEIGHT
        new_bottom = new_y + PLAYER_HEIGHT

        vertical_error = abs(
            new_bottom - target_y
        )

        # --------------------------------------------------------
        # GROUNDED TRANSITION
        # --------------------------------------------------------

        grounded_transition = (
            not old_grounded
            and
            new_grounded
        )

        # --------------------------------------------------------
        # PLATFORM TOP CROSSING
        #
        # Handles cases where Playwright/browser observation
        # skips the exact airborne -> grounded frame.
        # --------------------------------------------------------

        crossed_platform_top = (
            old_bottom <= target_y + 8.0
            and
            new_bottom >= target_y - 8.0
        )

        # --------------------------------------------------------
        # FALLING / DOWNWARD MOTION
        # --------------------------------------------------------

        falling = (
            new_vy >= -1.0
            or
            old_vy > 0.0
        )

        # --------------------------------------------------------
        # PLATFORM HEIGHT TOLERANCE
        # --------------------------------------------------------

        near_platform_top = (
            vertical_error <= 18.0
        )

        # --------------------------------------------------------
        # STRONG LANDING SIGNAL
        # --------------------------------------------------------

        if (
            grounded_transition
            and
            near_platform_top
        ):
            return True

        # --------------------------------------------------------
        # FALLBACK LANDING SIGNAL
        #
        # Browser may have skipped the grounded frame.
        # --------------------------------------------------------

        if (
            crossed_platform_top
            and
            falling
            and
            near_platform_top
        ):
            return True

        return False

    except (
        TypeError,
        ValueError,
        KeyError
    ):
        # Landing detection must NEVER crash the worker.
        return False


# ============================================================
# REWARD
# ============================================================

def calculate_reward(
    previous,
    current,
    route_memory=None
):
    if not previous or not current:
        return 0.0, []

    reward = -0.05
    events = []

    old_player = previous["player"]
    new_player = current["player"]

    old_x = old_player["x"]
    new_x = new_player["x"]

    target = get_target(
        previous,
        route_memory
    )

    # --------------------------------------------------------
    # Progress toward target
    # --------------------------------------------------------

    if target:

        target_center = (
            target["x"] +
            target["width"] / 2
        )

        old_center = (
            old_x + 10
        )

        new_center = (
            new_x + 10
        )

        old_distance = abs(
            target_center -
            old_center
        )

        new_distance = abs(
            target_center -
            new_center
        )

        progress = (
            old_distance -
            new_distance
        )

        progress = max(
            -5,
            min(5, progress)
        )

        reward += progress * 0.30

        # V7: reward getting the predicted landing point closer to the
        # usable target zone. This teaches braking, not just movement.
        old_pred = predicted_landing_x(old_player)
        new_pred = predicted_landing_x(new_player)
        target_left = target["x"] + 12
        target_right = target["x"] + target["width"] - 12
        old_error = 0.0 if target_left <= old_pred <= target_right else min(300.0, min(abs(old_pred-target_left), abs(old_pred-target_right)))
        new_error = 0.0 if target_left <= new_pred <= target_right else min(300.0, min(abs(new_pred-target_left), abs(new_pred-target_right)))
        landing_improvement = max(-8.0, min(8.0, old_error - new_error))
        reward += landing_improvement * 0.12

        if progress > 1:
            events.append(
                f"PROGRESS +{progress * 0.15:.2f}"
            )

    # --------------------------------------------------------
    # Landing
    # --------------------------------------------------------

    if landed_on_target(
        previous,
        current,
        route_memory
    ):

        reward += LANDING_REWARD

        events.append(
            f"LANDING +{LANDING_REWARD:.0f}"
        )

    # --------------------------------------------------------
    # Level progress
    # --------------------------------------------------------

    if (
        current["currentLevel"]
        >
        previous["currentLevel"]
    ):

        reward += LEVEL_REWARD + GOAL_REWARD

        events.append(
            f"LEVEL UP +{LEVEL_REWARD:.0f}"
        )
        events.append(
            f"GOAL +{GOAL_REWARD:.0f}"
        )

    # --------------------------------------------------------
    # Score increase
    # --------------------------------------------------------

    score_difference = (
        current["score"] -
        previous["score"]
    )

    if score_difference > 0:

        reward += min(
            20,
            score_difference * 0.5
        )

        events.append(
            f"SCORE +{score_difference * 0.5:.1f}"
        )

    # --------------------------------------------------------
    # Death
    # --------------------------------------------------------

    if (
        current["deaths"] >
        previous["deaths"]
    ):

        reward += DEATH_PENALTY

        events.append(
            f"DEATH {DEATH_PENALTY:.0f}"
        )

    return reward, events


# ============================================================
# Q LEARNING UPDATE
# ============================================================

def learn(
    memory,
    old_key,
    action,
    reward,
    new_state,
    done=False
):

    old_q = get_q_values(
        memory,
        old_key
    )

    current_value = (
        old_q.get(
            action,
            0.0
        )
    )

    if done:

        target_value = reward

    else:

        new_key = state_key(
            new_state
        )

        new_q = get_q_values(
            memory,
            new_key
        )

        future_value = max(
            new_q.values()
        )

        target_value = (
            reward +
            GAMMA *
            future_value
        )

    old_q[action] = (
        current_value +
        ALPHA *
        (
            target_value -
            current_value
        )
    )


# ============================================================
# TRAINING
# ============================================================

def double_q_choose(memory_a, memory_b, state, epsilon, evaluation=False):
    key = state_key(state)
    qa = get_q_values(memory_a, key)
    qb = get_q_values(memory_b, key)
    allowed = valid_actions(state)
    if not allowed:
        allowed = [WAIT]

    # In evaluation mode, no random exploration and no stochastic guidance.
    if not evaluation:
        guidance = landing_guidance(state)
        if guidance and guidance in allowed and random.random() < 0.78:
            return guidance, key
        if random.random() < epsilon:
            return random.choice(allowed), key

    values = {a: 0.5 * (qa.get(a, 0.0) + qb.get(a, 0.0)) for a in allowed}
    best = max(values.values())
    best_actions = [a for a, v in values.items() if v == best]
    return (best_actions[0] if evaluation else random.choice(best_actions)), key


def double_q_learn(memory_a, memory_b, old_key, action, reward, new_state, done=False, update_a=True):
    old_table = memory_a if update_a else memory_b
    other_table = memory_b if update_a else memory_a
    old_q = get_q_values(old_table, old_key)
    current = old_q.get(action, 0.0)

    if done:
        target = reward
    else:
        new_key = state_key(new_state)
        online_q = get_q_values(old_table, new_key)
        other_q = get_q_values(other_table, new_key)
        allowed = valid_actions(new_state) or [WAIT]
        best_action = max(allowed, key=lambda a: online_q.get(a, 0.0))
        target = reward + GAMMA * other_q.get(best_action, 0.0)

    old_q[action] = current + ALPHA * (target - current)
    return abs(target - current)


class FailureAnalyzer:
    """Turns failed transitions into explicit training objectives."""
    def __init__(self):
        self.stats = {}
        self.focus = None
        self.load()

    def classify(self, state, next_state, events):
        p = state.get("player", {})
        np = next_state.get("player", {})
        if next_state.get("deaths", 0) > state.get("deaths", 0):
            target = get_target(state, None)
            if target:
                center = target["x"] + target["width"] / 2
                error = predicted_landing_x(p) - center
                if abs(error) > 120:
                    return "overshoot" if error > 0 else "undershoot"
                if p.get("grounded") and p.get("velocityY", 0) == 0:
                    return "jump_timing"
                if abs(p.get("velocityX", 0)) < 1.0:
                    return "insufficient_movement"
            return "route_failure"
        if "LANDING" in " ".join(events):
            return "successful_landing"
        return None

    def record(self, kind, success=False):
        if not kind or kind == "successful_landing":
            return
        row = self.stats.setdefault(kind, {"attempts": 0, "successes": 0})
        row["attempts"] += 1
        row["successes"] += int(bool(success))
        self.recompute()

    def recompute(self):
        if not self.stats:
            self.focus = None
            return
        def priority(item):
            _, row = item
            attempts = row.get("attempts", 0)
            rate = row.get("successes", 0) / max(1, attempts)
            return attempts * (1.0 - rate)
        self.focus = max(self.stats.items(), key=priority)[0]

    def priority(self, kind):
        row = self.stats.get(kind, {})
        attempts = row.get("attempts", 0)
        rate = row.get("successes", 0) / max(1, attempts)
        return min(3.0, 1.0 + (1.0 - rate) * FAILURE_FOCUS_BONUS * min(attempts, 8))

    def save(self):
        try:
            with open(FAILURE_FILE, "w", encoding="utf-8") as f:
                json.dump({"stats": self.stats, "focus": self.focus, "saved_at": time.time()}, f, indent=2)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(FAILURE_FILE):
                with open(FAILURE_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.stats = d.get("stats", {})
                self.focus = d.get("focus")
        except Exception:
            self.stats = {}

    def summary(self):
        return {"focus": self.focus, "stats": self.stats}


class FailureReplayLab:
    """Persist failed trajectories and retain a short practice window around failure."""
    def __init__(self):
        self.trajectories = []
        self.load()

    def record(self, worker_id, trajectory, reason="failure"):
        if not trajectory:
            return
        self.trajectories.append({
            "worker": worker_id, "reason": reason, "timestamp": time.time(),
            "length": len(trajectory), "steps": trajectory[-REPLAY_WINDOW:]
        })
        self.trajectories = self.trajectories[-MAX_TRAJECTORIES:]

    def hardest(self, limit=5):
        return self.trajectories[-limit:]

    def save(self):
        try:
            os.makedirs(os.path.dirname(TRAJECTORY_FILE), exist_ok=True)
            tmp = TRAJECTORY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.trajectories, f, indent=2)
            os.replace(tmp, TRAJECTORY_FILE)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(TRAJECTORY_FILE):
                with open(TRAJECTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.trajectories = data if isinstance(data, list) else []
        except Exception:
            self.trajectories = []

    def summary(self):
        return {"trajectories": len(self.trajectories), "window": REPLAY_WINDOW, "recent": self.hardest()}



class HierarchicalController:
    """Two-level controller: strategic subgoal + low-level V31 planner.
    Accepts the game's full state dict (player/platforms/etc.) as used by workers.
    """
    OPTIONS = ["REACH_TARGET", "CENTER_ON_PLATFORM", "BRAKE", "RECOVER", "GOAL_APPROACH"]

    def __init__(self, planner, route_memory):
        self.planner = planner
        self.route_memory = route_memory
        self.current = "REACH_TARGET"
        self.target = None
        self.last_update_step = -1
        self.decisions = 0
        self.option_stats = {k: {"uses": 0, "success": 0, "reward": 0.0} for k in self.OPTIONS}

    @staticmethod
    def _kinematics(state):
        """Normalize the game's state dict into x,y,vx,vy,grounded."""
        if isinstance(state, dict):
            p = state.get("player") or {}
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("velocityX", p.get("vx", 0.0))),
                float(p.get("velocityY", p.get("vy", 0.0))),
                bool(p.get("grounded", False)),
            )
        if isinstance(state, (list, tuple)):
            vals = list(state)
            vals += [0.0] * max(0, 5 - len(vals))
            return tuple(vals[:5])
        return 0.0, 0.0, 0.0, 0.0, False

    @staticmethod
    def _target_center(target, fallback_x):
        if not isinstance(target, dict):
            return float(fallback_x)
        return float(target.get("x", fallback_x)) + float(target.get("width", 0.0)) / 2.0

    def choose_option(self, state, target, events, step, evaluation=False):
        x, y, vx, vy, grounded = self._kinematics(state)
        changed = target != self.target
        landing = any("land" in str(e).lower() for e in events)
        levelup = any(
            "level" in str(e).lower() or "goal" in str(e).lower()
            for e in events
        )

        if changed or landing or levelup or step - self.last_update_step >= 18:
            self.target = target
            self.last_update_step = step

            if not target:
                self.current = "RECOVER"
            elif levelup or bool(target.get("is_goal")):
                self.current = "GOAL_APPROACH"
            else:
                dx = self._target_center(target, x) - (x + 10.0)
                if abs(dx) < 35:
                    self.current = "CENTER_ON_PLATFORM"
                elif vx != 0 and ((dx > 0 and vx > 3.5) or (dx < 0 and vx < -3.5)):
                    self.current = "BRAKE"
                else:
                    self.current = "REACH_TARGET"
            self.decisions += 1

        return self.current

    def action(self, state, target, option, evaluation=False):
        x, y, vx, vy, grounded = self._kinematics(state)

        if option == "RECOVER":
            return "A" if vx > 0.5 else ("D" if vx < -0.5 else "W")

        if not target:
            return "W"

        tx = self._target_center(target, x)
        dx = tx - (x + 10.0)

        if option == "BRAKE":
            return "A" if vx > 0.25 else ("D" if vx < -0.25 else "W")

        if option == "CENTER_ON_PLATFORM":
            if dx > 12:
                return "D"
            if dx < -12:
                return "A"
            return "W"

        if option == "GOAL_APPROACH":
            if dx > 10:
                return "D"
            if dx < -10:
                return "A"
            return "W"

        return None

    def record(self, option, reward, success=False):
        if option not in self.option_stats:
            return
        d = self.option_stats[option]
        d["uses"] += 1
        d["reward"] += float(reward)
        if success:
            d["success"] += 1



# ============================================================
# V34 - META LEARNING / STRATEGY ADAPTATION
# ============================================================

V34_META_FILE = os.path.join(DATA_DIR, "meta_learning_v34.json")


class MetaLearningManager:
    """
    V34 layer: learns which strategy characteristics perform best.

    Each worker is assigned a strategy profile.  After every completed
    episode we score that profile, keep strong profiles, mutate weak ones,
    and preserve a small amount of exploration.
    """

    PARAMS = (
        "jump_aggression",
        "braking_strength",
        "target_centering",
        "exploration",
        "risk_tolerance",
        "action_hold_bias",
        "recovery_preference",
    )

    def __init__(self, population_size=8):
        self.population_size = int(population_size)
        self.generation = 0
        self.total_evaluations = 0
        self.best_score = -1e18
        self.best_strategy_id = None
        self.worker_strategy = {}
        self.population = []
        self.last_evolution = None
        self.load()
        if not self.population:
            self._seed_population()

    @staticmethod
    def _clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(v)))

    def _new_strategy(self, sid, parent=None):
        if parent:
            genes = dict(parent.get("genes", {}))
            for name in self.PARAMS:
                genes.setdefault(name, 0.5)
        else:
            genes = {name: random.uniform(0.25, 0.75) for name in self.PARAMS}

        return {
            "id": sid,
            "parent": parent.get("id") if parent else None,
            "genes": {k: self._clamp(v) for k, v in genes.items()},
            "episodes": 0,
            "reward_total": 0.0,
            "best_reward": -1e18,
            "successes": 0,
            "failures": 0,
            "score": 0.0,
        }

    def _seed_population(self):
        presets = [
            {"jump_aggression": .75, "braking_strength": .35, "target_centering": .65, "exploration": .60, "risk_tolerance": .70, "action_hold_bias": .60, "recovery_preference": .35},
            {"jump_aggression": .45, "braking_strength": .75, "target_centering": .80, "exploration": .30, "risk_tolerance": .25, "action_hold_bias": .45, "recovery_preference": .75},
            {"jump_aggression": .60, "braking_strength": .60, "target_centering": .70, "exploration": .45, "risk_tolerance": .50, "action_hold_bias": .50, "recovery_preference": .60},
            {"jump_aggression": .80, "braking_strength": .25, "target_centering": .45, "exploration": .80, "risk_tolerance": .80, "action_hold_bias": .70, "recovery_preference": .30},
            {"jump_aggression": .35, "braking_strength": .80, "target_centering": .85, "exploration": .20, "risk_tolerance": .20, "action_hold_bias": .40, "recovery_preference": .85},
            {"jump_aggression": .65, "braking_strength": .50, "target_centering": .60, "exploration": .65, "risk_tolerance": .60, "action_hold_bias": .55, "recovery_preference": .45},
            {"jump_aggression": .50, "braking_strength": .55, "target_centering": .75, "exploration": .50, "risk_tolerance": .40, "action_hold_bias": .50, "recovery_preference": .70},
            {"jump_aggression": .70, "braking_strength": .45, "target_centering": .55, "exploration": .70, "risk_tolerance": .65, "action_hold_bias": .65, "recovery_preference": .40},
        ]
        self.population = []
        for i in range(self.population_size):
            s = self._new_strategy(f"meta-{self.generation}-{i}")
            if i < len(presets):
                s["genes"].update(presets[i])
            self.population.append(s)

    def assign_worker(self, worker_id):
        wid = str(worker_id)
        if wid in self.worker_strategy:
            return self.worker_strategy[wid]
        # Spread workers over the current population.
        index = len(self.worker_strategy) % max(1, len(self.population))
        sid = self.population[index]["id"]
        self.worker_strategy[wid] = sid
        return sid

    def get_strategy(self, strategy_id):
        for s in self.population:
            if s["id"] == strategy_id:
                return s
        return self.population[0] if self.population else None

    def genes(self, worker_id):
        sid = self.worker_strategy.get(str(worker_id))
        strategy = self.get_strategy(sid)
        return dict(strategy.get("genes", {})) if strategy else {}

    def action_score(self, worker_id, action, state):
        """
        Convert learned strategy genes into a small action preference.
        This does not replace V33's Q/planner/tactical logic; it gently
        steers the final decision.
        """
        g = self.genes(worker_id)
        if not g:
            return 0.0

        p = state.get("player", {})
        vx = float(p.get("velocityX", 0.0))
        grounded = bool(p.get("grounded", False))
        score = 0.0

        if action in ("JUMP_RIGHT", "JUMP_LEFT"):
            score += 0.55 * g.get("jump_aggression", .5)
            if grounded:
                score += 0.20 * g.get("jump_aggression", .5)

        if action in ("LEFT", "RIGHT"):
            score += 0.35 * g.get("braking_strength", .5)
            if abs(vx) > 3:
                score += 0.25 * g.get("braking_strength", .5)

        if action == "NONE":
            score += 0.20 * g.get("recovery_preference", .5)

        # Exploration is intentionally tiny here; V33 epsilon remains primary.
        score += random.uniform(-0.04, 0.04) * g.get("exploration", .5)
        return score

    def record_episode(self, worker_id, reward, success=False):
        wid = str(worker_id)
        sid = self.assign_worker(wid)
        strategy = self.get_strategy(sid)
        if not strategy:
            return False

        reward = float(reward)
        strategy["episodes"] += 1
        strategy["reward_total"] += reward
        strategy["best_reward"] = max(strategy["best_reward"], reward)
        strategy["successes"] += int(bool(success))
        strategy["failures"] += int(not success)

        # Smoothed fitness: reward + success bonus, normalized by experience.
        avg = strategy["reward_total"] / max(1, strategy["episodes"])
        success_rate = strategy["successes"] / max(1, strategy["episodes"])
        strategy["score"] = avg + 25.0 * success_rate

        self.total_evaluations += 1
        if strategy["best_reward"] > self.best_score:
            self.best_score = strategy["best_reward"]
            self.best_strategy_id = strategy["id"]

        # Evolve after every complete population cycle.
        return self.total_evaluations % max(1, self.population_size) == 0

    def evolve(self):
        if not self.population:
            self._seed_population()
            return []

        ranked = sorted(self.population, key=lambda s: s.get("score", -1e18), reverse=True)
        elite_count = max(2, min(3, len(ranked)))
        elites = ranked[:elite_count]

        new_population = []
        for elite in elites:
            clone = json.loads(json.dumps(elite))
            clone["parent"] = elite["id"]
            clone["id"] = f"meta-{self.generation + 1}-{len(new_population)}"
            new_population.append(clone)

        while len(new_population) < self.population_size:
            parent = random.choice(elites)
            child = self._new_strategy(
                f"meta-{self.generation + 1}-{len(new_population)}",
                parent=parent,
            )
            # Stronger mutations early, smaller mutations as generations grow.
            mutation = max(0.04, 0.18 * (0.97 ** self.generation))
            for name in self.PARAMS:
                if random.random() < 0.75:
                    child["genes"][name] = self._clamp(
                        child["genes"][name] + random.gauss(0.0, mutation)
                    )
            new_population.append(child)

        self.generation += 1
        self.population = new_population

        # Reassign workers so they sample the new generation.
        for i, wid in enumerate(list(self.worker_strategy)):
            self.worker_strategy[wid] = self.population[i % len(self.population)]["id"]

        self.last_evolution = {
            "generation": self.generation,
            "elites": [x["id"] for x in elites],
            "best_score": max(x.get("score", -1e18) for x in elites),
        }
        return elites

    def summary(self):
        ranked = sorted(
            self.population,
            key=lambda s: s.get("score", -1e18),
            reverse=True,
        )
        return {
            "generation": self.generation,
            "population": len(self.population),
            "evaluations": self.total_evaluations,
            "best_score": self.best_score,
            "best_strategy_id": self.best_strategy_id,
            "last_evolution": self.last_evolution,
            "strategies": [
                {
                    "id": s["id"],
                    "score": round(s.get("score", 0.0), 3),
                    "episodes": s.get("episodes", 0),
                    "best_reward": s.get("best_reward", -1e18),
                    "successes": s.get("successes", 0),
                    "genes": {k: round(v, 3) for k, v in s.get("genes", {}).items()},
                }
                for s in ranked[:self.population_size]
            ],
        }

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = {
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "best_score": self.best_score,
            "best_strategy_id": self.best_strategy_id,
            "worker_strategy": self.worker_strategy,
            "population": self.population,
            "last_evolution": self.last_evolution,
        }
        try:
            with open(V34_META_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"⚠️ V34 meta save failed: {e}")

    def load(self):
        try:
            if not os.path.exists(V34_META_FILE):
                return
            with open(V34_META_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.generation = int(payload.get("generation", 0))
            self.total_evaluations = int(payload.get("total_evaluations", 0))
            self.best_score = float(payload.get("best_score", -1e18))
            self.best_strategy_id = payload.get("best_strategy_id")
            self.worker_strategy = {
                str(k): v for k, v in payload.get("worker_strategy", {}).items()
            }
            self.population = payload.get("population", [])
            self.last_evolution = payload.get("last_evolution")
            print(
                f"🧬 V34 MetaLearning loaded | generation={self.generation} "
                f"| strategies={len(self.population)}"
            )
        except Exception as e:
            print(f"⚠️ V34 meta load failed: {e}")
            self.population = []

# ============================================================
# V35 - AUTONOMOUS CURRICULUM GENERATION
# ============================================================

V35_CURRICULUM_FILE = os.path.join(DATA_DIR, "autonomous_curriculum_v35.json")


class AutonomousCurriculumV35:
    """
    V35 lets the agent choose what it should practice next.

    It turns recent failures/successes into training objectives, scores
    candidate objectives, and increases practice pressure on the weakest
    objective while still allowing exploration.
    """

    OBJECTIVES = (
        "jump_timing",
        "overshoot_control",
        "undershoot_control",
        "hazard_avoidance",
        "precision_landing",
        "route_completion",
        "goal_approach",
        "exploration",
    )

    def __init__(self):
        self.objectives = {
            name: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "reward_total": 0.0,
                "priority": 1.0,
                "last_reason": "initial",
            }
            for name in self.OBJECTIVES
        }
        self.focus = "exploration"
        self.focus_reason = "initial exploration"
        self.total_episodes = 0
        self.last_episode_reward = 0.0
        self.last_update = None
        self.load()
        self._recalculate()

    @staticmethod
    def _clamp(v, lo=0.25, hi=5.0):
        return max(lo, min(hi, float(v)))

    def _touch(self, objective, success=None, reward=0.0, reason=None):
        if objective not in self.objectives:
            return
        d = self.objectives[objective]
        d["attempts"] += 1
        d["reward_total"] += float(reward)
        if success is True:
            d["successes"] += 1
        elif success is False:
            d["failures"] += 1
        if reason:
            d["last_reason"] = reason

    def record_event(self, failure_kind=None, events=None, success=False, reward=0.0):
        events = events or []
        event_text = " ".join(str(x) for x in events)

        if failure_kind:
            mapping = {
                "jump_timing": "jump_timing",
                "overshoot": "overshoot_control",
                "undershoot": "undershoot_control",
                "insufficient_movement": "route_completion",
                "route_failure": "route_completion",
            }
            self._touch(
                mapping.get(failure_kind, "route_completion"),
                success=False,
                reward=reward,
                reason=f"failure:{failure_kind}",
            )

        if "LANDING" in event_text:
            self._touch("precision_landing", success=True, reward=reward, reason="landing")
        if "DEATH" in event_text or "SPIKE" in event_text:
            self._touch("hazard_avoidance", success=False, reward=reward, reason="hazard")
        if "GOAL" in event_text or "LEVEL_UP" in event_text or success:
            self._touch("goal_approach", success=True, reward=reward, reason="goal progress")

        if not failure_kind and not ("LANDING" in event_text):
            self._touch("exploration", success=success, reward=reward, reason="normal exploration")

        self._recalculate()

    def _recalculate(self):
        for name, d in self.objectives.items():
            attempts = max(1, d["attempts"])
            failure_rate = d["failures"] / attempts
            success_rate = d["successes"] / attempts
            avg_reward = d["reward_total"] / attempts

            # High failure + low success + poor reward = higher training need.
            need = 1.0 + 2.2 * failure_rate + 0.8 * (1.0 - success_rate)
            if avg_reward < 0:
                need += min(1.0, abs(avg_reward) / 100.0)
            # Recently untouched skills get a small exploration bonus.
            if d["attempts"] == 0:
                need += 0.35
            d["priority"] = round(self._clamp(need), 4)

        ranked = sorted(
            self.objectives.items(),
            key=lambda kv: kv[1]["priority"],
            reverse=True,
        )
        if ranked:
            self.focus = ranked[0][0]
            self.focus_reason = ranked[0][1]["last_reason"]

    def begin_episode(self):
        self.total_episodes += 1
        self._recalculate()
        return self.focus

    def record_episode(self, reward, success=False):
        self.last_episode_reward = float(reward)
        self._touch(
            self.focus,
            success=bool(success),
            reward=float(reward),
            reason=f"episode:{'success' if success else 'failure'}",
        )
        self._recalculate()
        self.last_update = time.time()

    def action_bias(self, action, state):
        """
        Gentle objective-specific steering. The existing V34/V33 policy
        remains the main controller.
        """
        focus = self.focus
        p = state.get("player", {})
        grounded = bool(p.get("grounded", False))
        vx = float(p.get("velocityX", 0.0))
        score = 0.0

        if focus == "jump_timing" and grounded and action in ("JUMP_RIGHT", "JUMP_LEFT"):
            score += 0.75
        elif focus == "overshoot_control" and action == "LEFT" and abs(vx) > 2.5:
            score += 0.80
        elif focus == "undershoot_control" and action == "RIGHT" and abs(vx) < 3.0:
            score += 0.65
        elif focus == "hazard_avoidance" and action == "NONE":
            score += 0.35
        elif focus == "precision_landing" and action in ("LEFT", "RIGHT"):
            score += 0.30
        elif focus == "route_completion" and action in ("RIGHT", "LEFT"):
            score += 0.25
        elif focus == "goal_approach" and action in ("RIGHT", "LEFT"):
            score += 0.30
        elif focus == "exploration":
            score += random.uniform(0.0, 0.20)

        priority = self.objectives.get(focus, {}).get("priority", 1.0)
        return score * min(1.5, priority / 2.0)

    def directive(self):
        priority = self.objectives.get(self.focus, {}).get("priority", 1.0)
        return {
            "focus": self.focus,
            "reason": self.focus_reason,
            "priority": round(priority, 3),
            "pressure": round(min(0.75, 0.15 + priority * 0.10), 3),
            "episodes": self.total_episodes,
        }

    def summary(self):
        ranked = sorted(
            self.objectives.items(),
            key=lambda kv: kv[1]["priority"],
            reverse=True,
        )
        return {
            "focus": self.focus,
            "reason": self.focus_reason,
            "directive": self.directive(),
            "total_episodes": self.total_episodes,
            "last_episode_reward": self.last_episode_reward,
            "objectives": [
                {
                    "name": name,
                    "priority": d["priority"],
                    "attempts": d["attempts"],
                    "successes": d["successes"],
                    "failures": d["failures"],
                    "success_rate": round(
                        d["successes"] / max(1, d["attempts"]), 3
                    ),
                    "last_reason": d["last_reason"],
                }
                for name, d in ranked
            ],
        }

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = {
            "objectives": self.objectives,
            "focus": self.focus,
            "focus_reason": self.focus_reason,
            "total_episodes": self.total_episodes,
            "last_episode_reward": self.last_episode_reward,
            "last_update": self.last_update,
        }
        try:
            with open(V35_CURRICULUM_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"⚠️ V35 curriculum save failed: {e}")

    def load(self):
        try:
            if not os.path.exists(V35_CURRICULUM_FILE):
                return
            with open(V35_CURRICULUM_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            saved = payload.get("objectives", {})
            for name in self.OBJECTIVES:
                if name in saved:
                    self.objectives[name].update(saved[name])
            self.focus = payload.get("focus", self.focus)
            self.focus_reason = payload.get("focus_reason", self.focus_reason)
            self.total_episodes = int(payload.get("total_episodes", 0))
            self.last_episode_reward = float(payload.get("last_episode_reward", 0.0))
            self.last_update = payload.get("last_update")
            print(
                f"🎓 V35 Curriculum loaded | focus={self.focus} "
                f"| episodes={self.total_episodes}"
            )
        except Exception as e:
            print(f"⚠️ V35 curriculum load failed: {e}")


# ============================================================
# V36 SELF-EVALUATION + V37 RESILIENCE + V52 PRIVACY
# ============================================================

V36_SELF_EVAL_FILE = os.path.join(DATA_DIR, "self_evaluation_v36.json")
V37_RESILIENCE_FILE = os.path.join(DATA_DIR, "resilience_brain_v37.json")

class PrivacySanitizerV52:
    """Remove identifying/network/device metadata from telemetry before persistence."""
    SENSITIVE_KEYS={"ip","ip_address","public_ip","local_ip","remote_ip","hostname","host","machine","machine_name","computer_name","username","user","home","home_dir","cwd","path","file_path","gps","latitude","longitude","location","device_id","fingerprint","mac","mac_address","cookies","user_agent"}
    PRIVATE_IP_RE=re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|127\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")
    def __init__(self): self.removed_fields=0; self.scrubbed_values=0
    def sanitize(self,obj):
        if isinstance(obj,dict):
            r={}
            for k,v in obj.items():
                n=str(k).lower().replace('-','_')
                if n in self.SENSITIVE_KEYS or n.endswith('_ip') or any(x in n for x in ('gps','latitude','longitude')):
                    self.removed_fields+=1; continue
                r[k]=self.sanitize(v)
            return r
        if isinstance(obj,list): return [self.sanitize(x) for x in obj]
        if isinstance(obj,str):
            old=obj; obj=self.PRIVATE_IP_RE.sub('[REDACTED_IP]',obj)
            obj=re.sub(r'(?i)(?:[A-Z]:\\|/home/|/Users/|/mnt/data/)[^\s"\']+','[REDACTED_PATH]',obj)
            if obj!=old: self.scrubbed_values+=1
            return obj
        return obj
    def pseudonymous_worker_id(self,raw_id):
        return 'worker_'+hashlib.sha256(str(raw_id).encode()).hexdigest()[:12]
    def summary(self): return {'removed_fields':self.removed_fields,'scrubbed_values':self.scrubbed_values}

class ResilienceBrainV37:
    """Bounded retry/backoff memory for transient browser/network failures."""
    def __init__(self): self.memory=self._load(); self.max_attempts=5; self.base_delay=.8; self.max_delay=12.0
    def _load(self):
        try:
            if os.path.exists(V37_RESILIENCE_FILE):
                with open(V37_RESILIENCE_FILE,encoding='utf-8') as f: return json.load(f)
        except Exception: pass
        return {'version':'V37','recoveries':{},'total_attempts':0,'successful_recoveries':0,'failed_recoveries':0}
    def fingerprint(self,problem,worker_id=None): return hashlib.sha256(f'{problem}|{worker_id or "unknown"}'.encode()).hexdigest()[:16]
    def begin(self,problem,worker_id=None):
        fp=self.fingerprint(problem,worker_id); item=self.memory['recoveries'].setdefault(fp,{'problem':str(problem),'attempts':0,'successes':0,'failures':0,'best_attempt':None}); item['attempts']+=1; self.memory['total_attempts']+=1; a=item['attempts']; self.save()
        if a>self.max_attempts: return {'retry':False,'attempt':a,'reason':'bounded_retry_limit','fingerprint':fp}
        delay=min(self.max_delay,self.base_delay*(2**(a-1)))*random.uniform(.85,1.15)
        return {'retry':True,'attempt':a,'delay':round(delay,2),'fingerprint':fp}
    def record(self,fp,success,elapsed=0):
        item=self.memory['recoveries'].setdefault(fp,{'problem':'unknown','attempts':0,'successes':0,'failures':0,'best_attempt':None})
        if success:
            item['successes']+=1; self.memory['successful_recoveries']+=1; b=item.get('best_attempt'); item['best_attempt']=item['attempts'] if b is None else min(b,item['attempts'])
        else: item['failures']+=1; self.memory['failed_recoveries']+=1
        item['last_elapsed']=round(float(elapsed),3); self.save()
    def save(self):
        try:
            os.makedirs(DATA_DIR,exist_ok=True)
            with open(V37_RESILIENCE_FILE,'w',encoding='utf-8') as f: json.dump(self.memory,f,indent=2)
        except Exception: pass
    def summary(self): return {'version':'V37','problem_types':len(self.memory.get('recoveries',{})),'total_attempts':self.memory.get('total_attempts',0),'successful_recoveries':self.memory.get('successful_recoveries',0),'failed_recoveries':self.memory.get('failed_recoveries',0)}

class SelfEvaluationV36:
    """Track improvement, regression and stagnation and emit training directives."""
    def __init__(self): self.state=self._load()
    def _load(self):
        try:
            if os.path.exists(V36_SELF_EVAL_FILE):
                with open(V36_SELF_EVAL_FILE,encoding='utf-8') as f: return json.load(f)
        except Exception: pass
        return {'version':'V36','focus':'jump_timing','difficulty':1,'episodes':0,'recent_rewards':[],'best_reward':None,'improvements':0,'regressions':0,'stagnation':0,'last_decision':'initial_training'}
    def record(self,reward):
        reward=float(reward); recent=self.state.setdefault('recent_rewards',[]); prev=recent[-10:]; avg=sum(prev)/len(prev) if prev else None; recent.append(round(reward,4)); del recent[:-30]; self.state['episodes']+=1
        best=self.state.get('best_reward')
        if best is None or reward>best: self.state['best_reward']=reward; self.state['improvements']+=1; self.state['stagnation']=0
        elif avg is not None and reward<avg: self.state['regressions']+=1; self.state['stagnation']+=1
        else: self.state['stagnation']+=1
        if self.state['stagnation']>=15: d='change_strategy'
        elif self.state['regressions']>=5: d='practice_regression'
        elif self.state['improvements']>=5: d='consider_higher_difficulty'
        else: d='continue_training'
        self.state['last_decision']=d; self.save(); return self.directive()
    def directive(self):
        d=self.state.get('last_decision','initial_training'); msgs={'initial_training':'Practice the current weakest skill.','change_strategy':'Stagnation detected. Change strategy and increase observation.','practice_regression':'Regression detected. Practice the weak behavior before advancing.','consider_higher_difficulty':'Repeated improvement detected. Consider a small difficulty increase.','continue_training':'Continue training the current objective.'}; return {'focus':self.state.get('focus','jump_timing'),'difficulty':self.state.get('difficulty',1),'episodes':self.state.get('episodes',0),'decision':d,'directive':msgs.get(d,msgs['continue_training'])}
    def save(self):
        try:
            os.makedirs(DATA_DIR,exist_ok=True)
            with open(V36_SELF_EVAL_FILE,'w',encoding='utf-8') as f: json.dump(self.state,f,indent=2)
        except Exception: pass
    def summary(self): return self.directive()|{'best_reward':self.state.get('best_reward'),'improvements':self.state.get('improvements',0),'regressions':self.state.get('regressions',0),'stagnation':self.state.get('stagnation',0)}


# ============================================================
# V53 — SECURE AUTONOMOUS MINDMESH
# Integrates V37–V53 architecture milestones into the V36 core.
# ============================================================
V53_DIR=os.path.join(DATA_DIR,'v53'); os.makedirs(V53_DIR,exist_ok=True)

class MindMeshV53:
    PROFILES=[('mind_01',['jump_timing','movement']),('mind_02',['movement','jump_timing']),('mind_03',['precision_landing','movement']),('mind_04',['hazard_avoidance','precision_landing']),('mind_05',['route_selection','hazard_avoidance']),('mind_06',['exploration','route_selection']),('mind_07',['recovery','exploration']),('mind_08',['goal_approach','recovery'])]
    def __init__(self):
        self.path=os.path.join(V53_DIR,'mindmesh.json'); self.minds={}; self.assignments={}; self.load()
    def _new(self,i,f): return {'mind_id':i,'focus':f,'episodes':0,'steps':0,'reward':0.0,'successes':0,'failures':0,'mutations':0,'action_bias':{'A':0.0,'D':0.0,'W':0.0,'NONE':0.0},'private_rules':[]}
    def load(self):
        try:
            d=json.load(open(self.path,encoding='utf-8')); self.minds=d.get('minds',{}); self.assignments=d.get('assignments',{})
        except Exception: self.minds={}; self.assignments={}
        for i,f in self.PROFILES: self.minds.setdefault(i,self._new(i,f))
    def save(self):
        tmp=self.path+'.tmp'; json.dump({'minds':self.minds,'assignments':self.assignments},open(tmp,'w',encoding='utf-8'),indent=2); os.replace(tmp,self.path)
    def assign(self,wid):
        wid=str(wid)
        if wid not in self.assignments: self.assignments[wid]=self.PROFILES[len(self.assignments)%len(self.PROFILES)][0]; self.save()
        return self.assignments[wid]
    def observe(self,mid,a,r,success=False):
        m=self.minds.get(mid)
        if not m:return
        m['steps']+=1; m['reward']+=float(r); m['successes' if success else 'failures']+=1
        if a in m['action_bias']:
            d=.001 if float(r)>0 else -.0005; m['action_bias'][a]=max(-.25,min(.25,m['action_bias'][a]+d))
    def episode(self,mid,r,success):
        m=self.minds.get(mid)
        if not m:return
        m['episodes']+=1; m['reward']+=float(r); m['successes' if success else 'failures']+=1; self.save()
    def bias(self,mid,a): return float(self.minds.get(mid,{}).get('action_bias',{}).get(a,0.0))
    def evolve(self,mid):
        m=self.minds.get(mid)
        if m:
            for a in m['action_bias']: m['action_bias'][a]=max(-.25,min(.25,m['action_bias'][a]*.995))
            m['mutations']+=1

class ExperimentEngineV53:
    def __init__(self): self.path=os.path.join(V53_DIR,'experiments.json'); self.total=0; self.resolved=0; self.history=[]; self.load()
    def load(self):
        try:
            d=json.load(open(self.path,encoding='utf-8')); self.total=d.get('total',0); self.resolved=d.get('resolved',0); self.history=d.get('history',[])[-100:]
        except Exception: pass
    def save(self): json.dump({'total':self.total,'resolved':self.resolved,'history':self.history[-100:]},open(self.path,'w',encoding='utf-8'),indent=2)
    def propose(self,mid,h): self.total+=1; return {'experiment_id':f'exp_{self.total:06d}','mind_id':str(mid),'hypothesis':str(h),'status':'testing'}
    def resolve(self,e,outcome,confidence):
        e=dict(e); e.update(status='resolved',outcome=str(outcome),confidence=max(0,min(1,float(confidence)))); self.resolved+=1; self.history.append(e); self.save(); return e

class SharedKnowledgeV53:
    def __init__(self): self.path=os.path.join(V53_DIR,'shared_knowledge.json'); self.items=[]; self.load()
    def load(self):
        try:self.items=json.load(open(self.path,encoding='utf-8')).get('items',[])[-100:]
        except Exception:self.items=[]
    def save(self):json.dump({'items':self.items[-100:]},open(self.path,'w',encoding='utf-8'),indent=2)
    def publish(self,mid,claim,evidence,confidence):
        x={'source_mind':str(mid),'claim':str(claim),'evidence':str(evidence),'confidence':max(0,min(1,float(confidence))),'tests':0,'confirmed':False}; self.items.append(x); self.save(); return x
    def test(self,i,success):
        if not 0<=int(i)<len(self.items):return None
        x=self.items[int(i)]; x['tests']+=1; x['confidence']=max(0,min(1,x['confidence']+(.08 if success else -.12))); x['confirmed']=x['tests']>=3 and x['confidence']>=.7; self.save(); return x

class RouteGraphV53:
    def __init__(self): self.path=os.path.join(V53_DIR,'route_graph.json'); self.nodes={}; self.edges={}; self.load()
    def load(self):
        try:
            d=json.load(open(self.path,encoding='utf-8')); self.nodes=d.get('nodes',{}); self.edges=d.get('edges',{})
        except Exception: pass
    def save(self):json.dump({'nodes':self.nodes,'edges':self.edges},open(self.path,'w',encoding='utf-8'),indent=2)
    def observe(self,s,n,success):
        try:
            a=s.get('player',{}); b=n.get('player',{}); na=f"{round(float(a.get('x',0))/50)}:{round(float(a.get('y',0))/50)}"; nb=f"{round(float(b.get('x',0))/50)}:{round(float(b.get('y',0))/50)}"; self.nodes.setdefault(na,{'visits':0}); self.nodes.setdefault(nb,{'visits':0}); self.nodes[na]['visits']+=1; e=self.edges.setdefault(f'{na}->{nb}',{'attempts':0,'successes':0}); e['attempts']+=1; e['successes']+=int(bool(success))
        except Exception: pass
        self.save()

class ConfidenceEngineV53:
    def __init__(self):self.path=os.path.join(V53_DIR,'confidence.json');self.table={};self.load()
    def load(self):
        try:self.table=json.load(open(self.path,encoding='utf-8'))
        except Exception:self.table={}
    def save(self):json.dump(self.table,open(self.path,'w',encoding='utf-8'),indent=2)
    def observe(self,k,success):
        r=self.table.setdefault(str(k),{'attempts':0,'successes':0});r['attempts']+=1;r['successes']+=int(bool(success))
    def confidence(self,k):
        r=self.table.get(str(k)); return .25 if not r else max(0,min(1,r['successes']/max(1,r['attempts'])))

class SchedulerV53:
    TZ='Asia/Kolkata'
    @classmethod
    def active(cls):
        from zoneinfo import ZoneInfo
        from datetime import datetime,time
        t=datetime.now(ZoneInfo(cls.TZ)).time()
        return time(6,30)<=t<time(9,30) or time(18,30)<=t<time(21,30)

class MetaReasonerV53:
    def diagnose(self,reward,died=False,level_up=False):
        if level_up:return 'increase_precision'
        if died and float(reward)<-10:return 'reduce_risk'
        if float(reward)>5:return 'reinforce_strategy'
        return 'explore'

class V53Coordinator:
    def __init__(self):
        self.mindmesh=MindMeshV53();self.experiments=ExperimentEngineV53();self.knowledge=SharedKnowledgeV53();self.routes=RouteGraphV53();self.confidence=ConfidenceEngineV53();self.reasoner=MetaReasonerV53()
    def assign(self,w):return self.mindmesh.assign(w)
    def observe(self,mid,s,n,a,r,events=None):
        events=events or []; died='death' in events or 'died' in events; level='level_up' in events or 'goal' in events; ok=level or float(r)>0; self.mindmesh.observe(mid,a,r,ok); self.confidence.observe(f'{mid}:{a}',ok); self.routes.observe(s,n,ok); return self.reasoner.diagnose(r,died,level)
    def adjustment(self,mid,a): return self.mindmesh.bias(mid,a)*.2+(self.confidence.confidence(f'{mid}:{a}')-.5)*.04
    def episode(self,mid,r,ok):
        self.mindmesh.episode(mid,r,ok)
        m=self.mindmesh.minds.get(mid)
        if m and m['episodes']%25==0:self.mindmesh.evolve(mid)
    def snapshot(self):return {'mindmesh':{'minds':self.mindmesh.minds,'assignments':self.mindmesh.assignments},'experiments':{'total':self.experiments.total,'resolved':self.experiments.resolved,'history':self.experiments.history[-25:]},'shared_knowledge':self.knowledge.items[-50:],'route_graph':{'nodes':self.routes.nodes,'edges':self.routes.edges},'confidence':self.confidence.table,'schedule':{'timezone':self.scheduler.TZ if hasattr(self,'scheduler') else 'Asia/Kolkata','morning':'06:30-09:30','evening':'18:30-21:30'}}
    def save(self):self.mindmesh.save();self.experiments.save();self.knowledge.save();self.routes.save();self.confidence.save()

class BrainHub:
    """Central HTTP brain. Workers communicate through localhost HTTP so the
    brain/worker boundary can later be moved to separate Render services."""
    def __init__(self, memory, route_memory, memory_b=None):
        self.privacy_v52 = PrivacySanitizerV52()
        self.resilience_v37 = ResilienceBrainV37()
        self.mindmesh_v53 = V53Coordinator()
        self.self_evaluation_v36 = SelfEvaluationV36()
        self.world_model = PredictiveWorldModel()
        self.planner = ModelBasedPlanner(self.world_model, depth=3)
        self.memory = memory
        self.memory_b = memory_b if memory_b is not None else {}
        self.route_memory = route_memory
        self.hierarchical = HierarchicalController(self.planner, self.route_memory)
        self.replay = []
        self.director = TrainingDirector()
        self.failure_analyzer = FailureAnalyzer()
        self.failure_lab = FailureReplayLab()
        self.evolution = EvolutionManager()
        self.precision = PrecisionLandingController()
        # V33.1 FIX: initialize the tactical world model before choose(),
        # experience(), or snapshot() can access it.
        self.tactical = TacticalWorldModel()
        # V34: meta-learning layer sits above the existing V31 evolution system.
        self.meta_learning = MetaLearningManager(population_size=8)
        self.curriculum_v35 = AutonomousCurriculumV35()
        self.evolution.load()
        self.replay_update_count = 0
        self.epsilon = EPSILON_START
        self.lock = threading.RLock()
        self.stats = {
            "steps": 0, "episodes": 0, "landings": 0, "goals": 0,
            "jumps": 0, "replay_updates": 0, "total_reward": 0.0,
            "best_x": 0.0, "replay_size": 0, "workers_online": 0,
            "registered": 0, "heartbeats": 0, "requests": 0, "evaluation_episodes": 0, "evaluation_reward": 0.0, "evaluation_best": -1e18, "champion_updates": 0, "double_q_updates": 0, "planner_decisions": 0,
            "hierarchical_decisions": 0,
            "evolutions": 0,
            "precision_decisions": 0, "timing_landings": 0,
            "tactical_decisions": 0, "safe_route_decisions": 0,
            "meta_evolutions": 0,
        }
        self.workers = {}

    def register(self, worker_id, region=None):
        with self.lock:
            genome_id = self.evolution.assign_worker(worker_id)
            meta_strategy_id = self.meta_learning.assign_worker(worker_id)
            self.workers[str(worker_id)] = {
                "worker_id": worker_id, "status": "online", "last_seen": time.time(),
                "steps": 0, "episodes": 0, "level": 1, "best_x": 0.0,
                "genome_id": genome_id, "meta_strategy_id": meta_strategy_id,
                "curriculum_focus": self.curriculum_v35.focus,
                "region": str(region or "UNKNOWN").upper(),
            }
            self.stats["registered"] += 1
            self.stats["workers_online"] = len(self.workers)
            return {"ok": True, "worker_id": worker_id, "epsilon": self.epsilon}

    def heartbeat(self, worker_id, info=None):
        with self.lock:
            w = self.workers.setdefault(str(worker_id), {"worker_id": worker_id})
            w.update({"status": "online", "last_seen": time.time()})
            if info:
                w.update(info)
            self.stats["heartbeats"] += 1
            self.stats["workers_online"] = sum(
                1 for x in self.workers.values() if time.time() - x.get("last_seen", 0) < 10
            )
            return {"ok": True}

    def choose(self, worker_id, state, events=None):
        with self.lock:
            evaluation = os.getenv("AI_EVAL_MODE", "0") == "1"
            epsilon = 0.0 if evaluation else self.epsilon
            action, key = double_q_choose(self.memory, self.memory_b, state, epsilon, evaluation=evaluation)

            # V31: short-horizon model-predictive planning. The planner chooses
            # the first action of the best predicted action sequence.
            target_for_plan = self.tactical.choose_target(state, self.route_memory) or get_target(state, self.route_memory)
            self.stats["tactical_decisions"] += 1
            if self.tactical.last.get("hazard_score", 0) <= 0:
                self.stats["safe_route_decisions"] += 1
            precision_action, precision_hold, precision_meta = self.precision.decide(state, target_for_plan, action)
            if precision_action in valid_actions(state) and not evaluation:
                action = precision_action
                self.stats["precision_decisions"] += 1
            option = self.hierarchical.choose_option(state, target_for_plan, events or [], self.stats.get("steps", 0), evaluation)
            strategic_action = self.hierarchical.action(state, target_for_plan, option, evaluation)
            if strategic_action in valid_actions(state) and random.random() > epsilon * 0.55:
                action = strategic_action
                self.stats["hierarchical_decisions"] = self.stats.get("hierarchical_decisions", 0) + 1
            elif target_for_plan and not evaluation:
                planned_action, plan, plan_score = self.planner.plan(state, target_for_plan)
                if planned_action in valid_actions(state) and random.random() > epsilon * 0.35:
                    action = planned_action
                    self.stats["planner_decisions"] += 1
            self.stats["hierarchical_goal"] = option

            # V31: apply the assigned evolutionary genome. This is the actual
            # policy variation: candidates receive different action preferences.
            worker = self.workers.get(str(worker_id), {})
            genome_id = worker.get("genome_id", 0)
            allowed = valid_actions(state)
            if allowed and not evaluation:
                scored = [(a, self.evolution.action_bias(genome_id, a) + (0.15 if a == action else 0.0)) for a in allowed]
                if random.random() < 0.70:
                    action = max(scored, key=lambda x: x[1])[0]

            # V31: actively bias decisions toward weak routes. This is the
            # curriculum control missing from V21: the director does not merely
            # report weak routes; it changes action selection for them.
            target = self.tactical.last.get("target") or get_target(state, self.route_memory)
            rid = route_id(target) if target else None
            priority = self.director.priority(rid) if rid else 1.0
            failure_focus = self.failure_analyzer.focus
            if failure_focus and not evaluation:
                allowed = valid_actions(state)
                p = state["player"]
                if failure_focus == "jump_timing" and p.get("grounded") and "JUMP_RIGHT" in allowed:
                    if target and target["x"] > p["x"]:
                        action = "JUMP_RIGHT"
                elif failure_focus == "overshoot" and "LEFT" in allowed:
                    action = "LEFT"
                elif failure_focus == "undershoot" and "RIGHT" in allowed:
                    action = "RIGHT"
            if rid and priority > 1.25 and not evaluation:
                player = state["player"]
                center = target["x"] + target["width"] / 2
                direction = "RIGHT" if center > player["x"] + 10 else "LEFT"
                if player.get("grounded", False):
                    preferred = "JUMP_RIGHT" if direction == "RIGHT" else "JUMP_LEFT"
                else:
                    preferred = direction
                allowed = valid_actions(state)
                # Probability rises with weakness, capped to avoid completely
                # overriding learned behavior.
                pressure = min(0.90, 0.25 + 0.18 * priority)
                if preferred in allowed and random.random() < pressure:
                    action = preferred

            # V34: strategy adaptation. Compare candidate actions using the
            # worker's learned meta-strategy, but keep the influence small so
            # V33 Q-learning/planning/tactical behavior remains dominant.
            if not evaluation:
                allowed = valid_actions(state)
                if allowed:
                    meta_scored = [
                        (
                            a,
                            self.meta_learning.action_score(worker_id, a, state)
                            + (0.10 if a == action else 0.0),
                        )
                        for a in allowed
                    ]
                    meta_action = max(meta_scored, key=lambda x: x[1])[0]
                    if random.random() < 0.35:
                        action = meta_action

            # V35: autonomous curriculum. The agent chooses a training
            # objective from its own failure/success history and gently
            # steers actions toward that weakness.
            curriculum = self.curriculum_v35.directive()
            if not evaluation:
                allowed = valid_actions(state)
                if allowed and random.random() < curriculum["pressure"]:
                    curriculum_scored = [
                        (
                            a,
                            self.curriculum_v35.action_bias(a, state)
                            + (0.05 if a == action else 0.0),
                        )
                        for a in allowed
                    ]
                    curriculum_action, curriculum_score = max(
                        curriculum_scored, key=lambda x: x[1]
                    )
                    if curriculum_score > 0.0:
                        action = curriculum_action

            if not evaluation:
                self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            self.stats["requests"] += 1
            # Recompute a conservative hold after strategic overrides so V32 controls
            # the actual key duration without changing the learned action semantics.
            _, final_hold_ms, final_precision = self.precision.decide(state, target_for_plan, action)
            return {"action": action, "key": key, "epsilon": self.epsilon, "hold_ms": int(final_hold_ms),
                    "precision": final_precision, "curriculum_priority": priority, "focus_route": self.director.focus_route}

    def experience(self, worker_id, payload):
        with self.lock:
            self.stats["steps"] += 1
            self.stats["total_reward"] += float(payload.get("reward", 0.0))
            key = payload["key"]
            action = payload["action"]
            reward = float(payload["reward"])
            next_state = payload["next_state"]
            done = bool(payload.get("done", False))
            old_state = payload.get("state") or {}
            events = payload.get("events", [])
            failure_kind = self.failure_analyzer.classify(old_state, next_state, events) if old_state else None
            if failure_kind:
                self.failure_analyzer.record(failure_kind, success=False)
            elif "LANDING" in " ".join(events) and self.failure_analyzer.focus:
                self.failure_analyzer.record(self.failure_analyzer.focus, success=True)

            # V35 observes every transition to decide what should be practiced.
            try:
                self.curriculum_v35.record_event(
                    failure_kind=failure_kind,
                    events=events,
                    success=bool(payload.get("level_up")),
                    reward=reward,
                )
            except Exception:
                pass
            evaluation = os.getenv("AI_EVAL_MODE", "0") == "1"
            updates = 0
            if not evaluation:
                update_a = (self.stats["steps"] % 2 == 0)
                double_q_learn(self.memory, self.memory_b, key, action, reward, next_state, done=done, update_a=update_a)
                self.stats["double_q_updates"] += 1
                next_key = state_key(next_state)
                add_experience(self.replay, key, action, reward, next_key, done)
                self.stats["replay_size"] = len(self.replay)
                if self.stats["steps"] % REPLAY_EVERY == 0:
                    updates = replay_learn(self.memory, self.replay)
                    self.stats["replay_updates"] += int(updates)
                    self.stats["replay_size"] = len(self.replay)

            if not evaluation and action in ("JUMP_RIGHT", "JUMP_LEFT"):
                self.stats["jumps"] += 1

            target = payload.get("target")
            try:
                tactical_success = bool(("LANDING" in " ".join(events)) or payload.get("level_up"))
                self.tactical.record_outcome(target, tactical_success, events)
            except Exception:
                pass
            if "LANDING" in " ".join(events):
                self.stats["landings"] += 1
                self.stats["timing_landings"] = self.stats.get("timing_landings", 0) + 1
                update_route_memory(self.route_memory, target, True)

            if payload.get("level_up"):
                self.stats["goals"] += 1
                try:
                    self.precision.learn_landing(old_state, target, next_state["player"]["x"] + 10.0)
                except Exception:
                    pass
            if done:
                self.stats["episodes"] += 1
                update_route_memory(self.route_memory, target, False)
                worker = self.workers.get(str(worker_id), {})
                genome_id = worker.get("genome_id", 0)
                episode_score = float(payload.get("episode_reward", reward))
                should_evolve = self.evolution.record_episode(
                    genome_id, episode_score,
                    goals=1 if payload.get("level_up") else 0,
                    landings=1 if "LANDING" in " ".join(events) else 0,
                )
                worker["episodes"] = worker.get("episodes", 0) + 1
                # V34: score the meta-strategy using the completed episode.
                meta_should_evolve = self.meta_learning.record_episode(
                    worker_id,
                    episode_score,
                    success=bool(payload.get("level_up")),
                )

                # V35 closes the loop: choose the next weakness after seeing
                # the complete episode outcome.
                self.curriculum_v35.record_episode(
                    episode_score,
                    success=bool(payload.get("level_up")),
                )
                try:
                    self.self_evaluation_v36.record(episode_score)
                except Exception:
                    pass
                worker["curriculum_focus"] = self.curriculum_v35.focus

                if should_evolve and not evaluation:
                    elites = self.evolution.evolve()
                    self.stats["evolutions"] = self.stats.get("evolutions", 0) + 1
                    print(f"🧬 V31 EVOLUTION → generation {self.evolution.generation} | elites: {[x.get('id') for x in elites]}")

                if meta_should_evolve and not evaluation:
                    meta_elites = self.meta_learning.evolve()
                    self.stats["meta_evolutions"] = self.stats.get("meta_evolutions", 0) + 1
                    print(
                        f"🧬 V34 META EVOLUTION → generation "
                        f"{self.meta_learning.generation} | elites: "
                        f"{[x.get('id') for x in meta_elites]}"
                    )
            try:
                _mid = self.workers.get(str(worker_id), {}).get("mind_id")
                if _mid:
                    self.mindmesh_v53.observe(_mid, old_state, next_state, action, reward, events)
                    if done:
                        self.mindmesh_v53.episode(_mid, payload.get("episode_reward", reward), bool(payload.get("level_up")))
            except Exception:
                pass

            if not evaluation and target:
                self.director.record(route_id(target), bool(("LANDING" in " ".join(events)) or payload.get("level_up")))
            if self.stats["steps"] % 100 == 0:
                self.director.save()
                self.failure_analyzer.save()
                self.failure_lab.save()
                self.evolution.save()
                self.meta_learning.save()
                self.curriculum_v35.save()
                self.tactical.save()
            self.resilience_v37.save()
            self.self_evaluation_v36.save()
            self.stats["best_x"] = max(self.stats["best_x"], next_state["player"]["x"])
            self.stats["replay_size"] = len(self.replay)

            w = self.workers.setdefault(str(worker_id), {"worker_id": worker_id})
            w["last_seen"] = time.time()
            w["steps"] = w.get("steps", 0) + 1
            w["level"] = next_state.get("currentLevel", 1)
            w["best_x"] = max(w.get("best_x", 0.0), next_state["player"]["x"])
            return {"ok": True, "epsilon": self.epsilon, "updates": updates}

    def record_evaluation(self, worker_id, reward, level=1, goals=0):
        with self.lock:
            reward = float(reward)
            self.stats["evaluation_episodes"] += 1
            self.stats["evaluation_reward"] += reward
            if reward > self.stats["evaluation_best"]:
                self.stats["evaluation_best"] = reward
                try:
                    with open(CHAMPION_A_FILE, "w", encoding="utf-8") as f:
                        json.dump(self.memory, f, indent=2)
                    with open(CHAMPION_B_FILE, "w", encoding="utf-8") as f:
                        json.dump(self.memory_b, f, indent=2)
                    self.stats["champion_updates"] += 1
                    return {"ok": True, "new_champion": True, "reward": reward}
                except Exception as e:
                    return {"ok": False, "new_champion": False, "error": str(e)}
            return {"ok": True, "new_champion": False, "reward": reward}

    def snapshot(self):
        with self.lock:
            now = time.time()
            workers = []
            for wid, w in self.workers.items():
                item = dict(w)
                item["status"] = "online" if now - w.get("last_seen", 0) < 10 else "offline"
                workers.append(item)
            return {
                **self.stats,
                "states": len(self.memory), "q_table_b_states": len(self.memory_b), "routes": len(self.route_memory),
                "replay": len(self.replay), "epsilon": round(self.epsilon, 4),
                "workers": workers,
                "curriculum_focus": self.director.focus_route,
                "curriculum": self.director.summary(),
                "failure_analysis": self.failure_analyzer.summary(),
                "failure_replay": self.failure_lab.summary(),
                "tactical_world": self.tactical.last,
                "evolution": self.evolution.summary(),
                "meta_learning": self.meta_learning.summary(),
                "autonomous_curriculum": self.curriculum_v35.summary(),
                "self_evaluation_v36": self.self_evaluation_v36.summary(),
                "resilience_v37": self.resilience_v37.summary(),
                "privacy_v52": self.privacy_v52.summary(),
            "mindmesh_v53": self.mindmesh_v53.snapshot(),
                "champion": {"best_eval_reward": self.stats["evaluation_best"], "updates": self.stats["champion_updates"]},
            }

    def save(self):
        with self.lock:
            save_memory(self.memory)
            self.world_model.save()
            save_memory_b(self.memory_b)
            save_route_memory(self.route_memory)
            self.director.save()
            self.failure_analyzer.save()
            self.failure_lab.save()
            self.meta_learning.save()
            self.curriculum_v35.save()
            self.tactical.save()
        self.mindmesh_v53.save()


def start_brain_server(brain):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if not BRAIN_API_KEY:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {BRAIN_API_KEY}"

        def do_GET(self):
            if not self._authorized():
                self._json(401, {"ok": False, "error": "unauthorized"}); return
            if self.path == "/":
                self._json(200, {"status": "online", "service": "AI Learning Lab V31", "mode": "central-brain"})
            elif self.path == "/health":
                self._json(200, {"status": "online", "service": "central-brain-v27"})
            elif self.path == "/stats":
                self._json(200, brain.snapshot())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                self._json(401, {"ok": False, "error": "unauthorized"}); return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/register":
                    out = brain.register(payload["worker_id"], payload.get("region"))
                elif self.path == "/heartbeat":
                    out = brain.heartbeat(payload["worker_id"], payload.get("info"))
                elif self.path == "/action":
                    out = brain.choose(payload["worker_id"], payload["state"], payload.get("events", []))
                elif self.path == "/experience":
                    out = brain.experience(payload["worker_id"], payload)
                elif self.path == "/evaluation":
                    out = brain.record_evaluation(payload["worker_id"], payload.get("reward", 0.0), payload.get("level", 1), payload.get("goals", 0))
                elif self.path == "/save":
                    brain.save(); out = {"ok": True}
                else:
                    self._json(404, {"error": "not found"}); return
                self._json(200, out)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((BRAIN_HOST, BRAIN_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"🧠 Central brain API listening on {BRAIN_HOST}:{BRAIN_PORT}")
    return server


# V53 PRIVACY HARDENING: every outbound BrainHub payload is sanitized at the
# network boundary.  This covers normal calls, retries, recovery calls, and
# any future caller that uses http_json().
WORKER_PRIVACY_SANITIZER = PrivacySanitizerV52()

def _sanitize_outgoing_payload(payload):
    if payload is None:
        return None
    # sanitize() returns a new structure for dict/list payloads, so the
    # training state/event objects used by the worker are not mutated.
    return WORKER_PRIVACY_SANITIZER.sanitize(payload)


def http_json(path, payload=None):
    import urllib.request

    # SECURITY BOUNDARY: sanitize immediately before serialization/network I/O.
    # Every BrainHub request therefore gets the same privacy treatment,
    # including calls made by recovery and retry paths.
    safe_payload = _sanitize_outgoing_payload(payload)
    url = f"{BRAIN_URL}{path}"
    data = None if safe_payload is None else json.dumps(safe_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if BRAIN_API_KEY:
        headers["Authorization"] = f"Bearer {BRAIN_API_KEY}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


async def brain_request(path, payload=None):
    # Keep this function intentionally thin: http_json() is the mandatory
    # privacy boundary, so all callers (including retry/recovery paths) are
    # covered automatically.
    return await asyncio.to_thread(http_json, path, payload)


async def _close_browser_safely(browser):
    try:
        if browser:
            await browser.close()
    except Exception:
        pass


def _failure_kind(exc=None, state=None, page=None, operation="STATE_READ"):
    """Classify transient worker failures without leaking machine/network metadata."""
    if state:
        return None

    text = str(exc or "").lower()

    if page is not None:
        try:
            if page.is_closed():
                return "PAGE_CLOSED"
        except Exception:
            return "PAGE_CLOSED"

    if operation == "BRAIN":
        if any(x in text for x in ("timed out", "timeout", "timedout")):
            return "BRAIN_TIMEOUT"
        if any(x in text for x in ("500", "502", "503", "504", "urlopen", "connection")):
            return "BRAIN_HTTP_ERROR"

    if any(x in text for x in ("target page", "browser has been closed", "target closed", "page closed")):
        return "PAGE_CLOSED"

    if any(x in text for x in ("cdp", "protocol", "execution context")):
        return "CDP_FAILURE"

    if operation == "ACTION":
        return "ACTION_FAILURE"

    if any(x in text for x in ("player", "game state", "not visible", "execution context")):
        return "GAME_NOT_READY"

    return "STATE_UNAVAILABLE" if operation == "STATE_READ" else "UNKNOWN_FAILURE"


async def _resilient_brain_request(path, payload, worker_id, resilience, *,
                                    attempts=5, default=None, required=True):
    """Retry BrainHub calls so a transient HTTP failure never kills training."""
    last_error = None

    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            # Defense in depth: create a sanitized retry payload once per
            # attempt. http_json() sanitizes again at the final network boundary.
            safe_payload = _sanitize_outgoing_payload(payload)
            result = await brain_request(path, safe_payload)
            if resilience:
                fp = resilience.fingerprint("BRAIN_HTTP", worker_id)
                resilience.record(fp, True, time.time() - started)
            return result
        except Exception as e:
            last_error = e
            kind = _failure_kind(e, operation="BRAIN")
            print(f"⚠️ BrainHub failure: {kind} | attempt {attempt}/{attempts}")

            if resilience:
                decision = resilience.begin(kind, worker_id)
                delay = float(decision.get("delay", min(12.0, 0.8 * (2 ** (attempt - 1)))))
            else:
                delay = min(12.0, 0.8 * (2 ** (attempt - 1)))

            print(f"🧠 Resilience analysis → retry in {delay:.2f}s")
            await asyncio.sleep(delay)

    if required:
        raise RuntimeError(f"BrainHub unavailable after {attempts} attempts: {last_error}")

    return default


async def _recover_browser(playwright, browser, context, page, cdp, worker_id, resilience):
    """Recreate browser/page/CDP without touching the learning brain or counters."""
    print(f"🔄 Worker {worker_id}: reconnecting browser/game...")

    await _close_browser_safely(browser)

    for attempt in range(1, 6):
        started = time.time()
        try:
            print(f"   🔧 Browser recovery attempt {attempt}/5")
            browser, context, page = await create_browser(playwright)
            cdp = await page.context.new_cdp_session(page)

            # Give the game JS a little time after navigation.
            await asyncio.sleep(0.5)

            state = await get_state(page, cdp)
            if state:
                if resilience:
                    fp = resilience.fingerprint("BROWSER_RECOVERY", worker_id)
                    resilience.record(fp, True, time.time() - started)
                print(f"✅ Worker {worker_id}: browser/game recovered")
                return browser, context, page, cdp, state

            print("   ⚠️ Browser opened, but game state is not ready yet.")
        except Exception as e:
            kind = _failure_kind(e, operation="BROWSER_RECOVERY")
            print(f"   ⚠️ Recovery failure: {kind}: {e}")

        if resilience:
            decision = resilience.begin("BROWSER_RECOVERY", worker_id)
            delay = float(decision.get("delay", 1.0))
        else:
            delay = min(12.0, 0.8 * (2 ** (attempt - 1)))

        print(f"   ⏳ Waiting {delay:.2f}s before recovery retry...")
        await asyncio.sleep(delay)

        await _close_browser_safely(browser)
        browser = context = page = cdp = None

    if resilience:
        fp = resilience.fingerprint("BROWSER_RECOVERY", worker_id)
        resilience.record(fp, False, 0)

    return None, None, None, None, None


async def _read_state_resilient(playwright, browser, context, page, cdp,
                                worker_id, resilience):
    """
    Read game state indefinitely until it is available.

    Recovery order:
      1. repeated state reads
      2. short page wait
      3. browser/page/CDP recreation
      4. repeat forever with bounded backoff

    This function intentionally has no terminal 'return None' path.
    """
    read_attempt = 0

    while True:
        read_attempt += 1
        started = time.time()

        try:
            if page is None or page.is_closed():
                raise RuntimeError("page closed")

            state = await get_state(page, cdp)
            if state:
                if read_attempt > 1:
                    print(f"✅ Worker {worker_id}: state recovered after {read_attempt} attempts")
                return browser, context, page, cdp, state

            kind = "GAME_NOT_READY"
            print(f"⚠️ State acquisition failed | reason={kind} | attempt={read_attempt}")
        except Exception as e:
            kind = _failure_kind(e, page=page, operation="STATE_READ")
            print(f"⚠️ State acquisition failed | reason={kind} | attempt={read_attempt}")
            print(f"   detail: {e}")

        if resilience:
            decision = resilience.begin(kind, worker_id)
            delay = float(decision.get("delay", 0.8))
        else:
            delay = min(12.0, 0.8 * (2 ** min(read_attempt - 1, 4)))

        # First try the cheap recovery path before destroying a healthy browser.
        if read_attempt <= 5:
            print(f"🧠 Resilience analysis...")
            print(f"⏳ Waiting {delay:.2f}s...")
            await asyncio.sleep(delay)
            continue

        print("🔄 Reconnecting browser/CDP because state is still unavailable...")
        browser, context, page, cdp, state = await _recover_browser(
            playwright, browser, context, page, cdp, worker_id, resilience
        )

        if state:
            return browser, context, page, cdp, state

        # Never terminate the worker because the game is temporarily unavailable.
        read_attempt = 0
        print("⚠️ Recovery cycle did not restore state.")
        print("🧠 Preserving learning state and starting another recovery cycle...")
        await asyncio.sleep(2.0)


async def train_worker(worker_id, p):
    """
    V53 worker loop with persistent browser/game/BrainHub recovery.

    The V53 learning stack remains untouched:
    Q-learning, replay, world model, planner, precision landing,
    tactical routing, evolution, curriculum, self-evaluation, privacy,
    resilience, and MindMesh all continue to operate through BrainHub.
    """
    trajectory = []
    browser = context = page = cdp = None
    episode_reward = 0.0
    episode_target = None
    local_episodes = 0
    local_step = 0

    # Local resilience object is deliberately independent from BrainHub so the
    # worker can recover even when BrainHub itself is temporarily unreachable.
    resilience = ResilienceBrainV37()

    _render_service = os.getenv("RENDER_SERVICE_NAME", "local")
    _render_instance = os.getenv("RENDER_INSTANCE_ID", "local")
    _worker_tag = hashlib.sha256(
        f"{_render_service}:{_render_instance}:{WORKER_REGION}:{worker_id}".encode("utf-8")
    ).hexdigest()[:16]
    safe_worker_id = f"{WORKER_REGION.lower()}_{_worker_tag}"

    async def ensure_browser():
        nonlocal browser, context, page, cdp
        if page is not None:
            try:
                if not page.is_closed() and cdp is not None:
                    return
            except Exception:
                pass

        browser, context, page, cdp, state = await _recover_browser(
            p, browser, context, page, cdp, safe_worker_id, resilience
        )
        if state is None:
            raise RuntimeError("Browser recovery unexpectedly returned without state")

    try:
        # Registration is also retried forever; a BrainHub restart must not kill
        # the browser worker.
        while True:
            try:
                await _resilient_brain_request(
                    "/register",
                    {"worker_id": safe_worker_id, "region": WORKER_REGION},
                    safe_worker_id,
                    resilience,
                    attempts=5,
                    required=True,
                )
                break
            except Exception as e:
                print(f"⚠️ Worker {worker_id}: BrainHub registration unavailable: {e}")
                print("🔁 Registration will continue retrying...")
                await asyncio.sleep(3)

        print(f"🧠 Worker {worker_id}: registered as {safe_worker_id}")

        await ensure_browser()

        browser, context, page, cdp, state = await _read_state_resilient(
            p, browser, context, page, cdp, safe_worker_id, resilience
        )

        print(f"✅ Worker {worker_id} ONLINE → central brain")
        print("🛡️ Persistent recovery mode: ACTIVE")

        while local_step < MAX_STEPS:
            # --------------------------------------------------------
            # 1. Guarantee a live page + valid state.
            # --------------------------------------------------------
            browser, context, page, cdp, state = await _read_state_resilient(
                p, browser, context, page, cdp, safe_worker_id, resilience
            )

            if episode_target is None:
                episode_target = get_target(state, None)

            # --------------------------------------------------------
            # 2. Ask the complete V53 BrainHub for an action.
            # --------------------------------------------------------
            try:
                decision = await _resilient_brain_request(
                    "/action",
                    {
                        "worker_id": safe_worker_id,
                        "state": state,
                        "events": [],
                    },
                    safe_worker_id,
                    resilience,
                    attempts=5,
                    required=True,
                )
            except Exception as e:
                print(f"⚠️ Action decision unavailable: {e}")
                print("🔄 Recovering BrainHub connection without losing the episode...")
                await asyncio.sleep(2)
                continue

            action = decision["action"]
            epsilon = decision["epsilon"]
            key = decision["key"]
            hold_ms = decision.get("hold_ms")

            # --------------------------------------------------------
            # 3. Execute action. A browser/input failure is recoverable.
            # --------------------------------------------------------
            try:
                ok = await execute_action(page, action, state, hold_ms)
            except Exception as e:
                ok = False
                print(f"⚠️ Action execution exception: {e}")

            if not ok:
                print(f"⚠️ ACTION FAILURE | worker={worker_id} | action={action}")
                print("🧠 Preserving episode/learning memory")
                browser, context, page, cdp, recovered_state = await _recover_browser(
                    p, browser, context, page, cdp, safe_worker_id, resilience
                )
                if recovered_state is not None:
                    state = recovered_state
                    print("✅ Action failure recovered — continuing learning")
                else:
                    print("⚠️ Action recovery still pending — retrying")
                continue

            # --------------------------------------------------------
            # 4. Read the next state. Never stop because it is temporarily
            #    unavailable.
            # --------------------------------------------------------
            browser, context, page, cdp, next_state = await _read_state_resilient(
                p, browser, context, page, cdp, safe_worker_id, resilience
            )

            # --------------------------------------------------------
            # 5. Normal V53 learning transition.
            # --------------------------------------------------------
            reward, events = calculate_reward(state, next_state, None)
            died = next_state["deaths"] > state["deaths"]
            level_up = next_state["currentLevel"] > state["currentLevel"]
            landed = "LANDING" in " ".join(events)

            target_payload = episode_target
            trajectory.append({
                "state": state_key(state),
                "action": action,
                "reward": reward,
                "events": events[-3:],
            })
            if len(trajectory) > 200:
                trajectory = trajectory[-200:]

            experience_payload = {
                "worker_id": safe_worker_id,
                "region": WORKER_REGION,
                "key": key,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": died,
                "events": events,
                "target": target_payload,
                "level_up": level_up,
                "episode_reward": episode_reward + reward,
                "state": state,
                "trajectory": trajectory[-REPLAY_WINDOW:],
                "reason": "death" if died else ("level_up" if level_up else "failure"),
            }

            try:
                await _resilient_brain_request(
                    "/experience",
                    experience_payload,
                    safe_worker_id,
                    resilience,
                    attempts=5,
                    required=True,
                )
            except Exception as e:
                # The transition is still kept locally in trajectory/state.
                # The worker does not terminate merely because the central brain
                # missed one telemetry packet.
                print(f"⚠️ Experience report unavailable: {e}")
                print("💾 Transition retained locally; continuing.")

            try:
                await _resilient_brain_request(
                    "/heartbeat",
                    {
                        "worker_id": safe_worker_id,
                        "region": WORKER_REGION,
                        "info": {
                            "level": next_state.get("currentLevel", 1),
                            "best_x": next_state["player"]["x"],
                        },
                    },
                    safe_worker_id,
                    resilience,
                    attempts=3,
                    required=False,
                )
            except Exception:
                pass

            episode_reward += reward
            local_step += 1

            if local_step % 20 == 0 or events or died:
                target = get_target(next_state, None)
                target_text = (
                    f"{target['x']:.0f}-{target['x'] + target['width']:.0f}"
                    if target else "NONE"
                )

                try:
                    snap = await _resilient_brain_request(
                        "/stats",
                        None,
                        safe_worker_id,
                        resilience,
                        attempts=3,
                        required=False,
                        default={},
                    )
                except Exception:
                    snap = {}

                print(f"\n[W{worker_id} S{local_step}] 🤖 {action}")
                print(
                    f"   🎮 X={next_state['player']['x']:.1f} "
                    f"Y={next_state['player']['y']:.1f} "
                    f"VX={next_state['player']['velocityX']:.1f} "
                    f"VY={next_state['player']['velocityY']:.1f}"
                )
                print(f"   🎯 Target={target_text} | Level={next_state['currentLevel']}")
                print(
                    f"   🛬 Predicted landing X="
                    f"{predicted_landing_x(next_state['player']):.1f}"
                )
                print(
                    f"   🧠 Central states={snap.get('states', '?')} "
                    f"replay={snap.get('replay', '?')} "
                    f"ε={snap.get('epsilon', epsilon):.4f}"
                )
                print(
                    f"   🌐 Workers online={snap.get('workers_online', '?')} "
                    f"| Total steps={snap.get('steps', '?')}"
                )
                print(
                    f"   📊 Deaths={next_state['deaths']} "
                    f"Reward={reward:.2f}"
                )
                for event in events:
                    print(f"   ⚡ {event}")

            if died:
                trajectory = []
                local_episodes += 1
                final_episode_reward = episode_reward

                if os.getenv("AI_EVAL_MODE", "0") == "1":
                    try:
                        await _resilient_brain_request(
                            "/evaluation",
                            {
                                "worker_id": safe_worker_id,
                                "region": WORKER_REGION,
                                "reward": final_episode_reward,
                                "level": next_state.get("currentLevel", 1),
                                "goals": 1 if level_up else 0,
                            },
                            safe_worker_id,
                            resilience,
                            attempts=3,
                            required=False,
                        )
                    except Exception:
                        pass

                print(
                    f"\n💀 WORKER {worker_id} EPISODE "
                    f"{local_episodes} FINISHED | Reward={final_episode_reward:.2f}"
                )

                episode_reward = 0.0
                episode_target = None
                await asyncio.sleep(0.25)

            state = next_state

    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Unexpected bugs are logged, but we keep the worker alive by entering
        # a recovery loop instead of silently terminating the training process.
        print(f"\n❌ Worker {worker_id} unexpected error: {type(e).__name__}: {e}")
        print("🛡️ Entering persistent recovery mode...")

        while True:
            try:
                browser, context, page, cdp, state = await _recover_browser(
                    p, browser, context, page, cdp, safe_worker_id, resilience
                )
                if state:
                    print(f"✅ Worker {worker_id}: recovered after unexpected error")
                    # Re-enter the worker from the next safe state. We deliberately
                    # do not reset BrainHub or learning memory here.
                    while True:
                        await asyncio.sleep(1)
                        try:
                            browser, context, page, cdp, state = await _read_state_resilient(
                                p, browser, context, page, cdp, safe_worker_id, resilience
                            )
                            # A successful recovery leaves the process alive. The
                            # outer training supervisor can restart this worker if
                            # desired; never silently exit.
                            print(
                                f"🧠 Worker {worker_id}: state stream recovered; "
                                "recovery supervisor remains active."
                            )
                            break
                        except Exception as recovery_error:
                            print(
                                f"⚠️ Recovery supervisor: "
                                f"{type(recovery_error).__name__}: {recovery_error}"
                            )
                    break
            except asyncio.CancelledError:
                raise
            except Exception as recovery_error:
                print(
                    f"⚠️ Persistent recovery failed: "
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
                await asyncio.sleep(3)

    finally:
        # Normal shutdown only happens when the process itself is stopped.
        await _close_browser_safely(browser)


async def supervised_train_worker(worker_id, p):
    """Keep a worker alive even if an unexpected exception escapes train_worker."""
    restart_count = 0

    while True:
        try:
            await train_worker(worker_id, p)
            # Each training session is TRAIN_STEPS (10,000 by default).
            # Render background workers normally continue with another session
            # so the instance remains useful after a session boundary.
            if os.getenv("RUN_ONE_SESSION", "0").strip().lower() in {"1", "true", "yes"}:
                print(f"\n🏁 Worker {worker_id}: one-session mode complete.")
                return
            # A normal MAX_STEPS completion is a session boundary. For a
            # persistent worker service, immediately start another session so
            # learning can continue from the same BrainHub/memory.
            restart_count += 1
            print(
                f"\n🔁 Worker {worker_id}: training session ended normally. "
                f"Restarting session #{restart_count} with preserved learning."
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            restart_count += 1
            print(
                f"\n💥 Worker supervisor {worker_id}: "
                f"{type(e).__name__}: {e}"
            )
            print(
                f"🛡️ Restarting worker #{restart_count} without resetting "
                "the central learning brain."
            )

        await asyncio.sleep(2.0)


# ============================================================
# LOCAL TRAINING DASHBOARD
# ============================================================

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080


def start_dashboard(brain):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/stats":
                body = json.dumps(brain.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            workers = """<div id='workers'></div>"""
            html = """<!doctype html><html><head><meta charset='utf-8'><title>AI Learning Lab V31</title>
<style>body{font-family:system-ui;margin:30px;max-width:1200px;background:#f6f6f6}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:white;border:1px solid #ccc;border-radius:12px;padding:16px}.v{font-size:28px;font-weight:700}table{width:100%;background:white;border-collapse:collapse;margin-top:20px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.online{font-weight:700}pre{white-space:pre-wrap}</style></head>
<body><h1>🧠 AI Learning Lab V31</h1><p>Central brain + 8 HTTP workers · localhost only</p><div id='grid' class='grid'></div><h2>👷 Workers</h2><div id='workers'></div><pre id='raw'></pre>
<script>async function tick(){try{let d=await (await fetch('/api/stats')).json();let names=[['Steps','steps'],['Episodes','episodes'],['Goals','goals'],['Landings','landings'],['Best X','best_x'],['States','states'],['Replay','replay'],['Routes','routes'],['Epsilon','epsilon'],['Reward','total_reward'],['Replay updates','replay_updates'],['Workers online','workers_online'],['Curriculum focus','curriculum_focus']];document.getElementById('grid').innerHTML=names.map(x=>`<div class='card'><div>${x[0]}</div><div class='v'>${d[x[1]]??'—'}</div></div>`).join('');document.getElementById('workers').innerHTML='<table><tr><th>Worker</th><th>Status</th><th>Level</th><th>Steps</th><th>Best X</th></tr>'+d.workers.map(w=>`<tr><td>Worker ${w.worker_id??'?'}</td><td class='online'>${w.status}</td><td>${w.level??'—'}</td><td>${w.steps??0}</td><td>${(w.best_x??0).toFixed?.(1)??w.best_x}</td></tr>`).join('')+'</table>';document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){}}tick();setInterval(tick,1000)</script></body></html>"""
            body=html.encode('utf-8'); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt, *args): return
    server=ThreadingHTTPServer((DASHBOARD_HOST,DASHBOARD_PORT),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f"📊 Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    return server


async def train_local():
    memory = load_memory()
    route_memory = load_route_memory()
    memory_b = load_memory_b()
    brain = BrainHub(memory, route_memory, memory_b)
    brain_server = start_brain_server(brain)
    dashboard = start_dashboard(brain)
    print("=" * 72)
    print("🧠 V31 EVOLUTION PERFORMANCE LAB |  CENTRAL BRAIN + WORKER HUB")
    print("🎯 REAL ADAPTIVE CURRICULUM CONTROL: ACTIVE")
    print("🧬 🧬 REAL EVOLVING STRATEGIES: ACTIVE")
    print(f"🚀 {NUM_ENVS} workers communicate with ONE central brain")
    print(f"🧠 Brain API: http://{BRAIN_HOST}:{BRAIN_PORT}")
    print(f"📊 Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("🌐 Target: Render-hosted game")
    print("👀 Visible browsers: ON")
    print("=" * 72)
    try:
        async with async_playwright() as p:
            tasks=[asyncio.create_task(supervised_train_worker(i+1,p)) for i in range(NUM_ENVS)]
            try: await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                print("\n🛑 Stopping workers...")
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
    finally:
        print("\n💾 Saving central brain...")
        brain.save()
        try: dashboard.shutdown(); dashboard.server_close()
        except Exception: pass
        try: brain_server.shutdown(); brain_server.server_close()
        except Exception: pass
    snap=brain.snapshot()
    try:
        with open(RUN_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "mode": "evaluation" if os.getenv("AI_EVAL_MODE", "0") == "1" else "training",
                "timestamp": time.time(),
                "steps": snap.get("steps", 0),
                "episodes": snap.get("episodes", 0),
                "best_x": snap.get("best_x", 0.0),
                "goals": snap.get("goals", 0),
                "landings": snap.get("landings", 0),
                "total_reward": snap.get("total_reward", 0.0),
                "evaluation_episodes": snap.get("evaluation_episodes", 0),
                "evaluation_reward": snap.get("evaluation_reward", 0.0),
                "evaluation_best": snap.get("evaluation_best", -1e18),
            }, f, indent=2)
    except Exception as e:
        print(f"⚠️ Run summary save failed: {e}")
    print("\n"+"="*72)
    print("🏁 V31 FINISHED")
    print(f"🚀 Best X: {snap['best_x']:.1f}")
    print(f"🧠 Learned states: {snap['states']}")
    print(f"🗺️ Routes: {snap['routes']}")
    print(f"💀 Episodes: {snap['episodes']}")
    print(f"🛬 Landings: {snap['landings']} | 🏆 Goals: {snap['goals']}")
    print(f"🔁 Replay updates: {snap['replay_updates']}")
    print(f"💰 Total reward: {snap['total_reward']:.2f}")
    print("="*72)


# ============================================================
# V31 AUTONOMOUS TRAINING MANAGER
# ============================================================

def run_auto(cycles, workers):
    import subprocess
    manager = AutonomousTrainingManager()
    print("=" * 72)
    print("🤖 V31 SELF-DIRECTED AUTONOMOUS TRAINING MANAGER ONLINE")
    print("🔍 FAILURE ANALYZER + TRAINING OBJECTIVES: ACTIVE")
    print("🔄 TRAIN → EVALUATE → KEEP CHAMPION → REPEAT")
    print(f"🧬 Cycles: {cycles} | Training workers: {workers} | Eval workers: {AUTO_EVAL_WORKERS}")
    print("=" * 72)

    script = os.path.abspath(__file__)
    for cycle in range(1, cycles + 1):
        print(f"\n🚀 CYCLE {cycle}/{cycles} — TRAINING")
        train_cmd = [sys.executable, script, "--workers", str(workers)]
        train_result = subprocess.run(train_cmd)
        if train_result.returncode != 0:
            print(f"❌ Training cycle failed with exit code {train_result.returncode}")
            break

        print(f"\n🧪 CYCLE {cycle}/{cycles} — EVALUATION")
        eval_cmd = [sys.executable, script, "--eval", "--workers", str(AUTO_EVAL_WORKERS)]
        eval_result = subprocess.run(eval_cmd)
        if eval_result.returncode != 0:
            print(f"❌ Evaluation cycle failed with exit code {eval_result.returncode}")
            break

        try:
            with open(RUN_SUMMARY_FILE, "r", encoding="utf-8") as f:
                summary = json.load(f)
            reward = float(summary.get("evaluation_best", summary.get("total_reward", -1e18)))
            level = int(summary.get("best_x", 0))
            goals = int(summary.get("goals", 0))
        except Exception as e:
            print(f"⚠️ Could not read evaluation summary: {e}")
            continue

        # Evaluation process already preserves a champion when its local
        # evaluation improves. The manager adds a cross-run baseline so the
        # training history survives process restarts.
        improved = manager.accept(reward, level, goals)
        if improved:
            print(f"🏆 NEW CHAMPION — eval reward {reward:.2f}")
        else:
            print(f"📉 No improvement — best remains {manager.best_eval_reward:.2f}")
            if manager.stagnation >= 3:
                print("🧬 STAGNATION DETECTED — evolution/curriculum will continue adapting")
        print(f"📊 Manager: {manager.summary()}")

    print("\n" + "=" * 72)
    print("🏁 V31 AUTONOMOUS MANAGER FINISHED")
    print(f"🏆 Best evaluation reward: {manager.best_eval_reward:.2f}")
    print(f"🧬 Manager generation: {manager.generation}")
    print(f"💤 Stagnation cycles: {manager.stagnation}")
    print("=" * 72)


# ============================================================
# V31 START MODES
# ============================================================

def parse_mode():
    import argparse
    parser = argparse.ArgumentParser(description="AI Learning Lab V31")
    parser.add_argument("--brain", action="store_true", help="run central brain server only")
    parser.add_argument("--workers", type=int, default=NUM_ENVS, help="local browser workers (1-32)")
    parser.add_argument("--steps", type=int, default=None, help="steps per training session; defaults to TRAIN_STEPS/10000")
    parser.add_argument("--region", choices=["US", "VA", "OH", "EU", "GB", "UK", "SG", "OR"], default=None, help="regional worker pool")
    parser.add_argument("--eval", action="store_true", help="run deterministic evaluation mode")
    parser.add_argument("--auto", action="store_true", help="run autonomous train/evaluate cycles")
    parser.add_argument("--schedule", action="store_true", help="run only during 06:30-09:30 and 18:30-21:30 Asia/Kolkata")
    parser.add_argument("--cycles", type=int, default=AUTO_TRAIN_CYCLES, help="autonomous manager cycles")
    return parser.parse_args()


def run_brain():
    memory = load_memory()
    route_memory = load_route_memory()
    memory_b = load_memory_b()
    brain = BrainHub(memory, route_memory, memory_b)
    server = start_brain_server(brain)
    print("=" * 72)
    print("🧠 V31 REMOTE CENTRAL BRAIN ONLINE")
    print("☁️  Designed for Render / remote hosting")
    print(f"🔐 API key: {'ON' if BRAIN_API_KEY else 'OFF — set BRAIN_API_KEY before exposing it'}")
    print("🌐 GET /health   GET /stats")
    print("=" * 72)
    try:
        while True:
            time.sleep(5)
            # Periodic checkpoint. On Render, use a persistent disk if you need
            # memory to survive service restarts/deploys.
            if int(time.time()) % 30 < 5:
                brain.save()
    except KeyboardInterrupt:
        print("\n🛑 Brain stopping...")
    finally:
        brain.save()
        server.shutdown()
        server.server_close()


async def train_remote(workers):
    print("=" * 72)
    print("🌐 V31 REMOTE WORKER LAB")
    print(f"🧠 Central brain: {BRAIN_URL}")
    print(f"🌍 Worker region: {WORKER_REGION} | relay: {BRAIN_RELAY_URL}")
    print(f"🎮 Workers: {workers}")
    print("👀 Visible browsers: ON")
    print("=" * 72)
    # Verify the remote brain before launching browsers.
    health = await brain_request("/health")
    print(f"✅ Brain health: {health}")
    async with async_playwright() as p:
        tasks = [asyncio.create_task(supervised_train_worker(i + 1, p)) for i in range(workers)]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n🛑 Stopping remote workers...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    args = parse_mode()
    if getattr(args, "steps", None):
        os.environ["TRAIN_STEPS"] = str(max(1, args.steps))
        globals()["MAX_STEPS"] = max(1, args.steps)
    if getattr(args, "region", None):
        WORKER_REGION = _REGION_ALIASES.get(args.region.upper(), "US")
        if not os.getenv("BRAIN_RELAY_URL", "").strip():
            BRAIN_RELAY_URL = REGIONAL_RELAY_URLS[WORKER_REGION]
        if not os.getenv("BRAIN_RELAY_API_KEY", "").strip():
            BRAIN_RELAY_API_KEY = ""
            for _key_env in REGIONAL_RELAY_KEY_ENVS[WORKER_REGION]:
                _key_value = os.getenv(_key_env, "")
                if _key_value:
                    BRAIN_RELAY_API_KEY = _key_value
                    break
    train_worker._scheduled_mode = bool(getattr(args, "schedule", False))
    if args.brain:
        run_brain()
    elif args.auto:
        run_auto(max(1, args.cycles), max(1, min(args.workers, 32)))
    else:
        if args.eval:
            os.environ["AI_EVAL_MODE"] = "1"
        NUM_ENVS = max(1, min(args.workers, 32))
        # A remote URL means this process is workers-only. Localhost keeps the
        # convenient V12-style all-in-one mode.
        if BRAIN_URL.startswith("http://127.0.0.1") or BRAIN_URL.startswith("http://localhost"):
            asyncio.run(train_local())
        else:
            asyncio.run(train_remote(NUM_ENVS))
