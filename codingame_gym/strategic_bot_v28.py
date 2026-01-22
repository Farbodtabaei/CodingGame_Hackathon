"""\
STRATEGIC BOT V28 (standalone)
==============================

This file is intentionally self-contained for easy CodinGame submission.

V28 focus:
- Add an endgame "imminent block" layer: if the opponent's best predicted
    connection can complete within a couple turns, proactively place a cheap track
    cell on that path (within a small budget cap) before continuing our own build.

This does not guarantee a particular winrate; validate vs your seed suite.
"""

import json
import os
import sys
import traceback
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set
import heapq
import itertools


class Town:
    def __init__(self, town_id: int, x: int, y: int, desired_connections: List[int]):
        self.id = town_id
        self.x = x
        self.y = y
        self.desired = desired_connections


class StrategicBotV28:
    # --- Core rules ---
    POINTS_PER_TURN = 3
    INSTABILITY_TO_INK = 4
    TOTAL_TURNS = 100

    # --- Pathing cost model (copied from current V20) ---
    OPP_TRACK_TRAVERSAL_PENALTY = 0.7219363615068788
    NEUTRAL_TRACK_TRAVERSAL_PENALTY = 0.17647968616821225
    SELF_HARM_ALPHA = 1.1065620844486594

    # --- V21 prediction knobs ---
    PRED_TOP_K = 6
    PRED_WEIGHT_SCALE = 0.6806829106909927

    # Opponent route estimation penalties
    MY_TRACK_BLOCKING_PENALTY = 2.25
    NEUTRAL_TRACK_PENALTY = 0.45

    # Conservative override rules
    PRED_OVERRIDE_THRESHOLD = 8.042114928635115
    PRED_COMPARE_MARGIN = 0.6403548253201966
    PRED_OVERRIDE_BEHIND_FOR_2HIT = 159
    PRED_DISABLE_OVERRIDE_WHEN_AHEAD = 144
    PRED_EXTRA_MARGIN_IF_BASE_1HIT = 2.75
    PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND = 85

    # Only predict routes involving towns near existing opponent tracks
    OPP_TOWN_NEAR_DIST = 7

    # --- V25: Anti-dense (opponent densify blocking) ---
    ANTI_DENSE_ENABLED = True
    ANTI_DENSE_ALPHA = 0.5678600136896579
    ANTI_DENSE_MAX_CELL_COST = 3
    ANTI_DENSE_IGNORE_IF_AHEAD_BY = 141

    # Use leftover budget to deny opponent cheap densify cells.
    ANTI_DENSE_BLOCK_PLACEMENTS = 2
    ANTI_DENSE_BLOCK_MAX_INSTABILITY = 2
    ANTI_DENSE_POISON_REGION = True
    ANTI_DENSE_POISON_MAX_CELL_COST = 2
    ANTI_DENSE_POISON_MAX_DIST_TO_FRONTIER = 3

    # --- V25: Opponent-path choke placement ---
    CHOKE_ENABLED = True
    CHOKE_TOP_K_PATHS = 4
    CHOKE_MAX_PLACEMENTS = 1
    CHOKE_MAX_CELL_COST = 2
    CHOKE_MIN_DELTA_COST = 0.0
    CHOKE_DISABLE_WHEN_AHEAD = 140
    CHOKE_START_TURN = 2

    # --- Densify (V24 baseline) ---
    DENSE_MAX_EXTRA_PLACEMENTS = 2
    DENSE_MAX_INSTABILITY = 2
    DENSE_AVOID_OPP_REGIONS = True
    # --- V25: Targeted spend along best connection path ---
    SUPPORT_ENABLED = True
    SUPPORT_MAX_PLACEMENTS = 2
    SUPPORT_MAX_CELL_COST = 2
    SUPPORT_MAX_INSTABILITY = 2

    # --- V25: Targeted densify fallback (non-random) ---
    TARGETED_DENSE_ENABLED = True
    TARGETED_DENSE_MAX_PLACEMENTS = 1
    TARGETED_DENSE_MAX_CELL_COST = 2
    TARGETED_DENSE_MAX_INSTABILITY = 2
    TARGETED_DENSE_MAX_DIST_TO_MY_TRACK = 3

    # --- V25: When to densify (avoid over-investing) ---
    DENSIFY_ONLY_WHEN_AHEAD_BY = 30
    DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ = 53.76476208195873
    DENSIFY_ONLY_WHEN_BEST_NEW_EV_LEQ = 38.29057469255342

    # --- V27: Tunable path switching ---
    # If a new best connection EV exceeds continue EV by this factor, switch.
    SWITCH_THRESHOLD = 2.325591280119858

    # --- V27: Opt-in pre-choke heuristic ---
    # V25 only chokes using leftover budget; this allows spending some budget
    # up-front to block high-value opponent paths.
    PRE_CHOKE_ENABLED = False
    PRE_CHOKE_THREAT_MIN = 70.0
    PRE_CHOKE_MAX_BUDGET = 1

    # --- V27: Opt-in high-confidence choke (safer than PRE_CHOKE) ---
    # Only spend budget before building if there is an exceptionally strong
    # choke placement available.
    HIGH_CONF_CHOKE_ENABLED = False
    HIGH_CONF_CHOKE_TURN_MIN = 3
    HIGH_CONF_CHOKE_THREAT_MIN = 80.0
    HIGH_CONF_CHOKE_SCORE_DELTA_MAX = 35
    HIGH_CONF_CHOKE_MAX_BUDGET = 1
    HIGH_CONF_CHOKE_MIN_CELL_SCORE = 240.0

    # --- V27: Opt-in connection tie-breaker (near-tie games) ---
    # When multiple connection candidates have similar EV, prefer options that
    # complete faster and gift fewer pre-owned cells to the opponent.
    # Disabled by default to preserve baseline behavior.
    CONN_TIEBREAK_ENABLED = True
    CONN_TIEBREAK_TURN_MIN = 6
    CONN_TIEBREAK_SCORE_DELTA_MAX = 25
    CONN_TIEBREAK_EV_WINDOW = 25.0
    CONN_TIEBREAK_W_FAST = 1.0
    CONN_TIEBREAK_W_GIFT = 0.25

    # --- V27: Opt-in late-game tempo mode ---
    # In late, close games, prefer connections that complete quickly even if their
    # EV is slightly lower, to convert tie-heavy matchups.
    # NOTE: EV scale here is "tens"+, not [0,1]. Keep window/bonus on that scale.
    TEMPO_ENABLED = True
    TEMPO_TURN_MIN = 72
    TEMPO_SCORE_DELTA_MAX = 120
    TEMPO_EV_WINDOW = 22.0
    TEMPO_TOP_K = 16
    TEMPO_MAX_TURNS_TO_COMPLETE = 4
    TEMPO_BONUS = 45.0
    TEMPO_W_FAST = 2.0

    # --- V28: Early-game tempo (loss-driven) ---
    # Logged V28 losses vs V25 frequently fall behind very early (turn ~15-20).
    # When we are significantly behind early, bias towards finishing a connection
    # quickly even if EV is a bit lower.
    V28_EARLY_TEMPO_ENABLED = True
    V28_EARLY_TEMPO_TURN_MAX = 30
    V28_EARLY_TEMPO_SCORE_DELTA_MIN = 0
    V28_EARLY_TEMPO_EV_WINDOW = 35.0
    V28_EARLY_TEMPO_TOP_K = 16
    V28_EARLY_TEMPO_MAX_TURNS_TO_COMPLETE = 5
    V28_EARLY_TEMPO_BONUS = 100.0
    V28_EARLY_TEMPO_W_FAST = 3.0

    # V28: endgame late densify (tie-break points).
    V28_LATE_DENSIFY_ENABLED = True
    V28_LATE_DENSIFY_TURN_MIN = 60
    V28_LATE_DENSIFY_SCORE_DELTA_MAX = 80

    # --- V28: Endgame race / imminent-block layers ---
    V28_ENDGAME_ENABLED = True
    V28_ENDGAME_TURN_MIN = 78
    V28_ENDGAME_CLOSE_SCORE_DELTA_MAX = 60

    V28_IMMINENT_BLOCK_ENABLED = True
    V28_IMMINENT_BLOCK_MAX_TURNS = 2
    V28_IMMINENT_BLOCK_MIN_PATH_VALUE = 70.0
    V28_IMMINENT_BLOCK_MAX_BUDGET = 2
    V28_IMMINENT_BLOCK_MAX_CELL_COST = 2
    V28_IMMINENT_BLOCK_MIN_CELL_SCORE = 160.0

    # V28: Midgame imminent-block (loss-driven)
    # Some V25 wins come from midgame large completes; allow a very selective
    # pre-block earlier than the endgame, but only when opponent's top predicted
    # path is both high-value and imminently completable.
    V28_MIDGAME_BLOCK_ENABLED = False
    V28_MIDGAME_BLOCK_TURN_MIN = 26
    V28_MIDGAME_BLOCK_TURN_MAX = 72
    V28_MIDGAME_BLOCK_SCORE_DELTA_MAX = 90
    V28_MIDGAME_BLOCK_MIN_PATH_VALUE = 180.0
    V28_MIDGAME_BLOCK_MAX_TURNS = 2
    V28_MIDGAME_BLOCK_MAX_BUDGET = 1

    # --- V28: Endgame disrupt-focus (major logic change) ---
    # In late, close games (the V25 tie-heavy regime), pick a high-impact region
    # and keep disrupting it until it inks (or becomes invalid).
    V28_DISRUPT_FOCUS_ENABLED = True
    V28_DISRUPT_FOCUS_TURN_MIN = 70
    V28_DISRUPT_FOCUS_SCORE_DELTA_MAX = 120
    V28_DISRUPT_ALLOW_2HIT_TURNS_LEFT_MIN = 16

    # Scoring weights for endgame disrupt (predicted opponent paths matter!).
    V28_DISRUPT_W_PRED = 1.25
    V28_DISRUPT_W_OPP_W = 1.00
    V28_DISRUPT_W_ANTI_DENSE = 0.50
    V28_DISRUPT_W_NEU_W = 0.15
    V28_DISRUPT_MIN_BENEFIT = 0.75

    # --- V27: Opt-in "panic disrupt" heuristic (counter-tuning knob) ---
    # Disabled by default to preserve baseline behavior.
    DISRUPT_PANIC_ENABLED = False
    DISRUPT_PANIC_BEHIND_BY_MIN = 40
    DISRUPT_PANIC_THREAT_MIN = 70.0
    DISRUPT_PANIC_ALLOW_2HIT = True
    DISRUPT_PANIC_TOP_K = 10
    DISRUPT_PANIC_MARGIN = 0.15
    # Scoring weights (tuned via CEM)
    DISRUPT_PANIC_W_PRED = 1.40
    DISRUPT_PANIC_W_OPP_W = 0.65
    DISRUPT_PANIC_W_ANTI_DENSE = 0.35

    # --- V25 experimental densify knobs (currently disabled / inert) ---
    DENSE_BASE_MAX_INSTABILITY = 2
    DENSE_ALLOW_INSTABILITY_IN_OWN_REGION = 2
    DENSE_TOWN_ADJ_BONUS = 0.0
    DENSE_NEAR_TOWN_RADIUS = 6
    DENSE_NEAR_TOWN_BONUS_SCALE = 0.0
    DENSE_SKIP_WHEN_ACTIVE_BUILD = False

    # --- V27: Optional disrupt model (env-var gated) ---
    _DISRUPT_MODEL_FEATURES = (
        "cand_score",
        "cand_opp_w",
        "cand_neu_w",
        "cand_my_w",
        "cand_pred_opp_w",
        "cand_hits_to_ink",
        "cand_turns_left",
        "cand_anti_dense",
        "cand_instability",
        "cand_avoid",
        "cand_in_my_track",
        "cand_in_opp_track",
        "g_opp_threat",
        "g_score_delta",
        "g_continue_ev",
        "g_best_new_ev",
        "g_turn",
    )


    def __init__(self):
        self.my_id = 0
        self.opp_id = 1
        self.width = 0
        self.height = 0
        self.towns: List[Town] = []
        self.town_by_id: Dict[int, Town] = {}
        self.region_ids: List[List[int]] = []
        self.cell_types: List[List[int]] = []

        self.turn = 0
        self.my_score = 0
        self.opp_score = 0

        self.completed_connections: Set[Tuple[int, int]] = set()
        self.region_instability: Dict[int, int] = defaultdict(int)
        self.region_inked: Set[int] = set()
        self.cell_tracks: Dict[Tuple[int, int], int] = {}
        self.town_regions: Set[int] = set()
        self.town_cells: Set[Tuple[int, int]] = set()

        self.active_build: Optional[Tuple[int, int]] = None
        self.active_build_path: Optional[List[Tuple[int, int]]] = None
        # If a build becomes blocked (e.g., inked region splits the path) we should
        # re-path quickly and avoid placing disconnected "dead" segments.
        self._active_build_stuck_turns: int = 0

        self.my_score_history: List[int] = []
        self.opp_score_history: List[int] = []

        self.detected_opp_bias: Optional[str] = 'high'
        self.my_bias_multiplier: float = -5.0

        self.opp_track_regions: Set[int] = set()
        self.my_track_regions: Set[int] = set()

        self.active_weight_by_region_my: Dict[int, float] = defaultdict(float)
        self.active_weight_by_region_opp: Dict[int, float] = defaultdict(float)
        self.active_weight_by_region_neutral: Dict[int, float] = defaultdict(float)

        self.hub_town_id: Optional[int] = None
        self.town_adjacent_cells: Set[Tuple[int, int]] = set()

        self.path_cache: Dict[Tuple, Optional[List[Tuple[int, int]]]] = {}

        self._pred_opp_scored_paths_turn: int = -1
        self._pred_opp_scored_paths: List[Tuple[float, List[Tuple[int, int]]]] = []

        # V28 endgame disrupt focus.
        self._v28_focus_disrupt_region: Optional[int] = None
        self._v28_debug: bool = bool(os.environ.get("V28_DEBUG"))

        # Structured per-turn trace (opt-in): set env var V28_TRACE=1.
        self._trace_enabled: bool = bool(int(os.environ.get("V28_TRACE") or 0))
        self._trace_last: Optional[dict] = None

        # --- V27 dataset/oracle hooks (opt-in) ---
        self._dataset_enabled: bool = bool(os.environ.get("V27_DATASET"))
        self._dataset_file: str = str(os.environ.get("V27_DATASET_FILE") or "").strip()
        if not self._dataset_file:
            self._dataset_enabled = False
        self._dataset_kind: str = str(os.environ.get("V27_DATASET_KIND") or "disrupt").strip() or "disrupt"
        self._dataset_seed: str = str(os.environ.get("V27_DATASET_SEED") or "").strip()
        self._dataset_as: str = str(os.environ.get("V27_DATASET_AS") or "").strip()
        try:
            self._dataset_max_turn: int = int(os.environ.get("V27_DATASET_MAX_TURN") or 0)
        except Exception:
            self._dataset_max_turn = 0

        try:
            self._force_disrupt_turn: int = int(os.environ.get("V27_FORCE_DISRUPT_TURN") or -1)
        except Exception:
            self._force_disrupt_turn = -1
        try:
            self._force_disrupt_region: int = int(os.environ.get("V27_FORCE_DISRUPT_REGION") or -999999)
        except Exception:
            self._force_disrupt_region = -999999

        # Optional disrupt model. Default is OFF.
        self._disrupt_model_file: str = str(os.environ.get("V27_DISRUPT_MODEL_FILE") or "").strip()
        self._disrupt_model_enabled: bool = bool(os.environ.get("V27_DISRUPT_MODEL")) and bool(self._disrupt_model_file)
        try:
            self._disrupt_model_margin: float = float(os.environ.get("V27_DISRUPT_MODEL_MARGIN") or 0.0)
        except Exception:
            self._disrupt_model_margin = 0.0
        self._disrupt_model_w: Optional[List[float]] = None
        self._disrupt_model_mean: Optional[List[float]] = None
        self._disrupt_model_std: Optional[List[float]] = None

        if self._disrupt_model_enabled:
            self._load_disrupt_model()

    def _load_disrupt_model(self) -> None:
        try:
            path = self._disrupt_model_file
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            feats = tuple(obj.get("feature_names") or ())
            if feats != tuple(self._DISRUPT_MODEL_FEATURES):
                return
            w = obj.get("w")
            mean = obj.get("mean")
            std = obj.get("std")
            if not (isinstance(w, list) and isinstance(mean, list) and isinstance(std, list)):
                return
            if not (len(w) == len(mean) == len(std) == len(self._DISRUPT_MODEL_FEATURES)):
                return
            self._disrupt_model_w = [float(x) for x in w]
            self._disrupt_model_mean = [float(x) for x in mean]
            self._disrupt_model_std = [float(x) if float(x) != 0.0 else 1.0 for x in std]
        except Exception:
            # Never let model loading affect gameplay.
            self._disrupt_model_w = None
            self._disrupt_model_mean = None
            self._disrupt_model_std = None

    def _disrupt_model_score(self, cand: dict, *, opp_threat: float, score_delta: int, continue_ev: float, best_new_ev: float) -> float:
        if not (self._disrupt_model_w and self._disrupt_model_mean and self._disrupt_model_std):
            return -1e18

        # Candidate-local
        x = [
            float(cand.get("score", -1e9)),
            float(cand.get("opp_w", 0.0)),
            float(cand.get("neu_w", 0.0)),
            float(cand.get("my_w", 0.0)),
            float(cand.get("pred_opp_w", 0.0)),
            float(cand.get("hits_to_ink", 0)),
            float(cand.get("turns_left", 0)),
            float(cand.get("anti_dense", 0.0)),
            float(cand.get("instability", 0)),
            1.0 if bool(cand.get("avoid")) else 0.0,
            1.0 if bool(cand.get("in_my_track")) else 0.0,
            1.0 if bool(cand.get("in_opp_track")) else 0.0,
            # Sample-global
            float(opp_threat),
            float(score_delta),
            float(continue_ev),
            float(best_new_ev),
            float(self.turn),
        ]

        w = self._disrupt_model_w
        mean = self._disrupt_model_mean
        std = self._disrupt_model_std
        s = 0.0
        for i in range(len(w)):
            s += w[i] * ((x[i] - mean[i]) / std[i])
        return float(s)

    def _softmax_probs(self, logits: List[float]) -> List[float]:
        if not logits:
            return []
        m = max(logits)
        exps = [pow(2.718281828459045, (v - m)) for v in logits]
        s = sum(exps)
        if s <= 0.0:
            return [1.0 / float(len(logits)) for _ in logits]
        return [v / s for v in exps]

    def _get_disruption_target_model(
        self,
        *,
        best_next_path: Optional[List[Tuple[int, int]]],
        opp_threat: float,
        score_delta: int,
        continue_ev: float,
        best_new_ev: float,
        baseline_target: Optional[int],
    ) -> Optional[int]:
        # Keep conservative behavior when ahead.
        if score_delta >= self.PRED_DISABLE_OVERRIDE_WHEN_AHEAD:
            return None
        if not (self._disrupt_model_w and self._disrupt_model_mean and self._disrupt_model_std):
            return None

        cands = self._get_disruption_candidates_for_dataset(best_next_path, limit=6)
        if not cands:
            return None

        # Filter out avoid regions (same logic as baseline override).
        filtered = [c for c in cands if not bool(c.get("avoid"))]
        if not filtered:
            return None

        logits: List[float] = []
        rids: List[int] = []
        base_idx: Optional[int] = None
        for c in filtered:
            rid = c.get("rid")
            if not isinstance(rid, int):
                continue
            if rid in self.region_inked or rid in self.town_regions:
                continue
            logit = self._disrupt_model_score(c, opp_threat=opp_threat, score_delta=score_delta, continue_ev=continue_ev, best_new_ev=best_new_ev)
            rids.append(int(rid))
            logits.append(float(logit))
            if baseline_target is not None and int(rid) == int(baseline_target):
                base_idx = len(rids) - 1

        if not rids:
            return None

        # If baseline isn't even in the model's comparable set, don't override.
        if base_idx is None:
            return None

        probs = self._softmax_probs(logits)
        best_i = 0
        best_p = -1.0
        for i, p in enumerate(probs):
            if p > best_p:
                best_p = float(p)
                best_i = int(i)

        # No change.
        if best_i == int(base_idx):
            return None

        # Only override baseline if the model is confidently better.
        base_p = float(probs[base_idx])
        if (best_p - base_p) < float(self._disrupt_model_margin):
            return None

        return int(rids[best_i])

    def _dataset_write(self, obj: dict) -> None:
        if not self._dataset_enabled:
            return
        if self._dataset_max_turn and self.turn > self._dataset_max_turn:
            return
        try:
            with open(self._dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
        except Exception:
            # Never let logging affect gameplay.
            return

    def _disrupt_avoid_set(self, best_next_path: Optional[List[Tuple[int, int]]]) -> Set[int]:
        avoid: Set[int] = set()
        if self.active_build_path:
            for x, y in self.active_build_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)
        if best_next_path:
            for x, y in best_next_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)
        return avoid

    def _disrupt_candidate_features(self, region_id: int, *, turns_left: int, avoid: Set[int], pred_opp: Dict[int, float], anti_dense: Dict[int, float]) -> dict:
        opp_w = float(self.active_weight_by_region_opp.get(region_id, 0.0))
        my_w = float(self.active_weight_by_region_my.get(region_id, 0.0))
        neu_w = float(self.active_weight_by_region_neutral.get(region_id, 0.0))
        instability = int(self.region_instability.get(region_id, 0) or 0)
        hits_to_ink = int(self.INSTABILITY_TO_INK - instability)
        anti_dense_pot = float(anti_dense.get(region_id, 0.0)) if anti_dense else 0.0
        pred_w = float(pred_opp.get(region_id, 0.0)) if pred_opp else 0.0

        base = (opp_w - self.SELF_HARM_ALPHA * my_w + 0.15 * neu_w)
        if anti_dense_pot:
            base += anti_dense_pot * float(self.ANTI_DENSE_ALPHA)

        score = -1e9
        if base > 0 and hits_to_ink > 0:
            score = (base * float(turns_left)) / float(max(hits_to_ink, 1))
            if hits_to_ink == 1:
                score *= 2.0
            elif hits_to_ink >= 2:
                score *= 0.65

        return {
            "rid": int(region_id),
            "score": float(score),
            "opp_w": float(opp_w),
            "my_w": float(my_w),
            "neu_w": float(neu_w),
            "instability": int(instability),
            "hits_to_ink": int(hits_to_ink),
            "turns_left": int(turns_left),
            "anti_dense": float(anti_dense_pot),
            "pred_opp_w": float(pred_w),
            "in_opp_track": bool(region_id in self.opp_track_regions),
            "in_my_track": bool(region_id in self.my_track_regions),
            "avoid": bool(region_id in avoid),
        }

    def _v28_disrupt_focus_target(
        self,
        *,
        best_next_path: Optional[List[Tuple[int, int]]],
        opp_threat: float,
        score_delta: int,
        continue_ev: float,
        best_new_ev: float,
    ) -> Optional[int]:
        if not bool(self.V28_DISRUPT_FOCUS_ENABLED):
            return None

        if int(self.turn) < int(self.V28_DISRUPT_FOCUS_TURN_MIN):
            return None

        if abs(int(score_delta)) > int(self.V28_DISRUPT_FOCUS_SCORE_DELTA_MAX):
            return None

        turns_left = int(self.TOTAL_TURNS - self.turn)
        if turns_left <= 0:
            return None

        avoid = self._disrupt_avoid_set(best_next_path)

        # If we already chose a focus region, keep hitting it until it inks.
        if self._v28_focus_disrupt_region is not None:
            rid = int(self._v28_focus_disrupt_region)
            if rid in self.region_inked or rid in self.town_regions or rid in avoid:
                self._v28_focus_disrupt_region = None
            else:
                instability = int(self.region_instability.get(rid, 0) or 0)
                if instability >= int(self.INSTABILITY_TO_INK):
                    self._v28_focus_disrupt_region = None
                else:
                    # Continue focusing.
                    return int(rid)

        # Choose a new focus.
        allow_2hit = turns_left >= int(self.V28_DISRUPT_ALLOW_2HIT_TURNS_LEFT_MIN)
        cands = self._get_disruption_candidates_for_dataset(best_next_path, limit=40)
        if not cands:
            return None

        best_rid: Optional[int] = None
        best_score = -1e18

        for c in cands:
            rid = c.get("rid")
            if not isinstance(rid, int):
                continue
            if bool(c.get("avoid")):
                continue
            if bool(c.get("in_my_track")):
                continue
            if int(rid) in self.region_inked or int(rid) in self.town_regions:
                continue

            hits = int(c.get("hits_to_ink", 0) or 0)
            if hits <= 0:
                continue
            if hits != 1 and not (allow_2hit and hits == 2):
                continue

            pred_w = float(c.get("pred_opp_w", 0.0) or 0.0)
            opp_w = float(c.get("opp_w", 0.0) or 0.0)
            my_w = float(c.get("my_w", 0.0) or 0.0)
            neu_w = float(c.get("neu_w", 0.0) or 0.0)
            anti_dense = float(c.get("anti_dense", 0.0) or 0.0)

            benefit = (
                float(self.V28_DISRUPT_W_PRED) * pred_w
                + float(self.V28_DISRUPT_W_OPP_W) * opp_w
                + float(self.V28_DISRUPT_W_ANTI_DENSE) * anti_dense
                + float(self.V28_DISRUPT_W_NEU_W) * neu_w
                - float(self.SELF_HARM_ALPHA) * my_w
            )

            if benefit < float(self.V28_DISRUPT_MIN_BENEFIT):
                continue

            s = float(benefit) * float(turns_left) / float(max(1, hits))
            # Prefer 1-hit inks heavily, but allow 2-hit when we have time.
            if hits == 1:
                s *= 2.25
            else:
                s *= 0.9

            # Mild bias: if opponent threat is high and we are not ahead, boost.
            if float(opp_threat) > float(self.DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ) and int(score_delta) <= 0:
                s *= 1.10

            if s > best_score:
                best_score = float(s)
                best_rid = int(rid)

        if best_rid is None:
            return None

        # Persist focus if it's not an immediate 1-hit ink.
        hits = int(self.INSTABILITY_TO_INK - int(self.region_instability.get(best_rid, 0) or 0))
        if hits >= 2:
            self._v28_focus_disrupt_region = int(best_rid)
        else:
            self._v28_focus_disrupt_region = None

        if self._v28_debug:
            try:
                print(f"V28 disrupt_focus pick rid={best_rid} hits={hits} turn={self.turn} sd={score_delta} opp_threat={opp_threat:.1f} continue_ev={continue_ev:.1f} best_new_ev={best_new_ev:.1f}", file=sys.stderr)
            except Exception:
                pass

        return int(best_rid)

    def _get_disruption_candidates_for_dataset(self, best_next_path: Optional[List[Tuple[int, int]]], limit: int = 24) -> List[dict]:
        turns_left = int(self.TOTAL_TURNS - self.turn)
        avoid = self._disrupt_avoid_set(best_next_path)
        pred_opp = self._predict_opponent_region_weights()
        anti_dense = self._opp_dense_potential_by_region() if self.ANTI_DENSE_ENABLED else {}

        region_pool: Set[int] = set()
        region_pool.update(self.active_weight_by_region_opp.keys())
        region_pool.update(self.active_weight_by_region_my.keys())
        region_pool.update(self.active_weight_by_region_neutral.keys())
        region_pool.update(self.opp_track_regions)
        region_pool.update(pred_opp.keys())
        region_pool.update(anti_dense.keys())

        out: List[dict] = []
        for rid in region_pool:
            if rid in self.region_inked or rid in self.town_regions:
                continue
            instability = int(self.region_instability.get(rid, 0) or 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue
            out.append(
                self._disrupt_candidate_features(
                    int(rid),
                    turns_left=turns_left,
                    avoid=avoid,
                    pred_opp=pred_opp,
                    anti_dense=anti_dense,
                )
            )

        out.sort(key=lambda c: float(c.get("score", -1e18)), reverse=True)
        return out[: int(limit)]

    # -------------------- Input --------------------

    def read_init(self):
        lines = []
        lines.append(input())
        width = int(input())
        height = int(input())
        lines.append(str(width))
        lines.append(str(height))
        for _ in range(height * width):
            lines.append(input())
        town_count = int(input())
        lines.append(str(town_count))
        for _ in range(town_count):
            lines.append(input())
        self.read_init_lines(lines)

    def read_init_lines(self, lines: List[str]):
        it = iter(lines)
        self.my_id = int(next(it))
        self.opp_id = 1 - self.my_id
        self.width = int(next(it))
        self.height = int(next(it))

        self.region_ids = []
        self.cell_types = []
        for _ in range(self.height):
            region_row = []
            type_row = []
            for _ in range(self.width):
                try:
                    inputs = next(it).split()
                    region_id = int(inputs[0])
                    cell_type = int(inputs[1]) if len(inputs) > 1 else 0
                    region_row.append(region_id)
                    type_row.append(cell_type)
                except (ValueError, IndexError, StopIteration):
                    region_row.append(-1)
                    type_row.append(0)
            self.region_ids.append(region_row)
            self.cell_types.append(type_row)

        try:
            town_count = int(next(it))
        except StopIteration:
            town_count = 0

        self.towns = []
        self.town_by_id = {}
        self.town_cells.clear()

        for _ in range(town_count):
            try:
                parts = next(it).split()
            except StopIteration:
                break
            if len(parts) < 3:
                continue

            town_id = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            desired_str = parts[3] if len(parts) > 3 else "-"

            desired = []
            if desired_str not in ("", "-", "x", "X"):
                for d in desired_str.split(","):
                    d = d.strip()
                    if d.isdigit():
                        desired.append(int(d))

            t = Town(town_id, x, y, desired)
            self.towns.append(t)
            self.town_by_id[town_id] = t
            self.town_cells.add((x, y))

        self.town_regions = set()
        for t in self.towns:
            if 0 <= t.y < self.height and 0 <= t.x < self.width:
                self.town_regions.add(self.region_ids[t.y][t.x])

        indeg = defaultdict(int)
        for t in self.towns:
            for d in t.desired:
                indeg[d] += 1
        if self.towns:
            self.hub_town_id = max(self.towns, key=lambda t: (len(t.desired) + indeg[t.id], -t.id)).id
        else:
            self.hub_town_id = None

        self.town_adjacent_cells.clear()
        for t in self.towns:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = t.x + dx, t.y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.town_cells:
                        self.town_adjacent_cells.add((nx, ny))

        self.turn = 0
        self.my_score = 0
        self.opp_score = 0
        self.active_build = None
        self.active_build_path = None
        self.path_cache.clear()
        self.my_score_history.clear()
        self.opp_score_history.clear()

        self._pred_opp_scored_paths_turn = -1
        self._pred_opp_scored_paths = []

    def read_turn(self):
        lines = [input(), input()]
        for _ in range(self.height * self.width):
            lines.append(input())
        self.read_turn_lines(lines)

    def read_turn_lines(self, lines: List[str]):
        it = iter(lines)
        try:
            self.my_score = int(next(it))
            self.opp_score = int(next(it))
        except StopIteration:
            return

        self.turn += 1
        self.my_score_history.append(self.my_score)
        self.opp_score_history.append(self.opp_score)

        self.completed_connections.clear()
        self.region_instability.clear()
        self.region_inked.clear()
        self.cell_tracks.clear()
        self.opp_track_regions.clear()
        self.my_track_regions.clear()

        self.active_weight_by_region_my.clear()
        self.active_weight_by_region_opp.clear()
        self.active_weight_by_region_neutral.clear()

        for y in range(self.height):
            for x in range(self.width):
                try:
                    inputs = next(it).split()
                    if len(inputs) < 3:
                        continue
                    track_owner = int(inputs[0])
                    instability = int(inputs[1])
                    inked = inputs[2] != "0"
                    connections_str = inputs[3] if len(inputs) > 3 else "x"
                except (StopIteration, IndexError, ValueError):
                    continue

                region_id = self.region_ids[y][x]
                if region_id == -1:
                    continue

                if track_owner != -1:
                    self.cell_tracks[(x, y)] = track_owner
                    if track_owner == self.opp_id:
                        self.opp_track_regions.add(region_id)
                    elif track_owner == self.my_id:
                        self.my_track_regions.add(region_id)

                self.region_instability[region_id] = max(self.region_instability[region_id], instability)
                if inked:
                    self.region_inked.add(region_id)

                if connections_str not in ("x", "-", ""):
                    pairs = []
                    for conn in connections_str.split(','):
                        conn = conn.strip()
                        if '-' not in conn:
                            continue
                        try:
                            a, b = conn.split('-')
                            t1, t2 = int(a), int(b)
                            pairs.append((t1, t2))
                            self.completed_connections.add((min(t1, t2), max(t1, t2)))
                        except ValueError:
                            continue

                    if pairs and region_id not in self.town_regions and region_id not in self.region_inked:
                        w = float(len(pairs))
                        if track_owner == self.my_id:
                            self.active_weight_by_region_my[region_id] += w
                        elif track_owner == self.opp_id:
                            self.active_weight_by_region_opp[region_id] += w
                        elif track_owner == 2:
                            self.active_weight_by_region_neutral[region_id] += w

        self._detect_opponent_behavior()

        if self.turn % 10 == 0:
            self.path_cache.clear()

        if self.active_build:
            conn_id = (min(self.active_build[0], self.active_build[1]), max(self.active_build[0], self.active_build[1]))
            if conn_id in self.completed_connections:
                self.active_build = None
                self.active_build_path = None

    # -------------------- Placement rules --------------------

    def can_place_track(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if (x, y) in self.town_cells:
            return False

        region_id = self.region_ids[y][x]
        if region_id == -1:
            return False
        if region_id in self.region_inked:
            return False

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return False

        if (x, y) in self.cell_tracks:
            return False

        return True

    def _is_existing_connection_cell(self, x: int, y: int) -> bool:
        return (x, y) in self.town_cells or (x, y) in self.cell_tracks

    # -------------------- Pathfinding --------------------

    def _empty_cell_place_cost(self, x: int, y: int) -> int:
        return int(self.cell_types[y][x] + 1)

    def _get_cell_cost(self, x: int, y: int) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')

        if (x, y) in self.town_cells:
            return 0.0

        region_id = self.region_ids[y][x]
        if region_id == -1 or region_id in self.region_inked:
            return float('inf')

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return float('inf')

        if (x, y) in self.cell_tracks:
            owner = self.cell_tracks[(x, y)]
            if owner == self.my_id:
                return 0.0
            if owner == self.opp_id:
                return float(self.OPP_TRACK_TRAVERSAL_PENALTY)
            return float(self.NEUTRAL_TRACK_TRAVERSAL_PENALTY)

        base_cost = self._empty_cell_place_cost(x, y)
        if base_cost > self.POINTS_PER_TURN:
            return float('inf')

        instability_penalty = instability * 2
        return float(base_cost + instability_penalty)

    def find_path(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        max_nodes: int = 1400,
        use_cache: bool = True,
    ) -> Optional[List[Tuple[int, int]]]:
        cache_key = (x0, y0, x1, y1)
        if use_cache and cache_key in self.path_cache:
            return self.path_cache[cache_key]

        def heuristic(x, y):
            return abs(x - x1) + abs(y - y1)

        order = itertools.count()
        open_set = [(0.0, 0.0, next(order), x0, y0)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(x0, y0): 0.0}
        nodes_expanded = 0

        while open_set and nodes_expanded < max_nodes:
            _, g, _, x, y = heapq.heappop(open_set)
            nodes_expanded += 1

            if x == x1 and y == y1:
                path = []
                cur = (x, y)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((x0, y0))
                result = list(reversed(path))
                if use_cache:
                    self.path_cache[cache_key] = result
                return result

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                step_cost = self._get_cell_cost(nx, ny)
                if step_cost == float('inf'):
                    continue

                tentative_g = g + step_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, tentative_g, next(order), nx, ny))

        if use_cache:
            self.path_cache[cache_key] = None
        return None

    def find_shortest_path_bfs(self, x0: int, y0: int, x1: int, y1: int) -> Optional[List[Tuple[int, int]]]:
        start = (x0, y0)
        goal = (x1, y1)
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                region_id = self.region_ids[ny][nx]
                if region_id in self.region_inked:
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    # -------------------- Cost / EV --------------------

    def calc_path_cost(self, path: List[Tuple[int, int]]) -> int:
        total = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                continue
            total += self._empty_cell_place_cost(x, y)
        return total

    def calc_remaining_cost(self, path: List[Tuple[int, int]]) -> int:
        return self.calc_path_cost(path)

    def _count_track_owners_on_path(self, path: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        my_cells = 0
        opp_cells = 0
        neutral_cells = 0
        empty_cells = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                owner = self.cell_tracks[(x, y)]
                if owner == self.my_id:
                    my_cells += 1
                elif owner == self.opp_id:
                    opp_cells += 1
                elif owner == 2:
                    neutral_cells += 1
            else:
                empty_cells += 1
        return my_cells, opp_cells, neutral_cells, empty_cells

    def calc_connection_ev(self, path: List[Tuple[int, int]], from_id: int = 0, to_id: int = 0) -> float:
        if not path:
            return 0.0

        cost = self.calc_path_cost(path)
        if cost <= 0 or cost >= 10**8:
            return 0.0

        turns_to_complete = (cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        my_cells, opp_cells, neutral_cells, empty_cells = self._count_track_owners_on_path(path)

        expected_points = empty_cells * scoring_turns

        speed_multiplier = 1.5 if cost <= 10 else (1.2 if cost <= 17 else 0.7568338133671997)
        weighted_points = expected_points * speed_multiplier
        efficiency = weighted_points / max(cost, 1)

        town_id_bonus = 0.0
        if from_id > 0 and to_id > 0:
            base_bonus = (from_id + to_id) * 53.71401146671039
            town_id_bonus = base_bonus * self.my_bias_multiplier

        regions_in_path = set(self.region_ids[y][x] for x, y in path if self.region_ids[y][x] != -1)
        shared_regions = regions_in_path & self.my_track_regions
        consolidation_bonus = float(len(shared_regions) * 26.88965731068639)

        overlap_bonus = float(my_cells * 4)

        junction_hits = sum(1 for x, y in path if (x, y) in self.town_adjacent_cells)
        junction_bonus = float(junction_hits * 2.164151903532464)

        hub_bonus = 0.0
        if self.hub_town_id is not None and (from_id == self.hub_town_id or to_id == self.hub_town_id):
            hub_bonus = 35.0

        gifting_penalty = float((opp_cells * 7 + neutral_cells * 2) * speed_multiplier)

        return weighted_points + efficiency * 100 + town_id_bonus + consolidation_bonus + overlap_bonus + junction_bonus + hub_bonus - gifting_penalty

    def calc_continue_ev(self) -> float:
        if not self.active_build or not self.active_build_path:
            return 0.0

        path = self.active_build_path
        blocked = sum(1 for x, y in path if self.region_ids[y][x] in self.region_inked)
        if blocked > len(path) * 0.5:
            return 0.0

        remaining_cost = self.calc_remaining_cost(path)
        turns_to_complete = (remaining_cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN

        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        _, _, _, empty_cells = self._count_track_owners_on_path(path)
        expected_points = empty_cells * scoring_turns

        my_cells, _, _, _ = self._count_track_owners_on_path(path)
        progress_ratio = my_cells / max(len(path), 1)
        sunk_cost_bonus = progress_ratio * 50

        return expected_points + sunk_cost_bonus

    # -------------------- Build execution --------------------

    def _advance_from_start(self, path: List[Tuple[int, int]], left: int, right: int) -> int:
        while left + 1 <= right:
            x, y = path[left + 1]
            if self._is_existing_connection_cell(x, y):
                left += 1
                continue
            break
        return left

    def _advance_from_end(self, path: List[Tuple[int, int]], left: int, right: int) -> int:
        while right - 1 >= left:
            x, y = path[right - 1]
            if self._is_existing_connection_cell(x, y):
                right -= 1
                continue
            break
        return right

    def _path_has_unplaceable_gap(self, path: List[Tuple[int, int]], left: int, right: int) -> bool:
        # True if the remaining path segment contains an empty cell that we can never place
        # (e.g., it got inked). Existing town/track cells are fine.
        if left < 0 or right >= len(path) or left >= right:
            return False
        for i in range(left + 1, right):
            x, y = path[i]
            if self._is_existing_connection_cell(x, y):
                continue
            if not self.can_place_track(x, y):
                return True
        return False

    def build_path_two_ended_stable(
        self,
        from_town: Town,
        to_town: Town,
        budget_override: Optional[int] = None,
        path_override: Optional[List[Tuple[int, int]]] = None,
    ) -> List[str]:
        actions: List[str] = []
        if budget_override is None:
            budget = self.POINTS_PER_TURN
        else:
            budget = max(0, min(self.POINTS_PER_TURN, int(budget_override)))

        path = path_override if path_override else None
        if not path or len(path) < 2:
            path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
        if not path or len(path) < 2:
            return actions

        left = 0
        right = len(path) - 1
        left = self._advance_from_start(path, left, right)
        right = self._advance_from_end(path, left, right)

        while budget > 0:
            left = self._advance_from_start(path, left, right)
            right = self._advance_from_end(path, left, right)

            if left >= right - 1:
                break

            # If the current cached path is now impossible (e.g., inked cell inside),
            # do NOT keep extending only one end (creates disconnected dead tracks).
            if self._path_has_unplaceable_gap(path, left, right):
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                if not new_path or new_path == path:
                    break
                path = new_path
                if self.active_build and self.active_build_path:
                    self.active_build_path = path
                left = 0
                right = len(path) - 1
                left = self._advance_from_start(path, left, right)
                right = self._advance_from_end(path, left, right)
                continue

            feasible = []

            sx, sy = path[left + 1]
            if self.can_place_track(sx, sy):
                c = self._empty_cell_place_cost(sx, sy)
                if c <= budget:
                    feasible.append((c, 0, sx, sy))

            ex, ey = path[right - 1]
            if self.can_place_track(ex, ey):
                c = self._empty_cell_place_cost(ex, ey)
                if c <= budget:
                    feasible.append((c, 1, ex, ey))

            if not feasible:
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                if not new_path or new_path == path:
                    break
                path = new_path
                if self.active_build and self.active_build_path:
                    self.active_build_path = path
                left = 0
                right = len(path) - 1
                left = self._advance_from_start(path, left, right)
                right = self._advance_from_end(path, left, right)
                continue

            # If we can only extend one end, try a fresh re-path once before committing.
            # This prevents one-sided growth when the other end is blocked by recent ink.
            if len(feasible) == 1:
                new_path = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
                if new_path and new_path != path:
                    path = new_path
                    if self.active_build and self.active_build_path:
                        self.active_build_path = path
                    left = 0
                    right = len(path) - 1
                    left = self._advance_from_start(path, left, right)
                    right = self._advance_from_end(path, left, right)
                    continue

            feasible.sort(key=lambda t: (t[0], t[1]))
            cost, side, x, y = feasible[0]
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(cost)
            self.cell_tracks[(x, y)] = self.my_id
            if side == 0:
                left += 1
            else:
                right -= 1

        return actions

    def _my_tracks_on_path(self, path: Optional[List[Tuple[int, int]]]) -> int:
        if not path:
            return 0
        n = 0
        for x, y in path:
            if self.cell_tracks.get((x, y)) == self.my_id:
                n += 1
        return n

    def _path_has_affordable_step(self, path: List[Tuple[int, int]]) -> bool:
        for x, y in path:
            if not self.can_place_track(x, y):
                continue
            return self._empty_cell_place_cost(x, y) <= self.POINTS_PER_TURN
        return False

    # -------------------- V25: Smart densify --------------------

    def _remaining_budget_after_actions(self, actions: List[str]) -> int:
        spent = 0
        for a in actions:
            if not a.startswith("PLACE_TRACKS"):
                continue
            parts = a.split()
            if len(parts) != 3:
                continue
            try:
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                continue
            spent += self._empty_cell_place_cost(x, y)
        return max(0, self.POINTS_PER_TURN - spent)

    def _dense_candidate_score(self, x: int, y: int) -> float:
        # V24: Prefer cheap cells that create junctions with existing own network.
        cost = self._empty_cell_place_cost(x, y)
        region_id = self.region_ids[y][x]
        instability = self.region_instability.get(region_id, 0)

        adj_my = 0
        adj_opp = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if (nx, ny) in self.town_cells:
                adj_my += 1
                continue
            owner = self.cell_tracks.get((nx, ny))
            if owner == self.my_id:
                adj_my += 1
            elif owner == self.opp_id:
                adj_opp += 1

        # Cost is king; also reward junction formation.
        score = 0.0
        score += (3 - cost) * 10.0
        score += adj_my * 6.0
        if adj_my >= 2:
            score += 12.0

        # Prefer stable regions.
        score += (self.DENSE_MAX_INSTABILITY - instability) * 2.5

        # Slightly prefer consolidating inside existing owned regions.
        if region_id in self.my_track_regions:
            score += 3.0

        # Avoid creating adjacency to opponent tracks (helps them latch on).
        score -= adj_opp * 4.0

        return score

    def _densify_network(self, budget: int) -> List[str]:
        if budget <= 0:
            return []

        actions: List[str] = []
        placed = 0

        # Candidate set = neighbors of own tracks and towns.
        frontier: Set[Tuple[int, int]] = set()
        anchors = set(self.town_cells)
        anchors.update([pos for pos, owner in self.cell_tracks.items() if owner == self.my_id])

        for ax, ay in anchors:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        while budget > 0 and placed < self.DENSE_MAX_EXTRA_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in list(frontier):
                if not self.can_place_track(x, y):
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                if self.region_instability.get(rid, 0) > self.DENSE_MAX_INSTABILITY:
                    continue
                if self.DENSE_AVOID_OPP_REGIONS and rid in self.opp_track_regions:
                    continue

                s = self._dense_candidate_score(x, y)
                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1

            # Update local state so next pick can build around the new cell.
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

            # Expand frontier from new cell.
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        return actions

    # -------------------- V21 prediction helpers --------------------

    def _town_near_opp_tracks(self, town: Town, opp_cells: List[Tuple[int, int]]) -> bool:
        if not opp_cells:
            return False
        tx, ty = town.x, town.y
        dmax = self.OPP_TOWN_NEAR_DIST
        for x, y in opp_cells:
            if abs(x - tx) + abs(y - ty) <= dmax:
                return True
        return False

    def _get_cell_cost_for_opponent_path(self, x: int, y: int) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')

        if (x, y) in self.town_cells:
            return 0.0

        region_id = self.region_ids[y][x]
        if region_id == -1 or region_id in self.region_inked:
            return float('inf')

        instability = self.region_instability.get(region_id, 0)
        if instability >= self.INSTABILITY_TO_INK:
            return float('inf')

        if (x, y) in self.cell_tracks:
            owner = self.cell_tracks[(x, y)]
            if owner == self.opp_id:
                return 0.0
            if owner == self.my_id:
                return float(self.MY_TRACK_BLOCKING_PENALTY)
            return float(self.NEUTRAL_TRACK_PENALTY)

        base_cost = int(self.cell_types[y][x] + 1)
        if base_cost > self.POINTS_PER_TURN:
            return float('inf')

        return float(base_cost + instability * 2)

    def _find_opponent_path(self, x0: int, y0: int, x1: int, y1: int, max_nodes: int = 1200) -> Optional[List[Tuple[int, int]]]:
        def heuristic(x, y):
            return abs(x - x1) + abs(y - y1)

        order = itertools.count()
        open_set = [(0.0, 0.0, next(order), x0, y0)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(x0, y0): 0.0}
        nodes_expanded = 0

        while open_set and nodes_expanded < max_nodes:
            _, g, _, x, y = heapq.heappop(open_set)
            nodes_expanded += 1

            if x == x1 and y == y1:
                path = []
                cur = (x, y)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((x0, y0))
                return list(reversed(path))

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                step_cost = self._get_cell_cost_for_opponent_path(nx, ny)
                if step_cost == float('inf'):
                    continue

                tentative_g = g + step_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, tentative_g, next(order), nx, ny))

        return None

    def _opp_candidate_value(self, path: List[Tuple[int, int]]) -> float:
        if not path:
            return 0.0

        cost = 0
        empty_cells = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                continue
            c = int(self.cell_types[y][x] + 1)
            if c > self.POINTS_PER_TURN:
                return 0.0
            cost += c
            empty_cells += 1

        if cost <= 0:
            return 0.0

        turns_to_complete = (cost + self.POINTS_PER_TURN - 1) // self.POINTS_PER_TURN
        turns_left = self.TOTAL_TURNS - self.turn
        scoring_turns = max(0, turns_left - turns_to_complete)
        if scoring_turns <= 0:
            return 0.0

        return float((empty_cells * scoring_turns) / max(cost, 1) * 100.0)

    def _predict_opponent_scored_paths(self) -> List[Tuple[float, List[Tuple[int, int]]]]:
        if self._pred_opp_scored_paths_turn == self.turn:
            return self._pred_opp_scored_paths

        scored_paths: List[Tuple[float, List[Tuple[int, int]]]] = []
        opp_cells = [pos for pos, owner in self.cell_tracks.items() if owner == self.opp_id]
        if opp_cells:
            for town in self.towns:
                if not self._town_near_opp_tracks(town, opp_cells):
                    continue
                for dest_id in town.desired:
                    conn_id = (min(town.id, dest_id), max(town.id, dest_id))
                    if conn_id in self.completed_connections:
                        continue
                    dest = self.town_by_id.get(dest_id)
                    if dest is None:
                        continue
                    if not self._town_near_opp_tracks(dest, opp_cells):
                        continue

                    path = self._find_opponent_path(town.x, town.y, dest.x, dest.y)
                    if not path or len(path) < 2:
                        continue

                    v = self._opp_candidate_value(path)
                    if v <= 0:
                        continue

                    scored_paths.append((v, path))

        scored_paths.sort(key=lambda t: -t[0])
        self._pred_opp_scored_paths = scored_paths[: self.PRED_TOP_K]
        self._pred_opp_scored_paths_turn = self.turn
        return self._pred_opp_scored_paths

    def _predict_opponent_region_weights(self) -> Dict[int, float]:
        top = self._predict_opponent_scored_paths()
        if not top:
            return {}

        pred: Dict[int, float] = defaultdict(float)
        for v, path in top:
            denom = float(max(len(path) - 1, 1))
            per_cell = v / denom
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                region_id = self.region_ids[y][x]
                if region_id == -1:
                    continue
                if region_id in self.town_regions or region_id in self.region_inked:
                    continue
                pred[region_id] += per_cell

        return pred

    def _choke_scored_cells(self, budget: int) -> List[Tuple[float, int, int, int]]:
        if budget <= 0:
            return []

        top = self._predict_opponent_scored_paths()
        if not top:
            return []

        top = top[: self.CHOKE_TOP_K_PATHS]

        best_cells: List[Tuple[float, int, int, int]] = []
        # (score, cost, x, y)
        for v, path in top:
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue
                cost = self._empty_cell_place_cost(x, y)
                if cost > budget or cost > self.CHOKE_MAX_CELL_COST:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.town_regions or rid in self.region_inked:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                empty_cost = float(cost + instability * 2)
                delta = float(self.MY_TRACK_BLOCKING_PENALTY) - empty_cost
                if delta < self.CHOKE_MIN_DELTA_COST:
                    continue

                # Prefer high-value paths, large delta, and low placement cost.
                s = float(v) * delta / float(max(cost, 1))
                best_cells.append((s, int(cost), x, y))

        best_cells.sort(reverse=True)
        return best_cells

    def _high_confidence_choke(self, budget: int, *, opp_threat: float, score_delta: int) -> List[str]:
        if not bool(self.HIGH_CONF_CHOKE_ENABLED) or budget <= 0:
            return []
        if int(self.turn) < int(self.HIGH_CONF_CHOKE_TURN_MIN):
            return []
        if abs(int(score_delta)) > int(self.HIGH_CONF_CHOKE_SCORE_DELTA_MAX):
            return []
        if float(opp_threat) < float(self.HIGH_CONF_CHOKE_THREAT_MIN):
            return []
        if (self.my_score - self.opp_score) >= self.CHOKE_DISABLE_WHEN_AHEAD:
            return []

        best_cells = self._choke_scored_cells(budget)
        if not best_cells:
            return []

        best_s, cost, x, y = best_cells[0]
        if float(best_s) < float(self.HIGH_CONF_CHOKE_MIN_CELL_SCORE):
            return []
        if budget < int(cost):
            return []
        if not self.can_place_track(x, y):
            return []

        self.cell_tracks[(x, y)] = self.my_id
        rid = self.region_ids[y][x]
        if rid != -1:
            self.my_track_regions.add(rid)
        return [f"PLACE_TRACKS {x} {y}"]

    def _choke_opponent_paths(self, budget: int) -> List[str]:
        if not self.CHOKE_ENABLED or budget <= 0:
            return []
        if self.turn < self.CHOKE_START_TURN:
            return []
        if (self.my_score - self.opp_score) >= self.CHOKE_DISABLE_WHEN_AHEAD:
            return []

        best_cells = self._choke_scored_cells(budget)
        if not best_cells:
            return []
        actions: List[str] = []
        placed = 0
        used: Set[Tuple[int, int]] = set()

        for s, cost, x, y in best_cells:
            if placed >= self.CHOKE_MAX_PLACEMENTS:
                break
            if budget < cost:
                continue
            if (x, y) in used:
                continue
            if not self.can_place_track(x, y):
                continue
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(cost)
            placed += 1
            used.add((x, y))
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def _opp_threat_value(self) -> float:
        top = self._predict_opponent_scored_paths()
        return float(top[0][0]) if top else 0.0

    def _opp_remaining_cost_simple(self, path: List[Tuple[int, int]]) -> int:
        cost = 0
        for x, y in path:
            if (x, y) in self.town_cells:
                continue
            if (x, y) in self.cell_tracks:
                continue
            c = int(self.cell_types[y][x] + 1)
            if c > self.POINTS_PER_TURN:
                return 10**9
            cost += c
        return int(cost)

    def _v28_imminent_block(
        self,
        budget: int,
        *,
        min_path_value: Optional[float] = None,
        max_turns: Optional[int] = None,
    ) -> List[str]:
        if not bool(self.V28_IMMINENT_BLOCK_ENABLED) or budget <= 0:
            return []

        min_v = float(self.V28_IMMINENT_BLOCK_MIN_PATH_VALUE) if min_path_value is None else float(min_path_value)
        max_t = int(self.V28_IMMINENT_BLOCK_MAX_TURNS) if max_turns is None else int(max_turns)

        top = self._predict_opponent_scored_paths()
        if not top:
            return []

        imminent: List[Tuple[float, List[Tuple[int, int]]]] = []
        for v, path in top[: max(1, int(self.CHOKE_TOP_K_PATHS))]:
            if float(v) < float(min_v):
                continue
            cost = self._opp_remaining_cost_simple(path)
            if cost >= 10**8:
                continue
            turns = (int(cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
            if turns <= int(max_t):
                imminent.append((float(v), path))

        if not imminent:
            return []

        best_cells: List[Tuple[float, int, int, int]] = []
        max_cell_cost = int(self.V28_IMMINENT_BLOCK_MAX_CELL_COST)
        for v, path in imminent:
            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue
                cost = self._empty_cell_place_cost(x, y)
                if cost > budget or cost > max_cell_cost:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.town_regions or rid in self.region_inked:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                empty_cost = float(cost + instability * 2)
                delta = float(self.MY_TRACK_BLOCKING_PENALTY) - empty_cost
                if delta <= 0.0:
                    continue
                s = float(v) * delta / float(max(cost, 1))
                best_cells.append((s, int(cost), x, y))

        if not best_cells:
            return []
        best_cells.sort(reverse=True)

        best_s, cost, x, y = best_cells[0]
        if float(best_s) < float(self.V28_IMMINENT_BLOCK_MIN_CELL_SCORE):
            return []
        if int(cost) > int(budget):
            return []
        if not self.can_place_track(x, y):
            return []

        self.cell_tracks[(x, y)] = self.my_id
        rid = self.region_ids[y][x]
        if rid != -1:
            self.my_track_regions.add(rid)
        return [f"PLACE_TRACKS {x} {y}"]

    def _support_path_placement_score(self, x: int, y: int, cost: int, end_a: Tuple[int, int], end_b: Tuple[int, int]) -> float:
        rid = self.region_ids[y][x]
        instability = self.region_instability.get(rid, 0)

        da = abs(x - end_a[0]) + abs(y - end_a[1])
        db = abs(x - end_b[0]) + abs(y - end_b[1])
        d = min(da, db)

        adj_opp = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if self.cell_tracks.get((nx, ny)) == self.opp_id:
                adj_opp += 1

        score = 0.0
        score += (3 - cost) * 8.0
        score += max(0.0, 10.0 - float(d))
        score += (self.SUPPORT_MAX_INSTABILITY - instability) * 2.0
        score -= adj_opp * 3.0
        return score

    def _support_best_path(self, budget: int, path: Optional[List[Tuple[int, int]]]) -> List[str]:
        if not self.SUPPORT_ENABLED or budget <= 0 or not path or len(path) < 2:
            return []

        end_a = path[0]
        end_b = path[-1]
        actions: List[str] = []
        placed = 0

        while budget > 0 and placed < self.SUPPORT_MAX_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in path:
                if (x, y) in self.town_cells:
                    continue
                if not self.can_place_track(x, y):
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability > self.SUPPORT_MAX_INSTABILITY:
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget or c > self.SUPPORT_MAX_CELL_COST:
                    continue

                s = self._support_path_placement_score(x, y, c, end_a, end_b)
                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def _targeted_densify(self, budget: int, focus: Set[Tuple[int, int]]) -> List[str]:
        if not self.TARGETED_DENSE_ENABLED or budget <= 0 or not focus:
            return []

        actions: List[str] = []
        placed = 0

        my_track_positions = [pos for pos, owner in self.cell_tracks.items() if owner == self.my_id]

        frontier: Set[Tuple[int, int]] = set()
        for fx, fy in focus:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        def dist_to_focus(x: int, y: int) -> int:
            # Focus sets are small; linear scan is fine.
            best = 10**9
            for fx, fy in focus:
                d = abs(x - fx) + abs(y - fy)
                if d < best:
                    best = d
                    if best <= 1:
                        break
            return int(best)

        while budget > 0 and placed < self.TARGETED_DENSE_MAX_PLACEMENTS:
            best = None
            best_score = -1e9

            for x, y in list(frontier):
                if not self.can_place_track(x, y):
                    continue
                c = self._empty_cell_place_cost(x, y)
                if c > budget or c > self.TARGETED_DENSE_MAX_CELL_COST:
                    continue
                rid = self.region_ids[y][x]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                if self.region_instability.get(rid, 0) > self.TARGETED_DENSE_MAX_INSTABILITY:
                    continue
                if self.DENSE_AVOID_OPP_REGIONS and rid in self.opp_track_regions:
                    continue

                if self.TARGETED_DENSE_MAX_DIST_TO_MY_TRACK >= 0 and my_track_positions:
                    near_my = False
                    for tx, ty in my_track_positions:
                        if abs(tx - x) + abs(ty - y) <= self.TARGETED_DENSE_MAX_DIST_TO_MY_TRACK:
                            near_my = True
                            break
                    if not near_my:
                        continue

                # Score: prefer cheap, near focus, and not adjacent to opponent.
                instability = self.region_instability.get(rid, 0)
                d = dist_to_focus(x, y)

                adj_my = 0
                adj_opp = 0
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if (nx, ny) in self.town_cells:
                        adj_my += 1
                        continue
                    owner = self.cell_tracks.get((nx, ny))
                    if owner == self.my_id:
                        adj_my += 1
                    elif owner == self.opp_id:
                        adj_opp += 1

                s = 0.0
                s += (3 - c) * 9.0
                s += max(0.0, 8.0 - float(d))
                s += adj_my * 3.5
                s -= adj_opp * 4.5
                s += (self.TARGETED_DENSE_MAX_INSTABILITY - instability) * 2.0

                if s > best_score:
                    best_score = s
                    best = (x, y, c)

            if best is None:
                break

            x, y, c = best
            actions.append(f"PLACE_TRACKS {x} {y}")
            budget -= int(c)
            placed += 1
            self.cell_tracks[(x, y)] = self.my_id
            rid = self.region_ids[y][x]
            if rid != -1:
                self.my_track_regions.add(rid)

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    frontier.add((nx, ny))

        return actions

    # -------------------- Disruption (V20 baseline + V21 override) --------------------

    def _get_disruption_target_v20(self) -> Optional[int]:
        turns_left = self.TOTAL_TURNS - self.turn

        anti_dense = self._opp_dense_potential_by_region() if self.ANTI_DENSE_ENABLED else {}

        def net_score(region_id: int) -> float:
            opp_w = self.active_weight_by_region_opp.get(region_id, 0.0)
            my_w = self.active_weight_by_region_my.get(region_id, 0.0)
            neu_w = self.active_weight_by_region_neutral.get(region_id, 0.0)

            instability = self.region_instability.get(region_id, 0)
            hits_to_ink = self.INSTABILITY_TO_INK - instability

            base = (opp_w - self.SELF_HARM_ALPHA * my_w + 0.15 * neu_w)
            if anti_dense:
                base += float(anti_dense.get(region_id, 0.0)) * float(self.ANTI_DENSE_ALPHA)
            if base <= 0:
                return -1e9

            score = (base * turns_left) / max(hits_to_ink, 1)
            if hits_to_ink == 1:
                score *= 2.0
            elif hits_to_ink >= 2:
                score *= 0.65
            return float(score)

        immediate: List[Tuple[int, float]] = []
        for region_id in set(
            list(self.active_weight_by_region_opp.keys())
            + list(self.active_weight_by_region_my.keys())
            + list(self.active_weight_by_region_neutral.keys())
            + (list(anti_dense.keys()) if anti_dense else [])
        ):
            if region_id in self.region_inked or region_id in self.town_regions:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue
            if (self.INSTABILITY_TO_INK - instability) == 1:
                s = net_score(region_id)
                if s > 0:
                    immediate.append((region_id, s))
        if immediate:
            return max(immediate, key=lambda t: t[1])[0]

        candidates: Dict[int, float] = {}

        for region_id in set(list(self.active_weight_by_region_opp.keys()) + list(self.opp_track_regions) + (list(anti_dense.keys()) if anti_dense else [])):
            if region_id in self.region_inked or region_id in self.town_regions:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue

            score = net_score(region_id)
            if score > 0:
                candidates[region_id] = max(candidates.get(region_id, 0.0), score)

        if not candidates:
            for region_id in self.opp_track_regions:
                if region_id in self.region_inked or region_id in self.town_regions:
                    continue
                instability = self.region_instability.get(region_id, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                my_w = self.active_weight_by_region_my.get(region_id, 0.0)
                if my_w > 0:
                    continue

                opp_tracks = sum(
                    1
                    for (x, y), owner in self.cell_tracks.items()
                    if owner == self.opp_id and self.region_ids[y][x] == region_id
                )
                if opp_tracks <= 0:
                    continue

                hits_to_ink = self.INSTABILITY_TO_INK - instability
                candidates[region_id] = (opp_tracks * turns_left * 0.25) / max(hits_to_ink, 1)

        if not candidates:
            return None

        return max(candidates.items(), key=lambda kv: kv[1])[0]

    def _opp_dense_potential_by_region(self) -> Dict[int, float]:
        # Estimate how much the opponent can densify next turns in each region by
        # counting cheap, placeable empty frontier cells adjacent to opponent tracks.
        # Higher => better disrupt target to slow V24-style densification.
        if (self.my_score - self.opp_score) >= self.ANTI_DENSE_IGNORE_IF_AHEAD_BY:
            return {}

        potentials: Dict[int, float] = defaultdict(float)
        seen_cells: Set[Tuple[int, int]] = set()

        for (x, y), owner in self.cell_tracks.items():
            if owner != self.opp_id:
                continue
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in self.town_cells:
                    continue
                if (nx, ny) in seen_cells:
                    continue
                if not self.can_place_track(nx, ny):
                    continue
                cost = self._empty_cell_place_cost(nx, ny)
                if cost > self.ANTI_DENSE_MAX_CELL_COST:
                    continue

                rid = self.region_ids[ny][nx]
                if rid == -1 or rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK:
                    continue

                seen_cells.add((nx, ny))
                # Prefer blocking very cheap placements.
                potentials[rid] += float(4 - cost)

        return potentials

    def _block_opponent_densify(self, budget: int) -> List[str]:
        if budget <= 0 or not self.ANTI_DENSE_ENABLED:
            return []
        if (self.my_score - self.opp_score) >= self.ANTI_DENSE_IGNORE_IF_AHEAD_BY:
            return []

        potentials = self._opp_dense_potential_by_region()
        if not potentials:
            return []

        # Consider only top few regions to keep this cheap.
        top_regions = sorted(potentials.items(), key=lambda kv: -kv[1])[:6]

        max_places = min(self.ANTI_DENSE_BLOCK_PLACEMENTS, max(0, budget))
        if max_places <= 0:
            return []

        frontier: List[Tuple[float, int, int, int, int]] = []
        # (score, cost, y, x, region)
        for rid, pot in top_regions:
            if rid in self.region_inked or rid in self.town_regions:
                continue
            instability = self.region_instability.get(rid, 0)
            if instability >= self.INSTABILITY_TO_INK or instability > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                continue

            # Enumerate cheap, placeable empty cells adjacent to opponent tracks in this region.
            for (x, y), owner in self.cell_tracks.items():
                if owner != self.opp_id:
                    continue
                if self.region_ids[y][x] != rid:
                    continue
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if (nx, ny) in self.town_cells:
                        continue
                    if not self.can_place_track(nx, ny):
                        continue
                    cost = self._empty_cell_place_cost(nx, ny)
                    if cost > budget or cost > self.ANTI_DENSE_MAX_CELL_COST:
                        continue
                    rr = self.region_ids[ny][nx]
                    if rr != rid or rr == -1 or rr in self.region_inked or rr in self.town_regions:
                        continue
                    inst2 = self.region_instability.get(rr, 0)
                    if inst2 >= self.INSTABILITY_TO_INK or inst2 > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                        continue

                    # Prefer blocking cheaper cells in higher-potential regions.
                    s = float(pot) * 2.0 + float(4 - cost)
                    frontier.append((s, cost, ny, nx, rid))

        # Region-poison candidates: placing a single cheap track anywhere in the region
        # makes V24 avoid densifying there (it avoids opponent-track regions).
        poison: List[Tuple[float, int, int, int, int]] = []
        if self.ANTI_DENSE_POISON_REGION:
            for rid, pot in top_regions:
                if rid in self.region_inked or rid in self.town_regions:
                    continue
                instability = self.region_instability.get(rid, 0)
                if instability >= self.INSTABILITY_TO_INK or instability > self.ANTI_DENSE_BLOCK_MAX_INSTABILITY:
                    continue

                best_cell = None
                best_cost = 10**9
                # Keep poison local: look for a cheap cell within a small radius of the
                # opponent's frontier in this region.
                frontier_seeds: Set[Tuple[int, int]] = set()
                for (ox, oy), owner in self.cell_tracks.items():
                    if owner != self.opp_id:
                        continue
                    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                        sx, sy = ox + dx, oy + dy
                        if not (0 <= sx < self.width and 0 <= sy < self.height):
                            continue
                        if (sx, sy) in self.town_cells:
                            continue
                        if self.region_ids[sy][sx] != rid:
                            continue
                        if not self.can_place_track(sx, sy):
                            continue
                        frontier_seeds.add((sx, sy))

                if not frontier_seeds:
                    continue

                r = int(self.ANTI_DENSE_POISON_MAX_DIST_TO_FRONTIER)
                scanned: Set[Tuple[int, int]] = set()
                for sx, sy in frontier_seeds:
                    for dx in range(-r, r + 1):
                        rem = r - abs(dx)
                        for dy in range(-rem, rem + 1):
                            x, y = sx + dx, sy + dy
                            if not (0 <= x < self.width and 0 <= y < self.height):
                                continue
                            if (x, y) in scanned or (x, y) in self.town_cells:
                                continue
                            scanned.add((x, y))
                            if self.region_ids[y][x] != rid:
                                continue
                            if not self.can_place_track(x, y):
                                continue
                            c = self._empty_cell_place_cost(x, y)
                            if c < best_cost:
                                best_cost = c
                                best_cell = (x, y)
                                if best_cost <= 1:
                                    break
                        if best_cost <= 1:
                            break
                    if best_cost <= 1:
                        break

                if best_cell is None:
                    continue
                if best_cost > budget or best_cost > self.ANTI_DENSE_POISON_MAX_CELL_COST:
                    continue

                # Prefer poisoning very cheap cells in high-potential regions.
                x, y = best_cell
                s = float(pot) * 3.0 + float(5 - best_cost)
                poison.append((s, int(best_cost), y, x, rid))

        if not frontier and not poison:
            return []

        frontier.sort(reverse=True)
        poison.sort(reverse=True)
        actions: List[str] = []
        placed = 0
        used: Set[Tuple[int, int]] = set()

        # Mix both sources: try best poison first, then best adjacency block.
        merged = poison[:]
        merged.extend(frontier)
        merged.sort(reverse=True)

        for s, cost, ny, nx, rid in merged:
            if placed >= max_places:
                break
            if budget < cost:
                continue
            if (nx, ny) in used:
                continue
            if not self.can_place_track(nx, ny):
                continue

            actions.append(f"PLACE_TRACKS {nx} {ny}")
            budget -= int(cost)
            placed += 1
            used.add((nx, ny))
            self.cell_tracks[(nx, ny)] = self.my_id
            if rid != -1:
                self.my_track_regions.add(rid)

        return actions

    def get_disruption_target(self, best_next_path: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
        v20_target = self._get_disruption_target_v20()

        # When comfortably ahead, avoid "creative" overrides; keep the reliable baseline.
        score_delta = self.my_score - self.opp_score
        if score_delta >= self.PRED_DISABLE_OVERRIDE_WHEN_AHEAD:
            return v20_target

        base_hits = None
        if v20_target is not None:
            base_instability = self.region_instability.get(v20_target, 0)
            base_hits = self.INSTABILITY_TO_INK - base_instability
            # If baseline is already an immediate 1-hit ink and we're not behind,
            # don't override it.
            if base_hits == 1 and score_delta >= -self.PRED_ALLOW_OVERRIDE_IF_BASE_1HIT_BEHIND:
                return v20_target

        avoid: Set[int] = set()
        if self.active_build_path:
            for x, y in self.active_build_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)

        if best_next_path:
            for x, y in best_next_path:
                if (x, y) in self.town_cells:
                    continue
                rid = self.region_ids[y][x]
                if rid != -1:
                    avoid.add(rid)

        pred_opp = self._predict_opponent_region_weights()

        best_region: Optional[int] = None
        best_w = 0.0

        allow_2hit = (-score_delta) >= self.PRED_OVERRIDE_BEHIND_FOR_2HIT

        for region_id, w in pred_opp.items():
            if w <= 0:
                continue
            if region_id in self.region_inked or region_id in self.town_regions or region_id in avoid:
                continue
            instability = self.region_instability.get(region_id, 0)
            if instability >= self.INSTABILITY_TO_INK:
                continue
            hits = (self.INSTABILITY_TO_INK - instability)
            if hits != 1:
                if not (allow_2hit and hits == 2):
                    continue
            # Never override a baseline 1-hit ink with a 2-hit ink.
            if base_hits == 1 and hits == 2:
                continue

            if region_id not in self.opp_track_regions and region_id not in self.active_weight_by_region_opp:
                continue
            if self.active_weight_by_region_my.get(region_id, 0.0) > 0.8:
                continue

            if w > best_w:
                best_w = float(w)
                best_region = region_id

        if best_region is not None:
            base_w = float(pred_opp.get(v20_target, 0.0)) if v20_target is not None else 0.0
            required = max(self.PRED_OVERRIDE_THRESHOLD, base_w + self.PRED_COMPARE_MARGIN)
            if base_hits == 1:
                required = max(required, base_w + self.PRED_COMPARE_MARGIN + self.PRED_EXTRA_MARGIN_IF_BASE_1HIT)
            if best_w >= required:
                return best_region

        return v20_target

    def _panic_disrupt_target(
        self,
        *,
        best_next_path: Optional[List[Tuple[int, int]]],
        opp_threat: float,
        score_delta: int,
        baseline_target: Optional[int],
    ) -> Optional[int]:
        if not bool(self.DISRUPT_PANIC_ENABLED):
            return None
        if int(score_delta) > -int(self.DISRUPT_PANIC_BEHIND_BY_MIN):
            return None
        if float(opp_threat) < float(self.DISRUPT_PANIC_THREAT_MIN):
            return None

        cands = self._get_disruption_candidates_for_dataset(best_next_path, limit=int(self.DISRUPT_PANIC_TOP_K))
        if not cands:
            return None

        allow_2hit = bool(self.DISRUPT_PANIC_ALLOW_2HIT)
        best_rid = None
        best_score = -1e18
        base_score = None

        for c in cands:
            rid = c.get("rid")
            if not isinstance(rid, int):
                continue
            if bool(c.get("avoid")):
                continue
            if rid in self.region_inked or rid in self.town_regions:
                continue
            hits = int(c.get("hits_to_ink", 0) or 0)
            if hits <= 0:
                continue
            if hits != 1 and not (allow_2hit and hits == 2):
                continue
            if bool(c.get("in_my_track")):
                continue

            # Panic score: favor predicted opponent importance, immediate opponent weight,
            # and anti-dense potential; prefer faster inks via /hits.
            pred_w = float(c.get("pred_opp_w", 0.0))
            opp_w = float(c.get("opp_w", 0.0))
            anti_dense = float(c.get("anti_dense", 0.0))
            s = (
                float(self.DISRUPT_PANIC_W_PRED) * pred_w
                + float(self.DISRUPT_PANIC_W_OPP_W) * opp_w
                + float(self.DISRUPT_PANIC_W_ANTI_DENSE) * anti_dense
            ) / float(max(1, hits))

            if baseline_target is not None and int(rid) == int(baseline_target):
                base_score = float(s)

            if s > best_score:
                best_score = float(s)
                best_rid = int(rid)

        if best_rid is None:
            return None
        if baseline_target is None:
            return int(best_rid)

        # Only override baseline if meaningfully better.
        if base_score is None:
            return None
        if (float(best_score) - float(base_score)) < float(self.DISRUPT_PANIC_MARGIN):
            return None
        if int(best_rid) == int(baseline_target):
            return None
        return int(best_rid)

    # -------------------- Candidates --------------------

    def get_connection_candidates(self) -> List[Tuple[int, int, List[Tuple[int, int]], float]]:
        candidates = []

        for town in self.towns:
            for dest_id in town.desired:
                conn_id = (min(town.id, dest_id), max(town.id, dest_id))
                if conn_id in self.completed_connections:
                    continue
                if dest_id not in self.town_by_id:
                    continue

                dest = self.town_by_id[dest_id]
                path = self.find_path(town.x, town.y, dest.x, dest.y)
                if not path or len(path) < 2:
                    continue

                if not self._path_has_affordable_step(path):
                    continue

                ev = self.calc_connection_ev(path, from_id=town.id, to_id=dest_id)
                if ev > 0:
                    candidates.append((town.id, dest_id, path, ev))

        candidates.sort(key=lambda x: -x[3])

        if candidates and bool(self.CONN_TIEBREAK_ENABLED):
            try:
                if int(self.turn) >= int(self.CONN_TIEBREAK_TURN_MIN) and abs(int(self.my_score - self.opp_score)) <= int(self.CONN_TIEBREAK_SCORE_DELTA_MAX):
                    best_ev = float(candidates[0][3])
                    window = float(self.CONN_TIEBREAK_EV_WINDOW)
                    w_fast = float(self.CONN_TIEBREAK_W_FAST)
                    w_gift = float(self.CONN_TIEBREAK_W_GIFT)

                    def tiebreak_bonus(item: Tuple[int, int, List[Tuple[int, int]], float]) -> float:
                        _, _, path, ev = item
                        if float(ev) < best_ev - window:
                            return 0.0
                        cost = self.calc_path_cost(path)
                        if cost <= 0 or cost >= 10**8:
                            return 0.0
                        turns_to_complete = (int(cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                        if turns_to_complete <= 0:
                            turns_to_complete = 1
                        _, opp_cells, neutral_cells, _ = self._count_track_owners_on_path(path)
                        gift = float(opp_cells * 7 + neutral_cells * 2)

                        bonus_fast = (-float(turns_to_complete)) * w_fast
                        bonus_gift = (-gift) * w_gift
                        return bonus_fast + bonus_gift

                    candidates.sort(key=lambda it: (-float(it[3]), -tiebreak_bonus(it)))
            except Exception:
                # Never let tie-breaker logic affect gameplay stability.
                pass
        return candidates


    def _detect_opponent_behavior(self):
        if self.turn < 5:
            return
        if self.detected_opp_bias is not None and self.turn < 15:
            return

        towns_with_opp_tracks = set()
        for (x, y), owner in self.cell_tracks.items():
            if owner != self.opp_id:
                continue
            min_dist = 10**9
            nearest = None
            for town in self.towns:
                dist = abs(town.x - x) + abs(town.y - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = town.id
            if nearest is not None and min_dist < 10:
                towns_with_opp_tracks.add(nearest)

        if len(towns_with_opp_tracks) < 2:
            return

        avg_opp_town_id = sum(towns_with_opp_tracks) / len(towns_with_opp_tracks)
        avg_all_town_id = sum(t.id for t in self.towns) / len(self.towns)

        if avg_opp_town_id > avg_all_town_id * 1.15:
            if self.detected_opp_bias != 'high':
                self.detected_opp_bias = 'high'
                self.my_bias_multiplier = -5.0
        elif avg_opp_town_id < avg_all_town_id * 0.85:
            if self.detected_opp_bias != 'low':
                self.detected_opp_bias = 'low'
                self.my_bias_multiplier = 6.0
        else:
            if self.detected_opp_bias != 'neutral':
                self.detected_opp_bias = 'neutral'
                self.my_bias_multiplier = -3.0

    # -------------------- Action selection --------------------

    def get_action(self) -> str:
        actions: List[str] = []

        continue_ev = self.calc_continue_ev()
        new_candidates = self.get_connection_candidates()
        best_new = new_candidates[0] if new_candidates else None
        best_new_ev = best_new[3] if best_new else 0.0
        best_new_path = best_new[2] if best_new else None

        opp_threat = self._opp_threat_value()
        score_delta = self.my_score - self.opp_score

        # Trace helpers (updated as we go)
        tempo_mode: Optional[str] = None
        endgame_block_fired: bool = False

        # V28: very selective midgame imminent-block.
        if best_new and bool(self.V28_MIDGAME_BLOCK_ENABLED):
            try:
                if int(self.turn) >= int(self.V28_MIDGAME_BLOCK_TURN_MIN) and int(self.turn) <= int(self.V28_MIDGAME_BLOCK_TURN_MAX) and abs(int(score_delta)) <= int(self.V28_MIDGAME_BLOCK_SCORE_DELTA_MAX):
                    my_cost = self.calc_path_cost(best_new_path) if best_new_path else 10**9
                    my_turns = (int(my_cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                    if my_turns <= 0:
                        my_turns = 1

                    top = self._predict_opponent_scored_paths()
                    if top:
                        opp_v, opp_path = top[0]
                        opp_cost = self._opp_remaining_cost_simple(opp_path)
                        opp_turns = (int(opp_cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                        if opp_turns <= 0:
                            opp_turns = 1

                        if float(opp_v) >= float(self.V28_MIDGAME_BLOCK_MIN_PATH_VALUE) and int(opp_turns) <= int(self.V28_MIDGAME_BLOCK_MAX_TURNS) and int(opp_turns) < int(my_turns):
                            pre_budget = max(0, min(int(self.POINTS_PER_TURN), int(self.V28_MIDGAME_BLOCK_MAX_BUDGET)))
                            if pre_budget > 0:
                                actions.extend(
                                    self._v28_imminent_block(
                                        pre_budget,
                                        min_path_value=float(self.V28_MIDGAME_BLOCK_MIN_PATH_VALUE),
                                        max_turns=int(self.V28_MIDGAME_BLOCK_MAX_TURNS),
                                    )
                                )
            except Exception:
                pass

        # V28: selective endgame imminent-block.
        # Only fire when we are late + close AND the opponent is projected to
        # finish their best predicted path sooner than our best connection.
        if best_new and bool(self.V28_ENDGAME_ENABLED):
            try:
                if int(self.turn) >= int(self.V28_ENDGAME_TURN_MIN) and abs(int(score_delta)) <= int(self.V28_ENDGAME_CLOSE_SCORE_DELTA_MAX):
                    my_cost = self.calc_path_cost(best_new_path) if best_new_path else 10**9
                    my_turns = (int(my_cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                    if my_turns <= 0:
                        my_turns = 1

                    top = self._predict_opponent_scored_paths()
                    if top:
                        opp_v, opp_path = top[0]
                        opp_cost = self._opp_remaining_cost_simple(opp_path)
                        opp_turns = (int(opp_cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                        if opp_turns <= 0:
                            opp_turns = 1

                        if float(opp_v) >= float(self.V28_IMMINENT_BLOCK_MIN_PATH_VALUE) and int(opp_turns) <= int(self.V28_IMMINENT_BLOCK_MAX_TURNS) and int(opp_turns) < int(my_turns):
                            pre_budget = max(0, min(int(self.POINTS_PER_TURN), int(self.V28_IMMINENT_BLOCK_MAX_BUDGET)))
                            if pre_budget > 0:
                                actions.extend(self._v28_imminent_block(pre_budget))
                                endgame_block_fired = True
            except Exception:
                pass

        if best_new and bool(self.TEMPO_ENABLED):
            try:
                if int(self.turn) >= int(self.TEMPO_TURN_MIN) and abs(int(score_delta)) <= int(self.TEMPO_SCORE_DELTA_MAX):
                    tempo_mode = "late"
                elif bool(self.V28_EARLY_TEMPO_ENABLED):
                    if int(self.turn) <= int(self.V28_EARLY_TEMPO_TURN_MAX) and int(score_delta) <= int(self.V28_EARLY_TEMPO_SCORE_DELTA_MIN):
                        tempo_mode = "early"

                if tempo_mode:
                    best_ev = float(best_new_ev)
                    if tempo_mode == "early":
                        window = float(self.V28_EARLY_TEMPO_EV_WINDOW)
                        top_k = max(1, int(self.V28_EARLY_TEMPO_TOP_K))
                        max_turns = max(1, int(self.V28_EARLY_TEMPO_MAX_TURNS_TO_COMPLETE))
                        bonus = float(self.V28_EARLY_TEMPO_BONUS)
                        w_fast = float(self.V28_EARLY_TEMPO_W_FAST)
                    else:
                        window = float(self.TEMPO_EV_WINDOW)
                        top_k = max(1, int(self.TEMPO_TOP_K))
                        max_turns = max(1, int(self.TEMPO_MAX_TURNS_TO_COMPLETE))
                        bonus = float(self.TEMPO_BONUS)
                        w_fast = float(self.TEMPO_W_FAST)

                    best_item = best_new
                    best_metric = -1e18

                    for item in new_candidates[:top_k]:
                        _, _, path, ev = item
                        ev_f = float(ev)
                        if ev_f < best_ev - window:
                            continue
                        cost = self.calc_path_cost(path)
                        if cost <= 0 or cost >= 10**8:
                            continue
                        turns_to_complete = (int(cost) + int(self.POINTS_PER_TURN) - 1) // int(self.POINTS_PER_TURN)
                        if turns_to_complete <= 0:
                            turns_to_complete = 1

                        metric = ev_f
                        metric += float(max_turns - turns_to_complete) * w_fast
                        if turns_to_complete <= max_turns:
                            metric += bonus

                        if metric > best_metric:
                            best_metric = metric
                            best_item = item

                    best_new = best_item
                    best_new_ev = float(best_new[3])
                    best_new_path = best_new[2]
            except Exception:
                # Never let tempo logic affect gameplay stability.
                pass

        switch_threshold = float(self.SWITCH_THRESHOLD)

        if bool(self.HIGH_CONF_CHOKE_ENABLED):
            pre_budget = max(0, min(self.POINTS_PER_TURN, int(self.HIGH_CONF_CHOKE_MAX_BUDGET)))
            if pre_budget > 0:
                actions.extend(
                    self._high_confidence_choke(
                        pre_budget,
                        opp_threat=float(opp_threat),
                        score_delta=int(score_delta),
                    )
                )

        if bool(self.PRE_CHOKE_ENABLED):
            if float(opp_threat) >= float(self.PRE_CHOKE_THREAT_MIN):
                pre_budget = max(0, min(self.POINTS_PER_TURN, int(self.PRE_CHOKE_MAX_BUDGET)))
                if pre_budget > 0:
                    actions.extend(self._choke_opponent_paths(pre_budget))

        if self.active_build and self.active_build_path:
            # Avoid abandoning partially-built connections.
            # Only switch away if we are stuck for multiple turns.
            if best_new and int(self._active_build_stuck_turns) >= 2 and best_new_ev > continue_ev * switch_threshold:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path
                self._active_build_stuck_turns = 0

            from_id, to_id = self.active_build
            from_town = self.town_by_id[from_id]
            to_town = self.town_by_id[to_id]
            refreshed = self.find_path(from_town.x, from_town.y, to_town.x, to_town.y, use_cache=False)
            if refreshed:
                self.active_build_path = refreshed
            else:
                # Path is no longer available (often due to ink). Don't keep building dead segments.
                self.active_build = None
                self.active_build_path = None
                self._active_build_stuck_turns = 0

            budget_for_build = self._remaining_budget_after_actions(actions)
            build_actions = self.build_path_two_ended_stable(
                from_town,
                to_town,
                budget_for_build,
                path_override=self.active_build_path,
            )
            actions.extend(build_actions)

            if budget_for_build > 0 and not build_actions:
                self._active_build_stuck_turns += 1
            else:
                self._active_build_stuck_turns = 0

            # If still stuck after trying to build, fall back to a new best connection,
            # but only after multiple stuck turns to avoid abandoning partial paths.
            if (budget_for_build > 0 and not build_actions) and best_new and int(self._active_build_stuck_turns) >= 2:
                from_id, to_id, path, _ = best_new
                self.active_build = (from_id, to_id)
                self.active_build_path = path
                self._active_build_stuck_turns = 0
                from_town = self.town_by_id[from_id]
                to_town = self.town_by_id[to_id]
                actions.extend(
                    self.build_path_two_ended_stable(
                        from_town,
                        to_town,
                        self._remaining_budget_after_actions(actions),
                        path_override=self.active_build_path,
                    )
                )

        elif best_new:
            from_id, to_id, path, _ = best_new
            self.active_build = (from_id, to_id)
            self.active_build_path = path
            self._active_build_stuck_turns = 0
            from_town = self.town_by_id[from_id]
            to_town = self.town_by_id[to_id]
            actions.extend(
                self.build_path_two_ended_stable(
                    from_town,
                    to_town,
                    self._remaining_budget_after_actions(actions),
                    path_override=self.active_build_path,
                )
            )

        # If we have leftover budget, place extra tracks to densify the network.
        budget_left = self._remaining_budget_after_actions(actions)
        if budget_left > 0:
            # If opponent has a strong near-term route, spend leftover to interfere first.
            if opp_threat > self.DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ and score_delta < self.DENSIFY_ONLY_WHEN_AHEAD_BY:
                actions.extend(self._choke_opponent_paths(budget_left))
                budget_left = self._remaining_budget_after_actions(actions)
                if budget_left > 0:
                    actions.extend(self._block_opponent_densify(budget_left))
                    budget_left = self._remaining_budget_after_actions(actions)

            # Otherwise, invest leftover into the currently committed build path (goal-directed).
            if budget_left > 0:
                support_path = self.active_build_path if self.active_build_path else best_new_path
                actions.extend(self._support_best_path(budget_left, support_path))
                budget_left = self._remaining_budget_after_actions(actions)

            # Anti-dense block/poison (works even when we also supported path).
            if budget_left > 0:
                actions.extend(self._block_opponent_densify(budget_left))
                budget_left = self._remaining_budget_after_actions(actions)

            # Only densify when it’s unlikely we’re sacrificing better scoring tempo.
            if budget_left > 0:
                if (
                    score_delta >= self.DENSIFY_ONLY_WHEN_AHEAD_BY
                    and opp_threat <= self.DENSIFY_ONLY_WHEN_OPP_THREAT_LEQ
                    and best_new_ev <= self.DENSIFY_ONLY_WHEN_BEST_NEW_EV_LEQ
                ):
                    actions.extend(self._densify_network(budget_left))
                else:
                    # If we're not allowed to full-densify, still do a tiny targeted placement
                    # near towns/endpoints to help future connections without random sprawl.
                    focus: Set[Tuple[int, int]] = set()

                    def add_neighborhood(p: Tuple[int, int], r: int) -> None:
                        px, py = p
                        for dx in range(-r, r + 1):
                            rem = r - abs(dx)
                            for dy in range(-rem, rem + 1):
                                nx, ny = px + dx, py + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    focus.add((nx, ny))

                    if self.active_build_path and len(self.active_build_path) >= 2:
                        add_neighborhood(self.active_build_path[0], 2)
                        add_neighborhood(self.active_build_path[-1], 2)
                    elif best_new_path and len(best_new_path) >= 2:
                        add_neighborhood(best_new_path[0], 2)
                        add_neighborhood(best_new_path[-1], 2)
                    else:
                        # Last resort: only consider town-adjacent cells that are near our
                        # existing tracks (keeps late-game from sprawling everywhere).
                        my_tracks = [pos for pos, owner in self.cell_tracks.items() if owner == self.my_id]
                        if my_tracks:
                            for x, y in self.town_adjacent_cells:
                                for tx, ty in my_tracks:
                                    if abs(tx - x) + abs(ty - y) <= self.TARGETED_DENSE_MAX_DIST_TO_MY_TRACK:
                                        focus.add((x, y))
                                        break

                    # V28: in very late, close games, spend leftover budget to pick up
                    # incremental points (tie-breaker) even if we're not ahead.
                    late_close = (
                        bool(self.V28_LATE_DENSIFY_ENABLED)
                        and int(self.turn) >= int(self.V28_LATE_DENSIFY_TURN_MIN)
                        and abs(int(score_delta)) <= int(self.V28_LATE_DENSIFY_SCORE_DELTA_MAX)
                    )

                    actions.extend(self._targeted_densify(budget_left, focus))
                    budget_left = self._remaining_budget_after_actions(actions)
                    if late_close and budget_left > 0:
                        actions.extend(self._targeted_densify(budget_left, focus))

        if self.turn >= 1:
            forced = (self._force_disrupt_turn >= 0 and self.turn == self._force_disrupt_turn)
            forced_region = None
            if forced:
                if int(self._force_disrupt_region) == -1:
                    forced_region = None
                else:
                    forced_region = int(self._force_disrupt_region)
                    # Skip obviously invalid forced regions to avoid wasting the action.
                    if forced_region in self.region_inked or forced_region in self.town_regions:
                        forced_region = None

            if forced:
                target = forced_region
            else:
                baseline_target = self.get_disruption_target(best_next_path=best_new_path)
                target = baseline_target

                v28_focus_target = self._v28_disrupt_focus_target(
                    best_next_path=best_new_path,
                    opp_threat=float(opp_threat),
                    score_delta=int(score_delta),
                    continue_ev=float(continue_ev),
                    best_new_ev=float(best_new_ev),
                )
                if v28_focus_target is not None:
                    target = int(v28_focus_target)
                else:
                    panic_target = self._panic_disrupt_target(
                        best_next_path=best_new_path,
                        opp_threat=float(opp_threat),
                        score_delta=int(score_delta),
                        baseline_target=baseline_target,
                    )
                    if panic_target is not None:
                        target = int(panic_target)

                if self._disrupt_model_enabled:
                    nn_target = self._get_disruption_target_model(
                        best_next_path=best_new_path,
                        opp_threat=float(opp_threat),
                        score_delta=int(score_delta),
                        continue_ev=float(continue_ev),
                        best_new_ev=float(best_new_ev),
                        baseline_target=baseline_target,
                    )
                    if nn_target is not None:
                        target = nn_target

            if self._dataset_enabled:
                self._dataset_write(
                    {
                        "kind": self._dataset_kind,
                        "seed": self._dataset_seed,
                        "as": self._dataset_as,
                        "turn": int(self.turn),
                        "my_score": int(self.my_score),
                        "opp_score": int(self.opp_score),
                        "score_delta": int(score_delta),
                        "opp_threat": float(opp_threat),
                        "continue_ev": float(continue_ev),
                        "best_new_ev": float(best_new_ev),
                        "active_build": list(self.active_build) if self.active_build else None,
                        "forced": bool(forced),
                        "forced_region": forced_region,
                        "chosen": target,
                        "candidates": self._get_disruption_candidates_for_dataset(best_new_path),
                    }
                )

            if target is not None:
                actions.append(f"DISRUPT {target}")

        out_action = ";".join(actions) if actions else "WAIT"

        if self._trace_enabled:
            try:
                self._trace_last = {
                    "turn": int(self.turn),
                    "my_score": int(self.my_score),
                    "opp_score": int(self.opp_score),
                    "score_delta": int(score_delta),
                    "continue_ev": float(continue_ev),
                    "best_new_ev": float(best_new_ev),
                    "opp_threat": float(opp_threat),
                    "active_build": list(self.active_build) if self.active_build else None,
                    "active_build_invested": int(self._my_tracks_on_path(self.active_build_path)),
                    "active_build_stuck": int(self._active_build_stuck_turns),
                    "tempo_mode": tempo_mode,
                    "endgame_block": bool(endgame_block_fired),
                    "action": str(out_action),
                }
            except Exception:
                self._trace_last = None

        return out_action


StrategicBot = StrategicBotV28


def main():
    bot = StrategicBotV28()
    bot.read_init()
    while True:
        try:
            bot.read_turn()
            action = bot.get_action()
            if getattr(bot, "_trace_enabled", False):
                tl = getattr(bot, "_trace_last", None)
                if isinstance(tl, dict):
                    print("V28_TRACE " + json.dumps(tl, ensure_ascii=False), file=sys.stderr)
        except EOFError:
            break
        except Exception as e:
            # Never crash: engine can hang if a player stops responding.
            try:
                print(f"V28 ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            except Exception:
                pass
            try:
                with open("v28_crash.log", "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"turn={getattr(bot, 'turn', '?')} my={getattr(bot, 'my_score', '?')} opp={getattr(bot, 'opp_score', '?')}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            action = "WAIT"

        print(action)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
