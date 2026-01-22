# Strategic Bot V29: Strategies and Iterative Tuning Process

## Overview
Strategic Bot V29 is a standalone CodinGame bot designed from scratch, with no code reuse from previous versions. Its goal is to compete effectively against earlier bots (e.g., V25) by leveraging advanced planning, opponent modeling, and search heuristics.

## Core Strategies
- **Custom Pathfinding:** Implements A* search for both self and opponent to evaluate shortest connections and plan progress.
- **Plan Selection:** Dynamically selects the most promising connection plan based on current board state and expected value (EV) heuristics.
- **Densification:** Identifies and reinforces weak points in the current plan to increase robustness against opponent disruption.
- **Disruption:** Predicts opponent's likely paths and targets critical cells to block or delay their progress.
- **1-Ply Action Search:** Generates candidate action bundles (builds, disrupts, densifies) and evaluates them using a fast heuristic to select the best move each turn.
- **Opponent Modeling:** Predicts top-K likely opponent connections and assigns weights to cells/regions for more effective disruption.

## Iterative Hyperparameter Tuning
1. **Initial Implementation:** Start with basic plan selection and pathfinding logic.
2. **Evaluation:** Run head-to-head matches (e.g., `eval_v29_vs_v25.py`) to assess win/loss rates and identify weaknesses.
3. **Heuristic Adjustment:** Tune weights for plan progress, disruption, and densification in the action search heuristic.
4. **Candidate Generation:** Experiment with the number and diversity of candidate bundles considered each turn.
5. **Opponent Prediction:** Refine opponent modeling to better anticipate and counter their strategies.
6. **Debugging & Tracing:** Add debug flags and trace outputs to analyze decision-making and diagnose suboptimal behaviors.
7. **Repeat:** Continuously re-evaluate, adjust parameters, and re-run matches until performance improves.

## Evaluation Process
- **Automated Matches:** Use provided evaluation scripts to run large batches of games between bot versions.
- **Metrics:** Track win/loss rates, plan completion rates, and disruption effectiveness.
- **Trace Analysis:** Inspect debug output for specific games to understand and fix poor decisions.
- **Goal:** Achieve a competitive win rate against strong previous versions without code reuse.

---
This iterative process ensures that each new bot version is both innovative and rigorously tested, with strategies and parameters continually refined based on empirical results.