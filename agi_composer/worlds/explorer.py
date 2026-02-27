"""
Explorer agent — selects actions by balancing exploration and exploitation.

The Explorer implements the core feedback loop of Phase 2:

    observe → update world model → plan → act → observe ...

Action selection balances two objectives:
1. **Exploitation**: Move toward the goal using the current world model
2. **Exploration**: Try actions/states with high uncertainty to learn more

This balance is controlled by a curiosity parameter that decays over time,
mirroring the simulated annealing temperature from Phase 1:
- Early: high curiosity → explore unknown states
- Late: low curiosity → exploit known path to goal

The Explorer also implements **information gain** as an intrinsic reward:
discovering a new rule (portal, key, door) is rewarding in itself,
encouraging the agent to seek surprising transitions.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agi_composer.worlds.grid_env import Action, GridWorld, Observation
from agi_composer.worlds.world_model import WorldModel, TransitionRule


@dataclass
class ExplorerConfig:
    """Configuration for the Explorer agent."""
    initial_curiosity: float = 1.0     # Starting exploration weight
    curiosity_decay: float = 0.995     # Decay per episode
    min_curiosity: float = 0.05        # Floor curiosity
    planning_depth: int = 10           # How far ahead to plan
    episodes_per_level: int = 50       # Episodes before declaring solved
    solve_threshold: int = 3           # Consecutive goal-reaches to declare solved
    seed: Optional[int] = None


@dataclass
class EpisodeLog:
    """Record of a single episode."""
    steps: int
    total_reward: float
    reached_goal: bool
    path: List[Tuple[int, int]]
    actions: List[Action]
    new_rules_discovered: int


@dataclass
class ExplorationResult:
    """Result of exploring a grid world."""
    solved: bool
    episodes_used: int
    world_model: WorldModel
    rules: List[TransitionRule]
    episode_logs: List[EpisodeLog]
    best_path: Optional[List[Action]]
    best_path_length: Optional[int]

    def summary(self) -> str:
        lines = [
            "═" * 55,
            "  Explorer — Exploration Result",
            "═" * 55,
            f"  Solved:            {'Yes' if self.solved else 'No'}",
            f"  Episodes used:     {self.episodes_used}",
            f"  Rules discovered:  {len(self.rules)}",
            f"  States explored:   {self.world_model.num_states_visited}",
            f"  Best path length:  {self.best_path_length or 'N/A'}",
            "",
        ]
        if self.episode_logs:
            rewards = [e.total_reward for e in self.episode_logs]
            goal_episodes = [e for e in self.episode_logs if e.reached_goal]
            lines.append(f"  Avg reward:        {np.mean(rewards):.2f}")
            lines.append(f"  Goal reached:      {len(goal_episodes)}/{len(self.episode_logs)} episodes")
            if goal_episodes:
                path_lens = [e.steps for e in goal_episodes]
                lines.append(f"  Avg goal path:     {np.mean(path_lens):.1f} steps")
                lines.append(f"  Best goal path:    {min(path_lens)} steps")
        lines.append("")
        lines.append("  Discovered rules:")
        for rule in self.rules:
            lines.append(f"    [{rule.rule_type:8s}] {rule.description}")
        lines.append("═" * 55)
        return "\n".join(lines)


class Explorer:
    """
    An agent that explores grid worlds to discover their rules.

    The exploration strategy:
    1. Start with random exploration (high curiosity)
    2. Build a world model from observations
    3. Gradually shift to goal-directed behavior
    4. Use information gain as intrinsic motivation

    This directly implements the 4 pillars:
    - Feedback loop: observations update model, model guides actions
    - Approximability: world model is refined with each observation
    - Composability: rules are composed (movement + wall + portal)
    - Exploration: curiosity drives the agent to discover new rules
    """

    def __init__(self, config: Optional[ExplorerConfig] = None):
        self.config = config or ExplorerConfig()
        self.rng = random.Random(self.config.seed)
        self.world_model = WorldModel()

    def explore(self, env: GridWorld,
                verbose: bool = False) -> ExplorationResult:
        """
        Explore a grid world across multiple episodes.

        Each episode:
        1. Reset the environment
        2. Select actions using curiosity-weighted strategy
        3. Observe and update world model
        4. Track whether the goal was reached

        Returns when solved (consistent goal-reaching) or max episodes hit.
        """
        curiosity = self.config.initial_curiosity
        episode_logs: List[EpisodeLog] = []
        consecutive_solves = 0
        best_path: Optional[List[Action]] = None
        best_path_length: Optional[int] = None

        for episode in range(self.config.episodes_per_level):
            log = self._run_episode(env, curiosity)
            episode_logs.append(log)

            if log.reached_goal:
                consecutive_solves += 1
                if best_path_length is None or log.steps < best_path_length:
                    best_path = log.actions
                    best_path_length = log.steps
            else:
                consecutive_solves = 0

            # Decay curiosity
            curiosity = max(
                self.config.min_curiosity,
                curiosity * self.config.curiosity_decay,
            )

            if verbose and episode % 5 == 0:
                goal_str = "✓" if log.reached_goal else "✗"
                print(
                    f"  [ep {episode:3d}] {goal_str} "
                    f"steps={log.steps:3d}  "
                    f"reward={log.total_reward:7.1f}  "
                    f"curiosity={curiosity:.3f}  "
                    f"rules={len(self.world_model.get_discovered_rules()):2d}  "
                    f"states={self.world_model.num_states_visited:3d}"
                )

            # Check if solved
            if consecutive_solves >= self.config.solve_threshold:
                if verbose:
                    print(f"  [ep {episode}] Solved! "
                          f"{consecutive_solves} consecutive goals reached.")
                break

        rules = self.world_model.get_discovered_rules()
        solved = consecutive_solves >= self.config.solve_threshold

        return ExplorationResult(
            solved=solved,
            episodes_used=len(episode_logs),
            world_model=self.world_model,
            rules=rules,
            episode_logs=episode_logs,
            best_path=best_path,
            best_path_length=best_path_length,
        )

    def _run_episode(self, env: GridWorld,
                     curiosity: float) -> EpisodeLog:
        """Run a single episode of exploration."""
        obs = env.reset()
        path = [obs.position]
        actions = []
        rules_before = len(self.world_model.get_discovered_rules())

        while not obs.done:
            state = obs.position
            inventory = obs.inventory
            action = self._select_action(state, curiosity, env)
            actions.append(action)

            obs = env.step(action)
            self.world_model.observe(state, action, obs, inventory)
            path.append(obs.position)

        rules_after = len(self.world_model.get_discovered_rules())

        return EpisodeLog(
            steps=env.steps,
            total_reward=env.total_reward,
            reached_goal=obs.reward > 0 and obs.done,
            path=path,
            actions=actions,
            new_rules_discovered=rules_after - rules_before,
        )

    def _select_action(self, state: Tuple[int, int],
                       curiosity: float, env: GridWorld) -> Action:
        """
        Select an action balancing exploration and exploitation.

        Score(action) = (1 - curiosity) * exploitation_value
                      + curiosity * exploration_value

        Exploitation: prefer actions that move toward known goals
        Exploration: prefer actions with high uncertainty (information gain)
        """
        all_actions = Action.all()
        scores = []

        for action in all_actions:
            exploit_score = self._exploitation_score(state, action)
            explore_score = self._exploration_score(state, action)

            combined = ((1 - curiosity) * exploit_score +
                        curiosity * explore_score)
            scores.append(combined)

        # Softmax selection with temperature
        scores_arr = np.array(scores, dtype=float)
        # Prevent overflow in softmax
        scores_arr = scores_arr - np.max(scores_arr)
        probs = np.exp(scores_arr * 3.0)  # Temperature scaling
        probs = probs / (probs.sum() + 1e-10)

        # Sample from distribution
        idx = self.rng.choices(range(len(all_actions)),
                               weights=probs.tolist(), k=1)[0]
        return all_actions[idx]

    def _exploitation_score(self, state: Tuple[int, int],
                            action: Action) -> float:
        """
        Score an action by how much it moves toward the goal.

        Uses the world model's predictions to evaluate outcomes.
        """
        predicted_next, confidence = self.world_model.predict(state, action)

        if confidence < 0.1:
            return 0.0  # No information — neutral

        # Check if predicted next state is a known goal
        if predicted_next in self.world_model._goal_positions:
            return 10.0

        # If we know the goal positions, compute distance reduction
        if self.world_model._goal_positions:
            goal = next(iter(self.world_model._goal_positions))
            current_dist = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
            next_dist = abs(predicted_next[0] - goal[0]) + abs(predicted_next[1] - goal[1])
            return (current_dist - next_dist) * confidence
        
        # No goal known yet — prefer moving to new states
        if predicted_next not in self.world_model._visited_states:
            return 0.5
        return 0.0

    def _exploration_score(self, state: Tuple[int, int],
                           action: Action) -> float:
        """
        Score an action by its expected information gain.

        High score for:
        - Actions never tried from this state
        - States never visited
        - High model uncertainty
        """
        uncertainty = self.world_model.get_uncertainty(state, action)

        # Bonus for potentially reaching unvisited states
        predicted_next, confidence = self.world_model.predict(state, action)
        novelty_bonus = 0.0
        if predicted_next not in self.world_model._visited_states:
            novelty_bonus = 0.5

        return uncertainty + novelty_bonus
