"""
World model — discovers transition rules from agent-environment interactions.

The world model maintains a set of **transition rules**, each of which is a
compositional expression that predicts the next state given the current state
and action. This is the Phase 2 analogue of Phase 1's function discovery:

    Phase 1: f(x) → y          (discover the generating function)
    Phase 2: T(state, action) → next_state  (discover the transition rules)

The model learns by:
1. Observing (state, action, next_state) transitions
2. Searching for compositional rules that explain the transitions
3. Using prediction accuracy + simplicity (energy) to rank rules
4. Selecting actions that maximize information gain (exploration)

Rule representation:
    Each rule is a conditional transition:
    IF condition(state, action) THEN next_state = transform(state)

    Conditions and transforms are compositions of discrete primitives:
    - position_equals(r, c)
    - action_is(UP/DOWN/LEFT/RIGHT)
    - adjacent(direction)
    - is_wall, is_empty, is_portal, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agi_composer.worlds.grid_env import Action, CellType, Observation


@dataclass
class Transition:
    """A single observed transition."""
    state: Tuple[int, int]
    action: Action
    next_state: Tuple[int, int]
    reward: float
    done: bool
    inventory_before: frozenset = frozenset()
    inventory_after: frozenset = frozenset()


@dataclass
class TransitionRule:
    """
    A discovered rule about how the world works.

    A rule says: "When you take action A from state S, you end up in state S'
    and get reward R."

    Rules can be:
    - Specific: "Moving UP from (1,0) leads to (0,0)"
    - General: "Moving UP decreases row by 1 (unless there's a wall)"
    - Compositional: "Stepping on a portal teleports you to its destination"

    The confidence reflects how many observations support this rule.
    """
    description: str
    condition: str         # Human-readable condition
    prediction: str        # Human-readable prediction
    confidence: float      # 0..1, how many observations support this
    num_supporting: int    # Number of supporting transitions
    num_contradicting: int # Number of contradicting transitions
    rule_type: str         # "movement", "wall", "portal", "key", "door", "goal"

    @property
    def accuracy(self) -> float:
        total = self.num_supporting + self.num_contradicting
        return self.num_supporting / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"Rule({self.description}, "
                f"acc={self.accuracy:.0%}, n={self.num_supporting})")


class WorldModel:
    """
    Learns a compositional model of the world from observations.

    The model discovers rules at three levels of abstraction:
    1. **Specific transitions**: (state, action) → next_state lookup table
    2. **General movement rules**: action deltas that apply everywhere
    3. **Special rules**: portals, keys, doors (compositional exceptions)

    This mirrors how a baby might learn:
    - First: memorize specific outcomes
    - Then: notice patterns (moving UP always decreases row)
    - Finally: discover exceptions (portals break the pattern)
    """

    def __init__(self):
        # Raw transition memory
        self.transitions: List[Transition] = []

        # Level 1: Specific transition table
        # Maps (state, action) → list of observed next_states
        self._transition_table: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}

        # Level 2: General movement model
        # Maps action → (delta_row, delta_col) if consistent
        self._movement_model: Dict[int, Optional[Tuple[int, int]]] = {
            a.value: None for a in Action.all()
        }

        # Level 3: Discovered special rules
        self._wall_positions: Set[Tuple[int, int]] = set()
        self._portal_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._goal_positions: Set[Tuple[int, int]] = set()
        self._key_positions: Dict[Tuple[int, int], bool] = {}  # pos → collected?
        self._door_positions: Set[Tuple[int, int]] = set()
        self._boundary: Optional[Tuple[int, int]] = None  # (max_row, max_col)

        # Discovered rules (for reporting)
        self.rules: List[TransitionRule] = []

        # Exploration tracking
        self._visited_states: Set[Tuple[int, int]] = set()
        self._visited_transitions: Set[Tuple[Tuple[int, int], int]] = set()

    def observe(self, state: Tuple[int, int], action: Action,
                observation: Observation,
                inventory_before: frozenset = frozenset()) -> None:
        """
        Record a transition and update the world model.

        This is the core learning loop — each observation refines the model.
        """
        next_state = observation.position
        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=observation.reward,
            done=observation.done,
            inventory_before=inventory_before,
            inventory_after=observation.inventory,
        )
        self.transitions.append(transition)

        # Update tracking
        self._visited_states.add(state)
        self._visited_states.add(next_state)
        self._visited_transitions.add((state, action.value))

        # Update specific transition table
        key = (state, action.value)
        if key not in self._transition_table:
            self._transition_table[key] = []
        self._transition_table[key].append(next_state)

        # Update boundary estimate
        max_r = max(state[0], next_state[0])
        max_c = max(state[1], next_state[1])
        if self._boundary is None:
            self._boundary = (max_r, max_c)
        else:
            self._boundary = (
                max(self._boundary[0], max_r),
                max(self._boundary[1], max_c),
            )

        # --- Level 2: Discover general movement rules ---
        self._update_movement_model(state, action, next_state)

        # --- Level 3: Discover special rules ---
        self._detect_walls(state, action, next_state, observation.reward)
        self._detect_portals(state, action, next_state)
        self._detect_goals(next_state, observation)
        self._detect_keys(state, next_state, inventory_before, observation.inventory)
        self._detect_doors(state, action, next_state, observation.reward,
                           inventory_before)

    def _update_movement_model(self, state: Tuple[int, int], action: Action,
                               next_state: Tuple[int, int]) -> None:
        """Discover the general movement delta for each action."""
        if state == next_state:
            # Agent didn't move — wall or boundary hit, skip for delta learning
            return

        delta = (next_state[0] - state[0], next_state[1] - state[1])

        # Check if this is a "normal" move (adjacent cell)
        if abs(delta[0]) + abs(delta[1]) == 1:
            current = self._movement_model[action.value]
            if current is None:
                self._movement_model[action.value] = delta
            elif current != delta:
                # Inconsistent — this action doesn't have a single delta
                # (shouldn't happen for normal movement)
                pass

    def _detect_walls(self, state: Tuple[int, int], action: Action,
                      next_state: Tuple[int, int], reward: float) -> None:
        """Detect walls: agent tried to move but stayed in place with penalty."""
        if state == next_state and reward < 0:
            # The cell the agent tried to enter is probably a wall
            dr, dc = action.delta()
            wall_pos = (state[0] + dr, state[1] + dc)
            # Could also be a boundary — check if it's within our known bounds
            if self._boundary and (wall_pos[0] < 0 or wall_pos[1] < 0 or
                                    wall_pos[0] > self._boundary[0] + 1 or
                                    wall_pos[1] > self._boundary[1] + 1):
                pass  # Likely a boundary, not a wall cell
            else:
                self._wall_positions.add(wall_pos)

    def _detect_portals(self, state: Tuple[int, int], action: Action,
                        next_state: Tuple[int, int]) -> None:
        """Detect portals: movement resulted in non-adjacent transition."""
        delta = (next_state[0] - state[0], next_state[1] - state[1])
        manhattan = abs(delta[0]) + abs(delta[1])
        if manhattan > 1 and state != next_state:
            # Non-local transition — likely a portal
            # The portal is at the cell the agent entered
            dr, dc = action.delta()
            portal_pos = (state[0] + dr, state[1] + dc)
            self._portal_map[portal_pos] = next_state

    def _detect_goals(self, next_state: Tuple[int, int],
                      obs: Observation) -> None:
        """Detect goals: high reward and episode ends."""
        if obs.done and obs.reward > 0:
            self._goal_positions.add(next_state)

    def _detect_keys(self, state: Tuple[int, int],
                     next_state: Tuple[int, int],
                     inv_before: frozenset, inv_after: frozenset) -> None:
        """Detect keys: inventory changed after moving to a cell."""
        new_items = inv_after - inv_before
        if new_items:
            self._key_positions[next_state] = True

    def _detect_doors(self, state: Tuple[int, int], action: Action,
                      next_state: Tuple[int, int], reward: float,
                      inventory: frozenset) -> None:
        """Detect doors: blocked without key, passable with key."""
        if state == next_state and reward < 0:
            dr, dc = action.delta()
            blocked_pos = (state[0] + dr, state[1] + dc)
            # If we've seen this position passable before, it might be a door
            if blocked_pos not in self._wall_positions:
                self._door_positions.add(blocked_pos)

    def predict(self, state: Tuple[int, int],
                action: Action) -> Tuple[Tuple[int, int], float]:
        """
        Predict the next state and reward for a given state-action pair.

        Returns (predicted_next_state, confidence).
        Confidence is 0..1, where 0 means "no idea" and 1 means "certain".
        """
        key = (state, action.value)

        # Level 1: Check specific memory
        if key in self._transition_table:
            outcomes = self._transition_table[key]
            # Return most common outcome
            from collections import Counter
            counts = Counter(outcomes)
            best = counts.most_common(1)[0]
            confidence = best[1] / len(outcomes)
            return best[0], confidence

        # Level 2: Use general movement model
        delta = self._movement_model[action.value]
        if delta is not None:
            predicted = (state[0] + delta[0], state[1] + delta[1])

            # Check if predicted position is a known wall
            if predicted in self._wall_positions:
                return state, 0.7  # Predict staying in place

            # Check if predicted position is a known portal
            if predicted in self._portal_map:
                return self._portal_map[predicted], 0.8

            return predicted, 0.5  # Medium confidence in general rule

        # Level 3: No information — return current state with no confidence
        return state, 0.0

    def get_uncertainty(self, state: Tuple[int, int],
                        action: Action) -> float:
        """
        How uncertain are we about this state-action pair?

        Returns 0..1, where 1 means maximum uncertainty (never tried).
        Used by the Explorer to select informative actions.
        """
        key = (state, action.value)

        if key in self._transition_table:
            # Seen before — low uncertainty
            n = len(self._transition_table[key])
            return max(0.0, 1.0 - n * 0.3)  # Decreases with more observations

        if state not in self._visited_states:
            return 1.0  # Never been here

        # Been to state but not tried this action
        return 0.8

    def get_discovered_rules(self) -> List[TransitionRule]:
        """Compile all discovered rules into a human-readable list."""
        rules = []

        # Movement rules
        action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        for action_val, delta in self._movement_model.items():
            if delta is not None:
                rules.append(TransitionRule(
                    description=f"{action_names[action_val]} moves by {delta}",
                    condition=f"action == {action_names[action_val]}",
                    prediction=f"position += {delta}",
                    confidence=0.9,
                    num_supporting=sum(
                        1 for t in self.transitions
                        if t.action.value == action_val and
                        t.next_state != t.state and
                        (t.next_state[0] - t.state[0],
                         t.next_state[1] - t.state[1]) == delta
                    ),
                    num_contradicting=0,
                    rule_type="movement",
                ))

        # Wall rules
        for wall_pos in self._wall_positions:
            rules.append(TransitionRule(
                description=f"Wall at {wall_pos}",
                condition=f"target == {wall_pos}",
                prediction="position unchanged, negative reward",
                confidence=0.8,
                num_supporting=1,
                num_contradicting=0,
                rule_type="wall",
            ))

        # Portal rules
        for portal_pos, dest in self._portal_map.items():
            rules.append(TransitionRule(
                description=f"Portal at {portal_pos} → {dest}",
                condition=f"entering {portal_pos}",
                prediction=f"teleport to {dest}",
                confidence=0.9,
                num_supporting=1,
                num_contradicting=0,
                rule_type="portal",
            ))

        # Goal rules
        for goal_pos in self._goal_positions:
            rules.append(TransitionRule(
                description=f"Goal at {goal_pos}",
                condition=f"position == {goal_pos}",
                prediction="episode ends, positive reward",
                confidence=1.0,
                num_supporting=1,
                num_contradicting=0,
                rule_type="goal",
            ))

        # Key rules
        for key_pos in self._key_positions:
            rules.append(TransitionRule(
                description=f"Key at {key_pos}",
                condition=f"entering {key_pos}",
                prediction="key added to inventory",
                confidence=0.9,
                num_supporting=1,
                num_contradicting=0,
                rule_type="key",
            ))

        self.rules = rules
        return rules

    @property
    def num_states_visited(self) -> int:
        return len(self._visited_states)

    @property
    def num_transitions_observed(self) -> int:
        return len(self.transitions)

    @property
    def num_unique_transitions(self) -> int:
        return len(self._visited_transitions)

    def summary(self) -> str:
        """Human-readable summary of the learned world model."""
        rules = self.get_discovered_rules()
        lines = [
            "═" * 50,
            "  World Model Summary",
            "═" * 50,
            f"  States visited:      {self.num_states_visited}",
            f"  Transitions observed: {self.num_transitions_observed}",
            f"  Unique transitions:   {self.num_unique_transitions}",
            f"  Rules discovered:     {len(rules)}",
            "",
        ]
        for rule in rules:
            lines.append(f"  [{rule.rule_type:8s}] {rule.description}")
        lines.append("═" * 50)
        return "\n".join(lines)
