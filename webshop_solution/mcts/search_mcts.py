import random
import logging

from webshop_solution.mcts.node import MCTSNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight rollout policy
# ---------------------------------------------------------------------------

_BUY_NOW = "buy now"


def _greedy_rollout(env, max_steps: int = 10) -> float:
    """
    Simulate forward from the current env state using a simple priority policy:
      1. Click 'buy now' immediately if present  → done, collect reward
      2. Otherwise click the first product-link (ASIN) on the page → navigate in
      3. Otherwise click randomly from available actions
    Returns final env reward (0.0–1.0 in WebShop).
    """
    reward = 0.0
    for _ in range(max_steps):
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        if not clickables:
            break

        if _BUY_NOW in clickables:
            _, reward, done, _ = env.step(f"click[{_BUY_NOW}]")
            break

        # Prefer product ASINs (short uppercase/lowercase alphanumeric – not navigation buttons)
        nav_skip = {"back to search", "next >", "< prev", "search"}
        product_links = [c for c in clickables if c not in nav_skip and len(c) <= 12]
        target = product_links[0] if product_links else random.choice(clickables)

        _, reward, done, _ = env.step(f"click[{target}]")
        if done:
            break

    return reward


# ---------------------------------------------------------------------------
# Focused MCTS: choose best search query from Expand(parallel) candidates
# ---------------------------------------------------------------------------

class SearchQueryMCTS:
    """
    Runs MCTS over a flat set of search query candidates that come from an
    Expand(parallel, subgoals=[...]) decision in the Reactree agent.

    Tree structure:
        root  (no action)
         ├─ node_0  action="search[query_0]"
         ├─ node_1  action="search[query_1]"
         └─ node_2  action="search[query_2]"

    Each MCTS iteration:
      Selection  → pick child with highest UCB
      Simulation → restore env to base_state, apply search action, run _greedy_rollout
      Backup     → propagate reward to node and root

    After `budget` iterations, the child with highest mean Q is returned as
    the best action.  The root node is also returned for preference-pair extraction.
    """

    def __init__(self, env, budget: int = 8, max_rollout_steps: int = 10, c: float = 1.414):
        self.env = env
        self.budget = budget
        self.max_rollout_steps = max_rollout_steps
        self.c = c

    # ------------------------------------------------------------------ main entrypoint
    def run(self, base_state, query_candidates: list[str]):
        """
        Args:
            base_state       : WebShopState snapshot before any search action.
            query_candidates : plain-text queries, e.g. ["gray pu desk mat", "non slip desk pad"].
                               Do NOT include the "search[...]" wrapper here.

        Returns:
            best_action : str  – e.g. "search[gray pu desk mat]"
            root        : MCTSNode – full tree for preference-pair extraction
        """
        if not query_candidates:
            raise ValueError("query_candidates must be non-empty")

        # Build flat tree: root + one child per candidate
        root = MCTSNode(action=None)
        root.visit_count = 1  # avoid log(0) in ucb during first selection
        for q in query_candidates:
            action = q if q.startswith("search[") else f"search[{q}]"
            root.children.append(MCTSNode(action=action, parent=root))

        for i in range(self.budget):
            # ---- Selection: UCB over direct children ------------------------
            node = max(root.children, key=lambda n: n.ucb(self.c))

            # ---- Simulation -------------------------------------------------
            reward = self._simulate(base_state, node.action)
            logger.info(
                f"[SearchQueryMCTS] iter={i+1}/{self.budget} "
                f"action={node.action!r} reward={reward:.4f}"
            )

            # ---- Backup -----------------------------------------------------
            node.update(reward)
            root.update(reward)

        # Restore the real env to base_state so the caller can continue
        base_state.restore_env(self.env)

        best_node = root.best_child()
        logger.info(f"[SearchQueryMCTS] best={best_node.action!r} Q={best_node.q:.4f}")
        return best_node.action, root

    def _simulate(self, base_state, search_action: str) -> float:
        """Restore env → apply search action → rollout → return reward."""
        base_state.restore_env(self.env)
        _, reward, done, _ = self.env.step(search_action)
        if done:
            return reward
        return _greedy_rollout(self.env, self.max_rollout_steps)

    # ------------------------------------------------------------------ preference pairs
    @staticmethod
    def build_preference_pairs(root: MCTSNode):
        """
        Produce ranked list and all (chosen > rejected) preference pairs from the MCTS tree.

        Returns:
            ranked : list[dict]  – sorted by Q descending
            pairs  : list[dict]  – every (winner, loser) combination
        """
        visited = [c for c in root.children if c.visit_count > 0]
        ranked = sorted(
            [{"action": c.action, "q": round(c.q, 4), "visits": c.visit_count} for c in visited],
            key=lambda x: x["q"],
            reverse=True,
        )
        pairs = [
            {
                "chosen": ranked[i]["action"],
                "rejected": ranked[j]["action"],
                "chosen_q": ranked[i]["q"],
                "rejected_q": ranked[j]["q"],
                "chosen_visits": ranked[i]["visits"],
                "rejected_visits": ranked[j]["visits"],
            }
            for i in range(len(ranked))
            for j in range(i + 1, len(ranked))
        ]
        return ranked, pairs