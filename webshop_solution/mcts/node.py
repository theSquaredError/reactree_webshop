import math


class MCTSNode:
    """
    Flat MCTS node.  All search-query candidates are direct children of the root,
    so the tree is always depth-1: root → [query_node_1, query_node_2, ...].
    """

    def __init__(self, action: str, parent=None):
        self.action = action          # e.g. "search[gray pu leather desk mat]" or None for root
        self.parent = parent
        self.children: list = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    # ------------------------------------------------------------------ metrics
    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb(self, c: float = 1.414) -> float:
        """Upper Confidence Bound score used during selection."""
        if self.visit_count == 0:
            return float("inf")           # always try unvisited nodes first
        parent_n = self.parent.visit_count if self.parent else self.visit_count
        return self.q + c * math.sqrt(math.log(parent_n + 1) / self.visit_count)

    # ------------------------------------------------------------------ updates
    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value

    # ------------------------------------------------------------------ helpers
    def best_child(self) -> "MCTSNode":
        """Greedy selection by mean Q – used after search to commit to best query."""
        return max(self.children, key=lambda n: n.q)

    def __repr__(self):
        return (
            f"MCTSNode(action={self.action!r}, "
            f"Q={self.q:.3f}, N={self.visit_count})"
        )