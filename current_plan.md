1. Hierarchical Tree Construction (Reactree Replication):
2. Multi-Decomposition Expansion using Reactree -> These candidates form the branching structure of the search space.
3. MCTS over Hierarchical Space:
    1. Nodes: Subgoal decompositions or executable actions
    2. Selection is guided by UCB based criterion


4. Expansion strategy 
    1. At each step "Should I decompose or act?"
    

    - Subgoal decomposition is going to be with a control flow, which needs to considered while executing the trajectory or going to next step.
    - Selection, expansion, simulation, backup, all needs to be defined by considering subgoals together with actions in the tree for MCTS.
    - How to create preference pairs at subgoal, action, and trajectory, and trajectory levels.
    - Training using three different kinds of pairs
