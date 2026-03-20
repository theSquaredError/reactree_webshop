# Reactree-MCTS
The current workflow includes:
1. The agent is given user instruction query
2. Agent is asked to act as a shopping agent and has to buy the product matching the user instruction.
3. It has design it's own search query and navigate through page.

There are three expand modes in reactree:
    1. sequence - follow one by one, if one fails whole sequence fails
    2. fallback - (attempt subgoals in order until one succeeds)
    3. parallel: achieve subgoals in parallel (this enables tasks to continue independtly, even if one subgoal fails)

Now in webshop, best search query need not include 


### Think out to improve search query which is actually put in the search box
changed the prompt to adjust for search query, 


### Current Challenges:
- parallel can be used to explore multiple products 
- fallback can be used if any one of the condition is failed, then backtrack
- sequence can be used to in situation's where 

### Current Status:
1. In the first search query the agent is almost giving whole user instruction and then further deciding to decompose the goal.