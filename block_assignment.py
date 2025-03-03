# block_assignment.py

from z3 import And, Not, ForAll, Int

def generate_initial_conditions(num_blocks: int):
    """
    Generate initial conditions for a tower configuration:
      - block0 is on the table.
      - For i = 1 to num_blocks-1, block i is stacked on block i-1.
      - Only the top block (block num_blocks-1) is clear.
      - Agent's hand is free.
    """
    def init_fn(s, n: int):
        constraints = []
        x = Int("x_init")
        constraints.append(s.handsfree() == True)
        constraints.append(ForAll([x], Not(s.hand(x))))
        constraints.append(s.table(0) == True)
        for i in range(1, n):
            constraints.append(s.stacked(i, i-1) == True)
            constraints.append(s.table(i) == False)
        for i in range(n-1):
            constraints.append(s.clear(i) == False)
        constraints.append(s.clear(n-1) == True)
        return And(constraints)
    return init_fn

def generate_goal_conditions(num_blocks: int):
    """
    Generate goal conditions for a reversed tower configuration:
      - block(num_blocks-1) is on the table.
      - For i = 0 to num_blocks-2, block i is stacked on block i+1.
      - Only block0 is clear.
      - Agent's hand is free.
    """
    def goal_fn(s, n: int):
        constraints = []
        constraints.append(s.handsfree() == True)
        constraints.append(s.table(n-1) == True)
        for i in range(n-1):
            constraints.append(s.table(i) == False)
        for i in range(n-1):
            constraints.append(s.stacked(i, i+1) == True)
        constraints.append(s.clear(0) == True)
        for i in range(1, n):
            constraints.append(s.clear(i) == False)
        return And(constraints)
    return goal_fn
