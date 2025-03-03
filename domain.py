# domain.py

from z3 import (
    Solver, sat, Int, And, Not, Or, ForAll, Implies,
    IntSort, BoolSort, Function
)

from typing import List, Tuple, Callable, Optional

class State:
    """
    Z3 representation of a blocks world state.
    Attributes:
        name: A string identifier for the state.
        table: A function mapping block indices (Int) to Bool, indicating if a block is on the table.
        hand: A function mapping block indices (Int) to Bool, indicating if a block is held in hand.
        stacked: A function mapping a pair of block indices (Int, Int) to Bool, indicating if the first block is stacked on the second.
        clear: A function mapping block indices (Int) to Bool, indicating if a block is clear (nothing on top).
        handsfree: A Boolean function indicating if the agent's hand is free.
    """
    def __init__(self, name: str ="s") -> None:
        self.name = name
        self.table = Function(f"{name}_table", IntSort(), BoolSort())
        self.hand = Function(f"{name}_hand", IntSort(), BoolSort())
        self.stacked = Function(f"{name}_stacked", IntSort(), IntSort(), BoolSort())
        self.clear = Function(f"{name}_clear", IntSort(), BoolSort())
        self.handsfree = Function(f"{name}_handsfree", BoolSort())

def pick_up_constraints(s1: State, s2: State, block: int) -> BoolRef:
    """
        Returns Z3 constraints modeling the 'pick up' action,
        where a block is picked up from the table.

        Preconditions (in state s1):
            - The block is on the table.
            - The block is clear.
            - The block is not in the hand.
        Effects (in state s2):
            - The block is in the hand.
            - The block is no longer on the table.
            - The block is not clear.
        Unchanged properties:
            - For all other blocks, the table, hand, and clear predicates remain the same.
            - The stacked relationship remains unchanged.

        Args:
            s1: The pre-action state.
            s2: The post-action state.
            block: The block index to pick up.

        Returns:
            A Z3 Boolean expression representing the pick up constraints.
        """
    x = Int("x_pu")
    y = Int("y_pu")
    return And(
        s1.table(block),
        s1.clear(block),
        Not(s1.hand(block)),

        s2.hand(block),
        Not(s2.table(block)),
        Not(s2.clear(block)),

        ForAll([x],
               Implies(x != block,
                       And(s1.table(x) == s2.table(x),
                           s1.hand(x) == s2.hand(x),
                           s1.clear(x) == s2.clear(x)))),
        ForAll([x, y], s1.stacked(x, y) == s2.stacked(x, y))
    )

def put_down_constraints(s1: State, s2: State, block: int) -> BoolRef:
    """
        Returns Z3 constraints modeling the 'put down' action,
        where a block held in hand is put down onto the table.

        Preconditions (in state s1):
            - The block is in the hand.
        Effects (in state s2):
            - The block is no longer in hand.
            - The block is placed on the table and becomes clear.
        Unchanged properties:
            - For all other blocks, the table, hand, and clear predicates remain the same.
            - The stacked relationship remains unchanged.

        Args:
            s1: The pre-action state.
            s2: The post-action state.
            block: The block index to put down.

        Returns:
            A Z3 Boolean expression representing the put down constraints.
    """
    x = Int("x_pd")
    y = Int("y_pd")
    return And(
        s1.hand(block),
        Not(s2.hand(block)),
        s2.table(block),
        s2.clear(block),

        ForAll([x],
               Implies(x != block,
                       And(s1.table(x) == s2.table(x),
                           s1.hand(x) == s2.hand(x),
                           s1.clear(x) == s2.clear(x)))),
        ForAll([x, y], s1.stacked(x, y) == s2.stacked(x, y))
    )

def stack_constraints(s1: State, s2: State, top_block: int, bottom_block: int) -> BoolRef:
    """
        Returns Z3 constraints modeling the 'stack' action,
        where a block (top_block) is placed on another block (bottom_block).

        Unchanged properties:
            - For all blocks except the top and bottom, their table, hand, and clear statuses remain unchanged.
            - The stacked relation for other pairs remains unchanged.
        Two possible cases are considered:
            Case A: The top block is initially on the table.
            Case B: The top block is initially in hand.

        Args:
            s1: The pre-action state.
            s2: The post-action state.
            top_block: The block to be moved (placed on another block).
            bottom_block: The block on which the top_block is stacked.

        Returns:
            A Z3 Boolean expression representing the stack constraints.
    """
    x = Int("x_st")
    y = Int("y_st")
    unchanged = And(
        ForAll(x,
               Implies(And(x != top_block, x != bottom_block),
                       And(s1.hand(x) == s2.hand(x),
                           s1.table(x) == s2.table(x),
                           s1.clear(x) == s2.clear(x)))),
        ForAll([x, y],
               Implies(And(x != top_block, y != bottom_block),
                       s1.stacked(x, y) == s2.stacked(x, y)))
    )
    caseA = And(
        s1.table(top_block),
        s1.clear(top_block),
        s1.clear(bottom_block),
        s1.handsfree(),

        Not(s2.table(top_block)),
        Not(s2.hand(top_block)),
        s2.stacked(top_block, bottom_block),
        s2.clear(top_block),
        Not(s2.clear(bottom_block)),
        s2.handsfree()
    )
    caseB = And(
        s1.hand(top_block),
        s1.clear(bottom_block),
        Not(s1.handsfree()),

        Not(s2.hand(top_block)),
        Not(s2.table(top_block)),
        s2.stacked(top_block, bottom_block),
        s2.clear(top_block),
        Not(s2.clear(bottom_block)),
        s2.handsfree()
    )
    return And(unchanged, Or(caseA, caseB))

def unstack_constraints(s1: State, s2: State, top_block: int, bottom_block: int) -> BoolRef:
    """
        Returns Z3 constraints modeling the 'unstack' action,
        where a block (top_block) is removed from being stacked on another block (bottom_block).

        Unchanged properties:
            - For all blocks except the top and bottom, their table, hand, and clear statuses remain unchanged.
            - The stacked relation for other pairs remains unchanged.
        Two possible cases are considered:
            Case A: The block is put on the table after unstacking.
            Case B: The block is held in the hand after unstacking.

        Args:
            s1: The pre-action state.
            s2: The post-action state.
            top_block: The block to be unstacked.
            bottom_block: The block from which the top_block is removed.

        Returns:
            A Z3 Boolean expression representing the unstack constraints.
    """
    x = Int("x_us")
    y = Int("y_us")
    unchanged = And(
        ForAll(x,
               Implies(And(x != top_block, x != bottom_block),
                       And(s1.table(x) == s2.table(x),
                           s1.hand(x) == s2.hand(x),
                           s1.clear(x) == s2.clear(x)))),
        ForAll([x, y],
               Implies(And(x != top_block, y != bottom_block),
                       s1.stacked(x, y) == s2.stacked(x, y)))
    )
    caseA = And(
        s1.stacked(top_block, bottom_block),
        s1.clear(top_block),
        s1.handsfree(),
        Not(s2.stacked(top_block, bottom_block)),
        s2.table(top_block),
        s2.clear(top_block),
        s2.clear(bottom_block),
        s2.handsfree()
    )
    caseB = And(
        s1.stacked(top_block, bottom_block),
        s1.clear(top_block),
        s1.handsfree(),
        Not(s2.stacked(top_block, bottom_block)),
        s2.hand(top_block),
        Not(s2.clear(top_block)),
        s2.clear(bottom_block),
        Not(s2.handsfree())
    )
    return And(unchanged, Or(caseA, caseB))

def verify_plan(
        num_blocks: int,
        init_fn: Callable[[State, int], BoolRef],
        goal_fn: Callable[[State, int], BoolRef],
        actions: List[Tuple[str, int, int]]
    ) -> Tuple[str, Optional[ModelRef]]:
    """
    Given a list of (action, b1, b2) tuples, plus an init and goal function,
    build states s0..sN, apply constraints, and check if the final state satisfies the goal.
    Returns ("sat", model) if satisfiable, otherwise ("unsat", None).
    """
    from z3 import Solver, sat
    states = [State(f"s{i}") for i in range(len(actions) + 1)]
    solver = Solver()
    solver.add(init_fn(states[0], num_blocks))
    for i, (act, b1, b2) in enumerate(actions):
        s_cur = states[i]
        s_next = states[i+1]
        if act == "pick_up":
            solver.add(pick_up_constraints(s_cur, s_next, b1))
        elif act == "put_down":
            solver.add(put_down_constraints(s_cur, s_next, b1))
        elif act == "stack":
            solver.add(stack_constraints(s_cur, s_next, b1, b2))
        elif act == "unstack":
            solver.add(unstack_constraints(s_cur, s_next, b1, b2))
        else:
            return "error", None
    solver.add(goal_fn(states[-1], num_blocks))
    result = solver.check()
    if result == sat:
        return "sat", solver.model()
    return "unsat", None
