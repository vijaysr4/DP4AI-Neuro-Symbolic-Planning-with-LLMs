# utils.py

import re

def parse_plan(llm_text: str):
    """
    Parse lines such as:
      1. pick_up 2
      2. stack 2 0
    and return a list of (action, b1, b2) tuples.
    If an action does not require a second block, b2 is set to -1.
    """
    actions = []
    lines = llm_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Remove leading numbering (e.g., "1. ")
        line = re.sub(r"^\d+\.\s*", "", line)
        tokens = line.split()
        if not tokens:
            continue
        action = tokens[0].lower()
        blocks = []
        for token in tokens[1:]:
            m = re.search(r"block(\d+)", token, re.IGNORECASE)
            if m:
                blocks.append(int(m.group(1)))
        b1 = blocks[0] if len(blocks) > 0 else -1
        b2 = blocks[1] if len(blocks) > 1 else -1
        actions.append((action, b1, b2))
    return actions

def find_failing_prefix(num_blocks: int, init_fn, actions) -> int:
    """
    For each prefix of the plan (without the goal constraints), check if it is unsat.
    Return the smallest k for which the prefix fails; if none fail, return -1.
    """
    from domain import State, pick_up_constraints, put_down_constraints, stack_constraints, unstack_constraints
    from z3 import Solver, sat, Int, And, Implies, ForAll
    for k in range(1, len(actions)+1):
        prefix = actions[:k]
        solver = Solver()
        states = [State(f"s{i}") for i in range(k+1)]
        solver.add(init_fn(states[0], num_blocks))
        for i, (act, b1, b2) in enumerate(prefix):
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
                return k
        if solver.check() == sat:
            continue
        else:
            return k
    return -1
