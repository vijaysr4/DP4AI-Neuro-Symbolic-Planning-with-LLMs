import sys
from model_selector import get_planner
from block_assignment import generate_initial_conditions, generate_goal_conditions
from domain import verify_plan
from utils import parse_plan, find_failing_prefix

def initial_state_description(num_blocks: int) -> str:
    blocks = [f"block{i}" for i in range(num_blocks)]
    description = (
        f"Initial state:\n"
        f"- A tower with {blocks[0]} at the bottom, then " +
        ", ".join(blocks[1:]) +
        f".\n- Agent's hand is free."
    )
    return description

def goal_state_description(num_blocks: int) -> str:
    blocks = [f"block{i}" for i in range(num_blocks)]
    description = (
        f"Goal state:\n"
        f"- {blocks[-1]} is on the table.\n" +
        "\n".join([f"- {blocks[i]} is stacked on {blocks[i+1]}" for i in range(num_blocks - 1)]) +
        f"\n- Only {blocks[0]} is clear."
    )
    return description

def main():
    # Usage: python main.py <model_choice> <num_blocks>
    # Example: python main.py gpt4 5
    if len(sys.argv) < 3:
        print("Usage: python main.py <model_choice> <num_blocks>")
        print("  e.g. python main.py gpt4 5")
        return

    model_choice = sys.argv[1]
    num_blocks = int(sys.argv[2])

    # Replace these keys with your actual keys:
    openai_api_key = "..." # Insert your actual OpenAI API key here
    hf_token = ".." # Insert your actual Hugging Face token here

    model_choice_lower = model_choice.strip().lower()
    if model_choice_lower in ["gpt4", "gpt-4o-mini", "gpt-4o"]:
        planner = get_planner(model_choice_lower, openai_api_key=openai_api_key, temperature=0.7)
    elif model_choice_lower in ["llama-3.1-8b-instruct", "deepseek-r1-7b"]:
        planner = get_planner(model_choice_lower, hf_token=hf_token, temperature=0.7)
    else:
        print(f"Unsupported model: {model_choice}")
        return

    # Generate initial and goal condition functions for the tower reversal example.
    init_fn = generate_initial_conditions(num_blocks)
    goal_fn = generate_goal_conditions(num_blocks)

    # Generate natural language descriptions for initial and goal states.
    init_description = initial_state_description(num_blocks)
    goal_description = goal_state_description(num_blocks)

    # Print the generated conditions.
    print("=== Generated Initial Conditions ===")
    print(init_description)
    print("\n=== Generated Goal Conditions ===")
    print(goal_description)
    print("====================================\n")

    # Construct the base prompt using the generated descriptions.
    allowed_actions = (
        "Allowed actions:\n"
        "  pick_up <block>\n"
        "  put_down <block>\n"
        "  stack <blockA> <blockB>\n"
        "  unstack <blockA> <blockB>"
    )

    base_prompt = f"""
We have a blocksworld with {num_blocks} blocks labeled 0..{num_blocks-1}.

{init_description}

{goal_description}

{allowed_actions}

Please produce a valid plan in at most {2 * num_blocks} steps.

Now produce YOUR plan as a numbered list:
""".strip()

    # The system message instructs the LLM on what output is expected.
    system_msg = "You are a helpful planner that outputs only a numbered list of actions for a blocksworld puzzle."

    max_iterations = 20
    iteration = 1

    # We will maintain an aggregated feedback string.
    aggregated_feedback = ""

    # Initially, use the base prompt.
    user_prompt = base_prompt

    while iteration <= max_iterations:
        print(f"\n=== Iteration {iteration} ===")
        print("Current Prompt:")
        print(user_prompt)
        print("====================================")

        response_text = planner.generate_plan(user_prompt, system_message=system_msg)
        print(f"\nAttempt #{iteration} response:")
        print(response_text)

        actions = parse_plan(response_text)
        result, _model = verify_plan(num_blocks, init_fn, goal_fn, actions)

        if result == "sat":
            print(f"\nPlan verified on iteration {iteration}!")
            print("Final Plan =", actions)
            return
        else:
            # Identify failing prefix to create counterexample feedback.
            fail_k = find_failing_prefix(num_blocks, init_fn, actions)
            if fail_k != -1:
                # Construct more specific feedback.
                feedback_detail = f"Step {fail_k} does not meet the required conditions (e.g., the state transition is invalid)."
                current_feedback = f"Your plan fails at step {fail_k}: {feedback_detail}"
            else:
                current_feedback = "Your plan fails to achieve the goal state. Please adjust the sequence of actions."

            # Aggregate the feedback.
            aggregated_feedback += "\n" + current_feedback

            # Print the feedback.
            print("\nCounterexample feedback:")
            print(current_feedback)

            iteration += 1
            if iteration > max_iterations:
                print("No verified plan found after maximum attempts.")
                return

            # Update the prompt: we use the base prompt plus an aggregated feedback section.
            user_prompt = (
                f"{base_prompt}\n\nFeedback from previous iterations:\n"
                f"{aggregated_feedback}\n\nPlease revise your plan accordingly."
            )

if __name__ == "__main__":
    main()
