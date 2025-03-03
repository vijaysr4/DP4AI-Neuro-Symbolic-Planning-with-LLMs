import csv
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
        "\n".join([f"- {blocks[i]} is stacked on {blocks[i + 1]}" for i in range(num_blocks - 1)]) +
        f"\n- Only {blocks[0]} is clear."
    )
    return description

def generate_base_prompt(num_blocks: int) -> str:
    init_desc = initial_state_description(num_blocks)
    goal_desc = goal_state_description(num_blocks)
    allowed_actions = (
        "Allowed actions:\n"
        "  pick_up <block>\n"
        "  put_down <block>\n"
        "  stack <blockA> <blockB>\n"
        "  unstack <blockA> <blockB>"
    )
    base_prompt = f"""We have a blocksworld with {num_blocks} blocks labeled 0..{num_blocks - 1}.

{init_desc}

{goal_desc}

{allowed_actions}

Please produce a valid plan in at most {2 * num_blocks} steps.

Now produce YOUR plan as a numbered list:"""
    return base_prompt

def run_planning(model, num_blocks, max_iterations, enhanced: bool):
    """
    Runs the planning process for a given problem size using the specified model.

    Parameters:
      - model: an instance of the planner from model_selector.
      - num_blocks: number of blocks in the blocksworld.
      - max_iterations: maximum allowed iterations.
      - enhanced: if True, uses dynamic prompt refinement with aggregated feedback; otherwise, uses the static prompt.

    Returns:
      A tuple (success, iterations, final_response, actions) where:
        - success: True if a verified plan is found.
        - iterations: number of iterations taken.
        - final_response: the last response text from the model.
        - actions: the parsed actions.
    """
    # Generate the initial and goal constraints for the current problem.
    init_fn = generate_initial_conditions(num_blocks)
    goal_fn = generate_goal_conditions(num_blocks)

    base_prompt = generate_base_prompt(num_blocks)
    user_prompt = base_prompt
    aggregated_feedback = ""
    iterations = 0
    success = False
    system_msg = "You are a helpful planner that outputs only a numbered list of actions for a blocksworld puzzle."

    while iterations < max_iterations:
        iterations += 1
        # Generate a plan with the current prompt.
        response_text = model.generate_plan(user_prompt, system_message=system_msg)
        actions = parse_plan(response_text)
        result, _model = verify_plan(num_blocks, init_fn, goal_fn, actions)
        if result == "sat":
            success = True
            break
        else:
            # Use the feedback function to identify the failing prefix.
            fail_k = find_failing_prefix(num_blocks, init_fn, actions)
            if fail_k != -1:
                current_feedback = f"Your plan fails at step {fail_k}: Step {fail_k} does not meet required conditions."
            else:
                current_feedback = "Your plan fails to achieve the goal state."
            # If enhanced, aggregate feedback; baseline does not update the prompt.
            if enhanced:
                aggregated_feedback += "\n" + current_feedback
                user_prompt = (
                    f"{base_prompt}\n\nFeedback from previous iterations:\n"
                    f"{aggregated_feedback}\n\nPlease revise your plan accordingly."
                )
            else:
                user_prompt = base_prompt  # Baseline: use static prompt.
    return success, iterations, response_text, actions

# ---------------- Experiment Runner ----------------

def run_tests():
    """
    Runs experiments for different problem sizes and configurations (baseline vs enhanced)
    and saves the results in a CSV file.
    """
    # Set up API keys / tokens here.
    openai_api_key = "..."  # Replace with your actual OpenAI API key.
    hf_token = "..."  # Replace with your actual Hugging Face token.

    # Choose a model. You can switch between "gpt4" and, e.g., "deepseek-r1-7b".
    model_choice = "gpt-4o"  # Change as needed.
    if model_choice.lower() == "gpt-4o":
        # For GPT-4, we need to provide openai_api_key.
        model = get_planner("gpt-4o", openai_api_key=openai_api_key, temperature=0.7)
    else:
        model = get_planner(model_choice, hf_token=hf_token, temperature=0.7)

    problem_sizes = [3, 4, 5]
    config_names = ["baseline", "enhanced"]
    runs_per_config = 10  # Number of independent runs per configuration
    max_iterations = 25

    results = []

    for num_blocks in problem_sizes:
        for config in config_names:
            use_enhanced = (config == "enhanced")
            for run in range(runs_per_config):
                success, iters, final_response, actions = run_planning(model, num_blocks, max_iterations,
                                                                       enhanced=use_enhanced)
                results.append({
                    "num_blocks": num_blocks,
                    "config": config,
                    "run": run,
                    "success": success,
                    "iterations": iters
                })
                print(f"Problem size {num_blocks}, config {config}, run {run}: success={success}, iterations={iters}")

    # Save results to a CSV file.
    with open("experiment_results.csv", "w", newline="") as csvfile:
        fieldnames = ["num_blocks", "config", "run", "success", "iterations"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print("Results saved to experiment_results.csv")

if __name__ == "__main__":
    run_tests()
