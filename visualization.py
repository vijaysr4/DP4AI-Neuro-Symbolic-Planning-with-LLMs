import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the folder if it doesn't exist.
os.makedirs("visualization", exist_ok=True)

# Load the CSV file
df = pd.read_csv("experimental_results/gpt-4o_experiment_results.csv")

# Compute average iterations per configuration and problem size.
avg_iters = df.groupby(["num_blocks", "config"])["iterations"].mean().reset_index()

# Pivot the DataFrame for easier plotting.
pivot = avg_iters.pivot(index="num_blocks", columns="config", values="iterations")

# Plot the results
pivot.plot(kind="bar", figsize=(10,6))
plt.title("GPT-4o Average Iterations by Problem Size and Configuration")
plt.xlabel("Number of Blocks")
plt.ylabel("Average Iterations")
plt.xticks(rotation=0)
plt.legend(title="Configuration")
plt.tight_layout()


plt.savefig("visualization/gpt-4o_average_iterations.png")

# Optionally, display the plot.
plt.show()
