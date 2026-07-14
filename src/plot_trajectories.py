import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot recorded evaluation trajectories alongside obstacle sweep zones.")
    parser.add_argument("--json-path", type=str, default="models/trajectories.json", help="Path to recorded trajectories JSON file")
    parser.add_argument("--output-plot", type=str, default="assets/evaluation_trajectories.png", help="Path to output plot PNG file")
    return parser.parse_args()

def main(args):
    if not os.path.exists(args.json_path):
        print(f"Error: Trajectory file not found at '{args.json_path}'. Run record_trajectories first:")
        print("python3 src/record_trajectories.py")
        return

    with open(args.json_path, "r") as f:
        trajectories = json.load(f)

    plt.figure(figsize=(9, 9))
    
    # Define vertical obstacle patrolling zones (from env.py configurations)
    obstacle_cols = [
        {"x": 150, "r": 35, "label": "Obs 1 (x=150, r=35)"},
        {"x": 200, "r": 40, "label": "Obs 2 (x=200, r=40)"},
        {"x": 300, "r": 45, "label": "Obs 3 (x=300, r=45)"},
        {"x": 400, "r": 50, "label": "Obs 4 (x=400, r=50)"}
    ]
    
    for i, obs in enumerate(obstacle_cols):
        # Swept x-range: [x - r, x + r]
        x_min = obs["x"] - obs["r"]
        x_max = obs["x"] + obs["r"]
        plt.axvspan(
            x_min, x_max, color="red", alpha=0.08, 
            label="Obstacle Patrol Sweep" if i == 0 else ""
        )

    # Plot each recorded trajectory
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, traj in enumerate(trajectories):
        coords = np.array(traj["path_coords"])
        start = traj["start_pos"]
        target = traj["target_pos"]
        success = traj["success"]
        
        # Route path line
        plt.plot(
            coords[:, 0], coords[:, 1], color=colors[i % len(colors)], linewidth=2.5, 
            label=f"Episode {traj['episode']} ({'Success' if success else 'Failed'})"
        )
        
        # Start node (Blue Circle)
        plt.scatter(
            start[0], start[1], color="#2980b9", marker="o", s=90, edgecolors='black', zorder=5,
            label="Start Position" if i == 0 else ""
        )
        
        # Target node (Green Star)
        plt.scatter(
            target[0], target[1], color="#27ae60", marker="*", s=180, edgecolors='black', zorder=5,
            label="Target Goal" if i == 0 else ""
        )

    # Styling the map axes
    plt.xlim(0, 600)
    plt.ylim(0, 600)
    plt.gca().invert_yaxis()  # Invert y-axis to match Pygame coordinates layout
    plt.xlabel("X Position (pixels)", fontsize=11)
    plt.ylabel("Y Position (pixels) [Inverted]", fontsize=11)
    plt.title("Safe RL Agent Navigation: Evaluation Trajectories", fontsize=13, fontweight='bold', pad=15)
    
    plt.legend(loc="upper right", bbox_to_anchor=(1.38, 1.0), frameon=True, borderpad=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save the output image
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.savefig(args.output_plot, bbox_inches="tight", dpi=150)
    print(f"Trajectory visualization graph successfully generated at: {args.output_plot}")
    plt.close()

if __name__ == "__main__":
    main(parse_args())
