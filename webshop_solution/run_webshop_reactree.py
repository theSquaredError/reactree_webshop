import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
import json
import os
from datetime import datetime
from webshop_solution.reactree_planner import ReactreeWebshopPlanner

def main():
    planner = ReactreeWebshopPlanner(num_products=100, seed=0)
    instruction = None  # Use None to sample a random goal from the environment
    report = planner.plan_and_run(instruction_text=instruction, top_k=1, verbose=True)
    trajectory = report.get("trajectory", [])
    
    log_dir = "webshop_trajectories"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        json.dump({"report": report, "trajectory": trajectory}, f, indent=2)
    print(f"\nTrajectory saved to {filepath}")
    print(f"Success: {report.get('success')} | Terminate: {report.get('terminate')}")

if __name__ == "__main__":
    main()