import json
import pickle
import pybullet as p
from pathlib import Path

from core.orchestrate import run_individual
from evolution.helpers import deserialize_network
from simulation.simulation import Simulation


if __name__ == "__main__":
    # SETTINGS
    RUN_NAME: str = "run_2025-12-01_18-45-28"
    GENERATION: int = 99
    ###
    
    
    params_path = Path(__file__).parent / "results" / RUN_NAME / "params.pkl"
    indiv_path = Path(__file__).parent / "results" / RUN_NAME / f"generation-{GENERATION}/best_individual.json"
    
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    indiv = deserialize_network(indiv_path)
    
    sim = Simulation(p.GUI, params.creature_path, params.settle_steps, params.time_step)
    
    run_individual(indiv, sim, params)