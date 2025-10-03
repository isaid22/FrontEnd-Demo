import numpy as np
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class Arm:
    """Beta-Bernoulli Thompson arm."""
    __slots__ = ("alpha", "beta", "total_reward", "num_pulls")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.total_reward = 0.0
        self.num_pulls = 0

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: int):
        """reward ∈ {0,1}"""
        self.alpha += reward
        self.beta += 1 - reward
        self.total_reward += reward
        self.num_pulls += 1


class ThompsonBandit:
    BACKUP_DIR = "bandit_backup"
    BACKUP_FILE = "bandit_state.json"
    
    def __init__(self, arm_ids: List[str]):
        self.arms: Dict[str, Arm] = {aid: Arm() for aid in arm_ids}
        # Create backup directory if it doesn't exist
        if not os.path.exists(self.BACKUP_DIR):
            os.makedirs(self.BACKUP_DIR)
        # Try to load previous state
        self.load_state()

    def save_state(self):
        """Save current bandit state to file."""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "arms": {
                aid: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "total_reward": arm.total_reward,
                    "num_pulls": arm.num_pulls
                }
                for aid, arm in self.arms.items()
            }
        }
        
        backup_path = os.path.join(self.BACKUP_DIR, self.BACKUP_FILE)
        # Create a backup of previous state
        if os.path.exists(backup_path):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(backup_path, 
                     os.path.join(self.BACKUP_DIR, f"bandit_state_{backup_timestamp}.json"))
        
        # Save current state
        with open(backup_path, 'w') as f:
            json.dump(state_data, f, indent=4)

    def load_state(self) -> bool:
        """Load bandit state from file if it exists."""
        try:
            backup_path = os.path.join(self.BACKUP_DIR, self.BACKUP_FILE)
            if not os.path.exists(backup_path):
                print(f"\nNo bandit state file found at: {backup_path}")
                print("Starting with fresh bandit state:")
                print("\nInitial arm values:")
                for aid, arm in self.arms.items():
                    print(f"\n{aid}:")
                    print(f"  α (alpha): {arm.alpha:.2f}")
                    print(f"  β (beta): {arm.beta:.2f}")
                    print(f"  Total Reward: {arm.total_reward}")
                    print(f"  Number of Pulls: {arm.num_pulls}")
                return False
                
            with open(backup_path, 'r') as f:
                state_data = json.load(f)
            
            print(f"\nLoading bandit state from: {backup_path}")
            print(f"State timestamp: {state_data.get('timestamp', 'Not recorded')}")
            print("\nLoaded arm values:")
            
            loaded_arms = set()
            for aid, arm_data in state_data["arms"].items():
                if aid in self.arms:
                    loaded_arms.add(aid)
                    self.arms[aid].alpha = arm_data["alpha"]
                    self.arms[aid].beta = arm_data["beta"]
                    self.arms[aid].total_reward = arm_data["total_reward"]
                    self.arms[aid].num_pulls = arm_data["num_pulls"]
                    print(f"\n{aid}:")
                    print(f"  α (alpha): {arm_data['alpha']:.2f}")
                    print(f"  β (beta): {arm_data['beta']:.2f}")
                    print(f"  Total Reward: {arm_data['total_reward']}")
                    print(f"  Number of Pulls: {arm_data['num_pulls']}")
                    if arm_data['num_pulls'] > 0:
                        avg_reward = arm_data['total_reward'] / arm_data['num_pulls']
                        print(f"  Average Reward: {avg_reward:.3f}")
            
            # Print any new arms that weren't in the saved state
            new_arms = set(self.arms.keys()) - loaded_arms
            if new_arms:
                print("\nNew arms (not in saved state):")
                for aid in sorted(new_arms):
                    arm = self.arms[aid]
                    print(f"\n{aid}:")
                    print(f"  α (alpha): {arm.alpha:.2f}")
                    print(f"  β (beta): {arm.beta:.2f}")
                    print(f"  Total Reward: {arm.total_reward}")
                    print(f"  Number of Pulls: {arm.num_pulls}")
            
            print("\nBandit state loaded successfully!")
            return True
        except Exception as e:
            print(f"\nError loading bandit state: {e}")
            return False

    def choose(self) -> str:
        """Return arm with highest sampled probability."""
        return max(self.arms, key=lambda aid: self.arms[aid].sample())

    def reward(self, arm_id: str, reward: int):
        self.arms[arm_id].update(reward)
        # Save state after each reward update
        self.save_state()

    def state(self) -> Dict[str, dict]:
        return {aid: {
                "alpha": arm.alpha,
                "beta": arm.beta,
                "total_reward": arm.total_reward,
                "num_pulls": arm.num_pulls,
                "average_reward": arm.total_reward / arm.num_pulls if arm.num_pulls > 0 else 0.0
            } for aid, arm in self.arms.items()}