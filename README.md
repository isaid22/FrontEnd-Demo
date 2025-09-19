# Thompson Sampling Demo

An interactive web-based demonstration of the Thompson Sampling algorithm for solving multi-armed bandit problems.

## 🎯 What is Thompson Sampling?

Thompson Sampling is a probabilistic approach to the exploration-exploitation dilemma in reinforcement learning. It's particularly effective for multi-armed bandit problems, where you need to balance exploring new options with exploiting known good options.

## 🚀 Features

- **Interactive Visualization**: Real-time demonstration of 5 different "arms" (slot machines) with varying success rates
- **Live Statistics**: Track total trials, rewards, success rate, and optimal selection rate
- **Dynamic Charts**: Cumulative reward visualization comparing actual performance vs theoretical maximum
- **Control Options**: Start/pause simulation, single-step through trials, or reset to start over
- **Educational Content**: Built-in explanation of how the algorithm works

## 🎮 How to Use

1. **Open the Demo**: Open `index.html` in your web browser
2. **Single Step**: Click "Single Step" to see individual decisions
3. **Run Simulation**: Click "Start Simulation" to watch the algorithm learn automatically
4. **Observe Learning**: Watch how the algorithm discovers that Arm 5 has the highest payout rate (80%)
5. **Reset**: Start over with "Reset" to see different learning paths

## 🧠 How It Works

The demo implements Thompson Sampling with these key components:

1. **Beta Distribution Modeling**: Each arm is modeled using a Beta distribution with α (successes + 1) and β (failures + 1) parameters
2. **Bayesian Sampling**: For each trial, the algorithm samples from each arm's Beta distribution
3. **Greedy Selection**: The arm with the highest sampled value is selected
4. **Posterior Update**: The selected arm's distribution is updated based on the observed reward

## 📊 Arm Configuration

- **Arm 1**: 10% success rate (lowest)
- **Arm 2**: 30% success rate  
- **Arm 3**: 60% success rate
- **Arm 4**: 20% success rate
- **Arm 5**: 80% success rate (optimal choice)

## 🎯 Expected Behavior

As the algorithm runs, you should observe:
- Initial random exploration of all arms
- Gradual convergence toward Arm 5 (the optimal choice)
- Optimal rate approaching 100% as trials increase
- Success rate approaching 80% (Arm 5's true rate)

## 🛠 Technical Implementation

- **Pure JavaScript**: No external dependencies
- **Canvas Visualization**: Custom chart rendering for performance tracking
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: 100ms interval for smooth animation

## 📈 Key Metrics

- **Total Trials**: Number of decisions made
- **Total Reward**: Cumulative successful outcomes
- **Success Rate**: Overall percentage of successful trials
- **Optimal Rate**: Percentage of times the best arm was selected

Perfect for understanding how Thompson Sampling balances exploration and exploitation in reinforcement learning!