# CMSI 4320 - Final Project

🤖 Reinforcement learning agents to play Pacman capture the flag

🏆 Winner of 2024 CMSI 4320 Final Tournament

## Game Objective

The game is a 2v2 capture the flag variant of Pacman, where each team is in control of 2 agents and must return food from the opposing teams side and defend their own food from being taken. When on your teams side, your agent is a ghost and can eat Pacman. If your agent crosses to the other team's side, it becomes a Pacman, and can be eaten by the other teams ghosts. The game ends when 18 pellets are returned or when the specified number of moves are used up.

<img width="750" alt="pactf" src="https://github.com/loosh/pacman-ctf-agents/assets/56782878/4ac69071-74f3-4f4f-b996-f6ba93838789">

## Tournament Rules

- Each team must create an agent that uses some sort of reinforcement learning to update the weights / policy of their agents.
- The weights must begin as either 0 or small random values and no hand-tweaking of the weights is allowed.

## Our Strategy

We split the roles between our 2 agents, with one defensive and one offensive agent

- Offensive Agent
  - Chases pellets that are not within 10 moves of a ghost (this radius shrinks as more pellets are eaten)
  - Negatively rewarded for taking moves that lead closer to an enemy ghost
  - Positively rewarded for returning home when carrying pellets

- Defensive Agent
  - Waits near the center pellet for invading Pacman to cross onto its side
  - Positively rewarded for moving towards invaders, and away if the invader has the power pellet
  - If the center pellet is eaten, it camps one of the 3 vertical exit points that the invader could leave through based on the invaders Y coordinate position


## Technical Approach

- Reinforcement Learning
  - Utilized Q-learning to update agent's policies
- State Approximation
  - Featurized the gamestate using mainly binary features (e.g. 1 for moving towards food)
  - Featurized eating each pellet so the agent could learn which pellets are safe / risky to eat
- Noisy Distance Decoding
  - If an enemy is more than 5 tiles away, you will receive a noisy distance reading
  - Implemented a particle filter to decode this noisy reading, which allowed our defensive agent to thrive

## Results

Our agents were extremely effective against other agents and ended up winning the tournament. For fun we also faced off against the previous champions from the last 2 years and our agents were also able to beat them (granted we only played 1 game against each).
