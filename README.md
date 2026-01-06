# Tetris_A

This project implements and trains Reinforcement Learning agents for Tetris, leveraging **Gymnasium Tetris** for the environment and **Stable Baselines3** for the PPO/DQN algorithm. 

---

## Environment Setup

The project uses **uv** for dependency management.

### 1. Using uv (Recommended)

```bash
uv sync
```
### 2. Using pip
Manually install dependencies as specified in `pyproject.toml`

## Running the Test
Run the test using:
```bash
uv run python main.py
```
The script will perform 10 quick random games using the pre trained model.

## Using Different Models
To test a different model:
1. Open ``main.py``
2. Change the file path inside ``PPO.load(...)``
```bash
PPO.load("G_models/PPO22_300M_8_5_2_1_3.zip")
```
   
## Model Repository

Model files are organized into two main directories:
- **G_models/**: Main models saved at various training steps
- **old_models/**: Legacy models (naming conventions may be inconsistent)
  
**Observation Size (obs_size)**

When using older models, ensure `obs_size` in `tetris_env.make_env()` matches the model requirements:
* **Check Filename**: If a number follows "PPO" (e.g., `PPO22_...`), `obs_size` is **22**.
* **Default**: If no number is present, `obs_size` is likely **16**.


