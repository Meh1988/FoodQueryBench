A large-scale instruction dataset designed for LLM-based food recommendation, user preference modeling, and health-aware personalization.


📂 Dataset Download:

https://drive.google.com/drive/folders/1-mCG644I6-hBsxr1cllNr49tfxtZtOYa


📌 Overview

469,336 instruction–response samples

Derived from 507K recipes + 1.4M real user reviews

Covers 10 diverse task categories

Includes temporal behavior, flavor memory, and health-aware trade-offs

Fully LLM-ready JSON format


🧠 Task Categories

#	Task Type	Focus

1–3	Preference & Ranking	Personalized taste modeling

4–6	Sequential & Temporal	Session behavior + preference drift

7	Diversity Recommendation	Novelty & variety

8	Flavor Memory	Long-term user taste

9–10	Health-Aware Recommendation	Nutrition & indulgence trade-offs


📂 File Structure

dataset/

├── task_1_preference_simulation.json

├── task_2_pairwise_preference.json

├── ...

├── task_10_health_tradeoff.json

└── instruction_dataset.json   # combined dataset



Each entry is formatted for instruction tuning:

{

  "task_type": "sequential_session",
  
  "instruction": "Recommend the next recipe...",
  
  "response": "...",
  
  "user_id": 1533,
  
  "context_recipe_ids": [...],
  
  "target_recipe_id": 10721
  
}


🛠 Requirements

python >= 3.8

pip install pandas numpy matplotlib seaborn



▶️ Usage Example

import json

with open("instruction_dataset.json") as f:

    data = json.load(f)

print(len(data), data[0])


🎯 Applications

LLM fine-tuning for food & nutrition reasoning

Personalized recipe recommendation

Session-based and temporal preference learning

Health-aware and sustainable recommendation systems

