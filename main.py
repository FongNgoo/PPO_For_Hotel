# main.py

from models.logistic_regression import LogisticDemandModel
from data.load_data import (
    load_hotel_data,
    load_trends_data,
    data_structure
)
from envs.pricing_env import PricingEnv
from models.actor_critic import ActorCritic
from algorithms.ppo import PPO
from trainers.trainer import Trainer

# ======================================================
# PATHS
# ======================================================
HOTEL_DATA_PATH = r"D:\Project\PPO_For_Hotel\data\resort_hotel_data.csv"
TREND_DATA_PATH = r"D:\Project\PPO_For_Hotel\data\google_trends_portugal_algarve_daily.csv"

ADR_REF = 95.0
SEED = 42


X, hotel_df, preprocessor = load_hotel_data(HOTEL_DATA_PATH, TREND_DATA_PATH)
trend_df = load_trends_data(TREND_DATA_PATH)  # vẫn giữ nếu data_structure cần

dataset = data_structure(hotel_df, trend_df)

demand_model = LogisticDemandModel(
    context_dim=X.shape[1],  # đúng dims sau preprocess
    adr_ref=ADR_REF,
    seed=SEED
)
demand_model.fit(X=X, y=(1 - hotel_df["is_canceled"].values))

env = PricingEnv(
    dataset=dataset,
    demand_model=demand_model,
    adr_ref=ADR_REF,
    lambda_reg=0.1,
    preprocessor=preprocessor,   # ← Truyền vào đây
    seed=SEED
)

state_dim = env.state_dim  # Bây giờ là dims đúng (sau one-hot)

model = ActorCritic(state_dim=state_dim, hidden_dim=128)

ppo = PPO(
    model=model,
    clip_eps=0.2,
    lr=3e-4,
    gamma=0.99
)

trainer = Trainer(
    env=env,
    model=model,
    ppo=ppo,
    steps_per_iter=256
)

# ======================================================
# 6. TRAIN
# ======================================================
trainer.train(iterations=500)
