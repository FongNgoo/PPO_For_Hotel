# envs/pricing_env.py
import pandas as pd
import numpy as np
import random


class PricingEnv:
    def __init__(
        self,
        dataset,
        demand_model,
        adr_ref=95.0,
        lambda_reg=0.1,
        preprocessor=None,          # ← BẮT BUỘC thêm
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.dataset = dataset
        self.demand_model = demand_model
        self.adr_ref = adr_ref
        self.lambda_reg = lambda_reg
        self.preprocessor = preprocessor   # Lưu lại

        if self.preprocessor is None:
            raise ValueError("PricingEnv requires preprocessor for state encoding")

        self.dates = list(dataset.keys())
        self.current_date = None
        self.booking_ids = None
        self.booking_ptr = 0

        # Infer state_dim an toàn
        sample_date = self.dates[0]
        sample_booking = next(iter(dataset[sample_date]["bookings"].values()))
        sample_background = dataset[sample_date]["background"]
        
        sample_dict = self._build_state_dict(sample_booking, sample_background)
        sample_df = pd.DataFrame([sample_dict])
        sample_vec = self.preprocessor.transform(sample_df)[0]
        self.state_dim = len(sample_vec)

    def reset(self):
        self.current_date = random.choice(self.dates)
        self.booking_ids = list(self.dataset[self.current_date]["bookings"].keys())
        random.shuffle(self.booking_ids)
        self.booking_ptr = 0
        return self._get_state()

    def step(self, alpha):
        booking = self._current_booking()
        state_vec = self._get_state()

        price = float(alpha) * self.adr_ref
        p_accept = self.demand_model.predict_proba(state_vec, price)

        num_nights = booking["booking_context"]["stays"]["total_nights"]
        reward = price * num_nights * p_accept - self.lambda_reg * (price - self.adr_ref) ** 2

        self.booking_ptr += 1
        done = self.booking_ptr >= len(self.booking_ids)

        info = {
            "date": self.current_date,
            "price": price,
            "num_nights": num_nights,
            "p_accept": p_accept,
            "expected_revenue": price * num_nights * p_accept,
        }

        next_state = None if done else self._get_state()
        return next_state, reward, done, info

    def _current_booking(self):
        return self.dataset[self.current_date]["bookings"][self.booking_ids[self.booking_ptr]]

    def _get_state(self):
        booking = self._current_booking()
        background = self.dataset[self.current_date]["background"]
        return self._encode_state(booking=booking, background=background)

    def _encode_state(self, *, booking, background):
        state_dict = self._build_state_dict(booking, background)
        state_df = pd.DataFrame([state_dict])
        transformed = self.preprocessor.transform(state_df)
        return transformed[0].astype(np.float32)

    def _build_state_dict(self, booking, background):
        # Lấy đầy đủ các cột numerical + categorical giống lúc fit
        stays = booking["booking_context"]["stays"]
        return {
            # Numerical
            "lead_time": booking["booking_context"].get("lead_time", 0),
            "stays_in_weekend_nights": stays.get("weekend_nights", 0),
            "stays_in_week_nights": stays.get("week_nights", 0),
            "adults": booking["customer_profile"].get("adults", 0),
            "children": booking["customer_profile"].get("children", 0),
            "babies": booking["customer_profile"].get("babies", 0),
            "previous_cancellations": booking["booking_context"].get("previous_cancellations", 0),
            "trend_mean": background.get("trend_mean", 0.0),
            "trend_max": background.get("trend_max", 0.0),  # Nếu có trong numerical

            # Categorical – BẮT BUỘC phải có, nếu không preprocessor sẽ lỗi
            "market_segment": booking["channel_info"].get("market_segment", "Unknown"),
            "distribution_channel": booking["channel_info"].get("distribution_channel", "Unknown"),
            "arrival_date_month": pd.to_datetime(booking["booking_context"]["arrival_date"]).strftime("%B"),
            "is_weekend": str(background.get("is_weekend", False)).lower(),       # OneHotEncoder cần string hoặc bool nhất quán
            "is_holiday_pt": str(background.get("is_holiday_pt", False)).lower(),
            "is_before_holiday": str(background.get("is_before_holiday", False)).lower(),
            "is_after_holiday": str(background.get("is_after_holiday", False)).lower(),
        }