# envs/pricing_env.py

import numpy as np
import random


class PricingEnv:
    """
    Dynamic Personalized Pricing Environment
    (Contextual MDP)

    - State: booking + customer + market background
    - Action: price multiplier (alpha)
    - Reward: expected total revenue per booking
    """

    def __init__(
        self,
        dataset,
        demand_model,
        adr_ref=95.0,
        lambda_reg=0.1,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.dataset = dataset
        self.demand_model = demand_model
        self.adr_ref = adr_ref
        self.lambda_reg = lambda_reg

        # One episode = one booking date
        self.dates = list(dataset.keys())
        self.current_date = None
        self.booking_ids = None
        self.booking_ptr = 0

        # --------------------------------------------------
        # Infer state dimension robustly
        # --------------------------------------------------
        sample_date = self.dates[0]
        sample_booking = next(
            iter(dataset[sample_date]["bookings"].values())
        )
        sample_background = dataset[sample_date]["background"]

        sample_state = self._encode_state(
            booking=sample_booking,
            background=sample_background
        )
        self.state_dim = len(sample_state)

    # ==================================================
    # RESET: new episode (new date)
    # ==================================================
    def reset(self):
        self.current_date = random.choice(self.dates)
        self.booking_ids = list(
            self.dataset[self.current_date]["bookings"].keys()
        )
        random.shuffle(self.booking_ids)
        self.booking_ptr = 0
        return self._get_state()

    # ==================================================
    # STEP
    # ==================================================
    def step(self, alpha):
        booking = self._current_booking()
        state_vec = self._get_state()

        # Price decision
        price = float(alpha) * self.adr_ref

        # Demand response
        p_accept = self.demand_model.predict_proba(
            state_vec, price
        )

        # Number of nights
        num_nights = booking["booking_context"]["stays"]["total_nights"]

        # Reward = expected total revenue
        reward = (
            price * num_nights * p_accept
            - self.lambda_reg * (price - self.adr_ref) ** 2
        )

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

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================
    def _current_booking(self):
        booking_id = self.booking_ids[self.booking_ptr]
        return self.dataset[self.current_date]["bookings"][booking_id]

    def _get_state(self):
        booking = self._current_booking()
        background = self.dataset[self.current_date]["background"]
        return self._encode_state(
            booking=booking,
            background=background
        )

    def _encode_state(self, *, booking, background):
        """
        Encode booking + background into numeric state vector
        Missing features are safely defaulted to 0
        """

        customer = booking["customer_profile"]
        context = booking["booking_context"]
        stays = context["stays"]

        state = [
            # ----------------------
            # Booking / customer
            # ----------------------
            context.get("lead_time", 0),
            customer.get("adults", 0),
            customer.get("children", 0),
            customer.get("babies", 0),
            stays.get("total_nights", 1),
            context.get("previous_cancellations", 0),

            # ----------------------
            # Market / calendar
            # ----------------------
            float(background.get("is_weekend", 0)),
            background.get("trend_mean", 0.0),
            float(background.get("is_holiday_pt", 0)),
            float(background.get("is_before_holiday", 0)),
            float(background.get("is_after_holiday", 0)),
        ]

        return np.array(state, dtype=np.float32)

