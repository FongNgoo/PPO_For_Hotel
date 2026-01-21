import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticDemandModel:
    """
    Logistic regression demand model:
    P(accept | context, price)

    This model is trained offline and used as
    an environment response function.
    """

    def __init__(
        self,
        context_dim,
        adr_ref=95.0,
        seed=None
    ):
        self.context_dim = context_dim
        self.adr_ref = adr_ref
        self.seed = seed

        # Logistic Regression from sklearn
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed
        )

        self.fitted = False

    # ==================================================
    # TRAIN DEMAND MODEL (OFFLINE)
    # ==================================================
    def fit(self, X, y):
        """
        Fit logistic regression:
        y = 1 if accept, 0 otherwise
        """
        self.model.fit(X, y)
        self.fitted = True

    # ==================================================
    # UTILITY FUNCTION
    # u = beta^T x - price / adr_ref
    # ==================================================
    def utility(self, context, price):
        if not self.fitted:
            raise RuntimeError("Demand model must be fitted before use.")

        linear_term = np.dot(self.model.coef_[0], context)
        intercept = self.model.intercept_[0]

        return intercept + linear_term - price / self.adr_ref

    # ==================================================
    # P(accept | context, price)
    # ==================================================
    def predict_proba(self, context, price):
        u = self.utility(context, price)
        return 1.0 / (1.0 + np.exp(-u))

    # ==================================================
    # SAMPLE CUSTOMER DECISION
    # ==================================================
    def sample_booking(self, context, price):
        """
        Bernoulli sampling for environment transition
        """
        p = self.predict_proba(context, price)
        return np.random.rand() < p
