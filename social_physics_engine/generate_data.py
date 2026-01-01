import pandas as pd
import numpy as np
import time

class GoogleJules:
    def __init__(self, num_users=1000):
        self.num_users = num_users
        self.users = pd.DataFrame({
            'id': range(num_users),
            'x': np.random.rand(num_users),
            'y': np.random.rand(num_users),
            'influence': np.random.rand(num_users)
        })

    def calculate_interactions(self):
        """
        Calculates interaction scores between all pairs of users.
        Interaction = (Influence_A * Influence_B) / Distance_AB
        """
        start_time = time.time()

        # Vectorized optimization using NumPy broadcasting
        x = self.users['x'].values
        y = self.users['y'].values
        inf = self.users['influence'].values
        ids = self.users['id'].values

        # Compute pairwise squared Euclidean distances: (x_i - x_j)^2 + (y_i - y_j)^2
        # Use broadcasting: (N, 1) - (1, N) -> (N, N)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute pairwise influence products
        inf_prod = inf[:, np.newaxis] * inf[np.newaxis, :]

        # Avoid division by zero (distance to self is 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = inf_prod / dists

        # Set diagonal to 0 (self-interaction) or any NaNs resulting from 0/0
        np.fill_diagonal(scores, 0)

        # Flatten the arrays to create the result DataFrame
        # We need all pairs (i, j) where i != j
        # Create a mask for non-diagonal elements
        mask = ~np.eye(len(ids), dtype=bool)

        # Meshgrid for IDs
        id_matrix_a = np.repeat(ids[:, np.newaxis], len(ids), axis=1)
        id_matrix_b = np.repeat(ids[np.newaxis, :], len(ids), axis=0)

        interactions_df = pd.DataFrame({
            'user_a': id_matrix_a[mask],
            'user_b': id_matrix_b[mask],
            'score': scores[mask]
        })

        end_time = time.time()
        print(f"Calculated {len(interactions_df)} interactions in {end_time - start_time:.4f} seconds.")

        return interactions_df

if __name__ == "__main__":
    engine = GoogleJules(num_users=200) # Small number for testing
    df = engine.calculate_interactions()
    print(df.head())
