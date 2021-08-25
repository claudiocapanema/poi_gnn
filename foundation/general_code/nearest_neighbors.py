import sklearn.neighbors as nb

class NearestNeighbors:

    @classmethod
    def find_radius_neighbors(self, gt_points, dp_points, radius):
        neigh = nb.NearestNeighbors(
            radius=radius,
            algorithm='ball_tree',
            metric='haversine',
            n_jobs=-1)
        neigh = neigh.fit(dp_points)
        rng = neigh.radius_neighbors(gt_points)
        distances = rng[0]
        indexes = rng[1]
        return distances, indexes