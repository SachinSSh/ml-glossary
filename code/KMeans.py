
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class AdvancedKMeans:
    """
    Advanced K-Means implementation with multiple initialization strategies,
    convergence diagnostics, and performance optimization.
    """
    
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, 
                 tol=1e-4, n_init=10, random_state=None, distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.distance_metric = distance_metric
        
        # Convergence tracking
        self.inertia_history_ = []
        self.n_iter_ = 0
        self.convergence_threshold_reached_ = False
        
    def _distance_matrix(self, X, centroids):
        """Compute distance matrix between points and centroids."""
        if self.distance_metric == 'euclidean':
            return cdist(X, centroids, metric='euclidean')
        elif self.distance_metric == 'manhattan':
            return cdist(X, centroids, metric='cityblock')
        elif self.distance_metric == 'cosine':
            return cdist(X, centroids, metric='cosine')
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _initialize_centroids(self, X):
        """Initialize centroids using specified strategy."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            return X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        elif self.init == 'k-means++':
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids with probability proportional to squared distance
            for i in range(1, self.n_clusters):
                distances = np.min(cdist(X, centroids[:i]), axis=1)
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                
                cumulative_prob = np.cumsum(probabilities)
                r = np.random.random()
                centroids[i] = X[np.searchsorted(cumulative_prob, r)]
            
            return centroids
        
        elif self.init == 'pca':
            # Initialize along principal components
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(self.n_clusters-1, X.shape[1]))
            X_pca = pca.fit_transform(X)
            
            # Distribute centroids along first principal component
            pc1_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), self.n_clusters)
            centroids_pca = np.zeros((self.n_clusters, X_pca.shape[1]))
            centroids_pca[:, 0] = pc1_range
            
            return pca.inverse_transform(centroids_pca)
    
    def _assign_clusters(self, X, centroids):
        """Assign points to nearest centroids."""
        distances = self._distance_matrix(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as cluster means."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                centroids[i] = X[mask].mean(axis=0)
            else:
                # Handle empty clusters by reinitializing
                centroids[i] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def _compute_inertia(self, X, centroids, labels):
        """Compute within-cluster sum of squares."""
        inertia = 0
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                cluster_points = X[mask]
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-Means clustering to data."""
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        # Multiple random initializations
        for init_run in range(self.n_init):
            centroids = self._initialize_centroids(X)
            inertia_history = []
            
            # Main K-Means iterations
            for iteration in range(self.max_iter):
                # Assignment step
                labels = self._assign_clusters(X, centroids)
                
                # Update step
                new_centroids = self._update_centroids(X, labels)
                
                # Compute inertia
                inertia = self._compute_inertia(X, new_centroids, labels)
                inertia_history.append(inertia)
                
                # Check convergence
                centroid_shift = np.linalg.norm(new_centroids - centroids)
                if centroid_shift < self.tol:
                    self.convergence_threshold_reached_ = True
                    break
                
                centroids = new_centroids
            
            # Keep best result across initializations
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                self.inertia_history_ = inertia_history
                self.n_iter_ = iteration + 1
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """Assign new points to existing clusters."""
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        """Fit clustering and return cluster labels."""
        return self.fit(X).labels_
