"""
Similarity Searcher for finding similar customers in tabular data.

Uses Annoy (Approximate Nearest Neighbors) for fast similarity search
to retrieve relevant examples for in-context learning.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import polars as pl
from annoy import AnnoyIndex


class SimilaritySearcher:
    """
    Similarity searcher for tabular customer data using Annoy.
    
    This class builds an index on numerical features and enables fast
    retrieval of similar customers for in-context learning.
    """
    
    def __init__(
        self,
        reference_df: pl.DataFrame,
        similarity_features: List[str],
        metric: str = 'euclidean',
        n_trees: int = 10
    ):
        """
        Initialize the SimilaritySearcher.
        
        Args:
            reference_df (pl.DataFrame): Polars DataFrame containing reference customers
            similarity_features (List[str]): List of numerical feature column names to use for similarity
            metric (str): Distance metric ('euclidean', 'angular', 'manhattan', 'hamming', 'dot')
            n_trees (int): Number of trees for Annoy index (more trees = better accuracy, slower build)
        """
        self.reference_df = reference_df
        self.similarity_features = similarity_features
        self.metric = metric
        self.n_trees = n_trees
        
        # Validate features exist in dataset
        self._validate_features()
        
        # Extract and normalize feature vectors
        self.feature_vectors, self.feature_mean, self.feature_std = self._extract_and_normalize_features()
        
        # Build Annoy index
        self.index = self._build_index()
    
    def _validate_features(self):
        """Validate that all similarity features exist in the DataFrame."""
        df_columns = self.reference_df.columns
        missing_features = [f for f in self.similarity_features if f not in df_columns]
        
        if missing_features:
            raise ValueError(
                f"Similarity features not found in reference DataFrame: {missing_features}\n"
                f"Available columns: {df_columns}"
            )
    
    def _extract_and_normalize_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature vectors from DataFrame and normalize them.
        
        Returns:
            Tuple of (feature_vectors, mean, std) for normalization
        """
        # Extract features using Polars - select columns and convert to numpy
        # Fill null values with 0.0 before conversion
        feature_vectors = (
            self.reference_df
            .select(self.similarity_features)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )
        
        # Compute normalization statistics
        feature_mean = np.mean(feature_vectors, axis=0)
        feature_std = np.std(feature_vectors, axis=0)
        
        # Avoid division by zero
        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        
        # Normalize
        normalized_vectors = (feature_vectors - feature_mean) / feature_std
        
        return normalized_vectors, feature_mean, feature_std
    
    def _build_index(self) -> AnnoyIndex:
        """Build Annoy index from normalized feature vectors."""
        n_features = len(self.similarity_features)
        index = AnnoyIndex(n_features, self.metric)
        
        # Add all vectors to index
        for i, vector in enumerate(self.feature_vectors):
            index.add_item(i, vector)
        
        # Build index
        index.build(self.n_trees)
        
        return index
    
    def _normalize_query_vector(self, query_vector: np.ndarray) -> np.ndarray:
        """Normalize a query vector using the reference dataset statistics."""
        return (query_vector - self.feature_mean) / self.feature_std
    
    def search(
        self,
        query_item: Dict[str, Any],
        top_k: int = 5,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar customers to the query item.
        
        Args:
            query_item (Dict[str, Any]): Customer data to find similar examples for
            top_k (int): Number of similar examples to return
            include_distances (bool): Whether to include similarity distances in results
        
        Returns:
            List[Dict[str, Any]]: List of similar customer records with optional distances
        """
        # Extract and normalize query vector
        query_vector = []
        for feature in self.similarity_features:
            value = query_item.get(feature)
            if value is None:
                query_vector.append(0.0)
            else:
                query_vector.append(float(value))
        
        query_vector = np.array(query_vector, dtype=np.float32)
        normalized_query = self._normalize_query_vector(query_vector)
        
        # Search index
        indices, distances = self.index.get_nns_by_vector(
            normalized_query,
            top_k,
            include_distances=True
        )
        
        # Retrieve full records from DataFrame
        similar_customers = []
        for idx, distance in zip(indices, distances):
            # Get row from DataFrame and convert to dict
            customer = self.reference_df[int(idx)].to_dicts()[0]
            
            if include_distances:
                # Convert distance to similarity score (0-1, higher is more similar)
                # For euclidean distance, use exponential decay
                similarity_score = np.exp(-distance)
                customer['_similarity_score'] = float(similarity_score)
                customer['_similarity_distance'] = float(distance)
            
            similar_customers.append(customer)
        
        return similar_customers
    
    def batch_search(
        self,
        query_items: List[Dict[str, Any]],
        top_k: int = 5,
        include_distances: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar customers for a batch of query items.
        
        Args:
            query_items (List[Dict[str, Any]]): List of customer data to find similar examples for
            top_k (int): Number of similar examples to return per query
            include_distances (bool): Whether to include similarity distances in results
        
        Returns:
            List[List[Dict[str, Any]]]: List of similar customer lists for each query
        """
        return [
            self.search(query_item, top_k, include_distances)
            for query_item in query_items
        ]
