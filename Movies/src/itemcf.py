from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 基于物品的协作过滤推荐算法
class ItemCF:
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix
        self.item_similarity = self.compute_item_similarity()

    def compute_item_similarity(self):
        item_similarity = cosine_similarity(self.user_movie_matrix.T)
        return pd.DataFrame(item_similarity, index=self.user_movie_matrix.columns, columns=self.user_movie_matrix.columns)

    def recommend(self, movie_id, top_n=10):
        similar_movies = self.item_similarity[movie_id].sort_values(ascending=False)[1:top_n+1].index
        return similar_movies
