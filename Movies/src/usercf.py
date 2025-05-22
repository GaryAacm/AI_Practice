from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 基于用户的协作过滤推荐算法
class UserCF:
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix
        self.user_similarity = self.compute_user_similarity()

    def compute_user_similarity(self):
        user_similarity = cosine_similarity(self.user_movie_matrix)
        return pd.DataFrame(user_similarity, index=self.user_movie_matrix.index, columns=self.user_movie_matrix.index)

    def recommend(self, user_id, top_n=10):
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:top_n+1].index
        recommended_movies = []

        for user in similar_users:
            user_ratings = self.user_movie_matrix.loc[user]
            rated_movies = user_ratings[user_ratings > 0].index
            recommended_movies.extend(rated_movies)
        
        recommended_movies = list(set(recommended_movies))
        return recommended_movies[:top_n]
