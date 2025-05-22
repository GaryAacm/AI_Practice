import os
import pandas as pd
import requests
import zipfile


# 下载MovieLens数据集
def download_dataset(url="http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", dest_folder="../data"):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    zip_path = os.path.join(dest_folder, "ml-latest-small.zip")
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("数据集已下载")
    
    # 解压数据集
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print("数据集已解压")

# 加载MovieLens数据集
def load_data(data_folder="../data/ml-latest-small"):
    movies = pd.read_csv(os.path.join(data_folder, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_folder, "ratings.csv"))
    return movies, ratings

# 创建用户-电影评分矩阵
def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    return user_movie_matrix.fillna(0)
