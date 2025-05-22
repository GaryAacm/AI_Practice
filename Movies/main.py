from src.data_loader import download_dataset, load_data, create_user_movie_matrix
from src.usercf import UserCF
from src.itemcf import ItemCF

# 输出电影名称
def print_movie_titles(movie_ids, movies):
    movie_titles = movies[movies['movieId'].isin(movie_ids)]['title']
    for title in movie_titles:
        print(title)

def main():
    # 下载并加载数据
    download_dataset()
    movies, ratings = load_data()
    user_movie_matrix = create_user_movie_matrix(ratings)

    # 基于用户的协作过滤推荐
    usercf_model = UserCF(user_movie_matrix)
    user_recommendations = usercf_model.recommend(user_id=1, top_n=10)
    print("基于用户协作过滤推荐的电影：")
    print_movie_titles(user_recommendations, movies)

    # 基于物品的协作过滤推荐
    itemcf_model = ItemCF(user_movie_matrix)
    item_recommendations = itemcf_model.recommend(movie_id=50, top_n=10)
    print("\n基于物品协作过滤推荐的电影：")
    print_movie_titles(item_recommendations, movies)

if __name__ == "__main__":
    main()
