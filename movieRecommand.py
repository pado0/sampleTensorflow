# https://velog.io/@skarb4788/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%98%81%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import ast
import warnings; warnings.filterwarnings('ignore')

movies = pd.read_csv('tmdb_5000_movies.csv')

# 콘솔에 표시할 로우 컬럼 수
pd.set_option('display.max_columns', 2)
pd.set_option('display.max_rows', 5)

# 사용할 컬럼만 가져오기
movies_df = movies[['id', 'title', 'genres', 'keywords']]

# literal_eval로 string to Dict 형변환
movies_df['genres'] = movies_df['genres'].apply(ast.literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(ast.literal_eval)
print(movies_df.head(2))

# dict형 value 값을 특성으로 사용하도록 변경한다. movies_df에 String<List> 형태만 남기고 id는 지운다.
movies_df['genres'] = movies_df['genres'].apply(lambda x : [y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [y['name'] for y in x])
print(movies_df.head(2))

# genre의 단어를 문장으로 변환
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
movies_df.head(2)

# 문장 변환한 장르 정보를 기준으로 코사인 유사도 측정을 위해 CounterVectorize
## 벡터 변환 이해
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)

# 코사인 유사도 결과
## 실제 계산하여 id가 어떻게 보존되는지 확인
genre_sim = cosine_similarity(genre_mat, genre_mat)

# 유사도 높은 것 기준 내림차순 정렬하여
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:-1])


# sort된 id랑 영화이름 맵핑
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]


similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
print(similar_movies)