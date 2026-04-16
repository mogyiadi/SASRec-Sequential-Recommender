import pandas as pd


def preprocess_data(ratings_file = 'ratings.dat'):
    """
    This function converts the explicit ratings to binary implicit ratings from the dataset.
    It also generates chronological interaction sequences for each user.
    It filters out users with fewer than 5 interactions.
    Finally, it applies a leave-one-out split for each user.

    The data has the following columns: user_id, sequence, target.

    The final preprocessed data is saved as train.json, test.json, and val.json.

    :param ratings_file: Path to the ratings.dat file
    """
    # Load dfs
    ratings_df = pd.read_csv(ratings_file, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

    # Create a new DataFrame to store the preprocessed ratings
    # Only keep positive ratings (>= 4)
    preprocessed_ratings_df = ratings_df[ratings_df['rating'] >= 4].copy()
    preprocessed_ratings_df['rating'] = 1

    # Sorting by timestamp per user
    preprocessed_ratings_df = preprocessed_ratings_df.sort_values(['user_id','timestamp'])

    # Drop users with less than 5 interactions
    preprocessed_ratings_df = preprocessed_ratings_df.groupby('user_id').filter(lambda x: len(x) >= 5)

    # Re-index movies
    unique_movies = sorted(preprocessed_ratings_df['movie_id'].unique())
    movie_mapping = {movie_id: idx for idx, movie_id in enumerate(unique_movies, start=1)}
    preprocessed_ratings_df['movie_id'] = preprocessed_ratings_df['movie_id'].map(movie_mapping)

    # Applying a leave-one-out split
    train_list = []
    val_list = []
    test_list = []

    for user_id, movie_ids in preprocessed_ratings_df.groupby('user_id')['movie_id']:
        movie_ids = list(movie_ids)
        for i in range(1, len(movie_ids) - 2):
            train_list.append({'user_id': user_id, 'input': movie_ids[:i], 'target': movie_ids[i]})
        val_list.append({'user_id': user_id, 'input': movie_ids[:-2], 'target': movie_ids[-2]})
        test_list.append({'user_id': user_id, 'input': movie_ids[:-1], 'target': movie_ids[-1]})

    # Convert lists to DataFrames
    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    test_df = pd.DataFrame(test_list)

    # # Save the preprocessed data
    train_df.to_json('train.json', orient='records')
    test_df.to_json('test.json', orient='records')
    val_df.to_json('val.json', orient='records')


if __name__ == "__main__":
    preprocess_data(ratings_file='data/ratings.dat')
