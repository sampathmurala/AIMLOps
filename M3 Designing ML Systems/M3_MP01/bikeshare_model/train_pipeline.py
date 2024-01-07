import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # print('read training data', data.shape)
    # print(data.head())
    
    # divide train and test
    # print('divide into train and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size, 
        random_state=config.model_config.random_state,
    )
    
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(X_train.head()) 
    
    # Pipeline fitting
    bikeshare_pipe.fit(X_train, y_train)
    
    # print(X_test.head(2), X_test.info())
    y_pred = bikeshare_pipe.predict(X_test)
    
    print(f"'predictions': {y_pred}")
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"mse: {mse}")

    # Calculate R-squared (R2) score
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared (R2) score: {r2}')

    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_pipe)
    # printing the score


if __name__ == "__main__":
    run_training()