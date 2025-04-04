from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_RFmodel(x, y):
    try:
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
        print("✅ Dataset split into training and testing sets.")
    except ValueError as ve:
        print(f"❌ Error during train/test split: {ve}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error during dataset split: {e}")
        raise

    try:
        # Create and train the Random Forest model
        rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
        rfmodel = rf.fit(x_train, y_train)
        print("✅ Random Forest model trained successfully.")
    except Exception as e:
        print(f"❌ Error training Random Forest model: {e}")
        raise

    try:
        # Save the trained model to file
        with open('models/RFmodel.pkl', 'wb') as f:
            pickle.dump(rfmodel, f)
        print("✅ Trained model saved to 'models/RFmodel.pkl'.")
    except FileNotFoundError:
        print("❌ 'models' directory does not exist.")
        raise
    except Exception as e:
        print(f"❌ Error saving model to file: {e}")
        raise

    return rfmodel, x_train, x_test, y_test
