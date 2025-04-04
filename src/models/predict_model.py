# Import mean absolute error
from sklearn.metrics import mean_absolute_error

# Function to predict and evaluate
def evaluate_model(rfmodel, x_train, x_test, y_test):
    try:
        # Predict on the testing set
        y_pred = rfmodel.predict(x_test)

        # Predict on the training set
        ytrain_pred = rfmodel.predict(x_train)

        # Evaluate using MAE
        test_mae = mean_absolute_error(y_test, y_pred)

        print("✅ Model evaluated successfully.")
        return test_mae

    except ValueError as ve:
        print(f"❌ Value error during prediction or evaluation: {ve}")
        raise
    except AttributeError as ae:
        print(f"❌ Model may not be trained yet: {ae}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error during model evaluation: {e}")
        raise
