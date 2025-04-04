from src.data.data_processing import load_and_clean_data
from src.features.features_eng import create_new_variables
from src.features.features_eng import Create_dummy_variables
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_Boxplot, plot_Correlation_Heatmap

if __name__ == "__main__":
    try:
        # Load and preprocess the data
        data_path = "data/raw/real_estate.csv"
        df = load_and_clean_data(data_path)
        print("✅ Data loaded and cleaned successfully.")
    except FileNotFoundError:
        print(f"❌ File not found at: {data_path}")
        exit()
    except Exception as e:
        print(f"❌ Error loading or cleaning data: {e}")
        exit()

    try:
        df = create_new_variables(df)
        print("✅ Feature engineering (new variables) completed.")
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        exit()

    try:
        x, y = Create_dummy_variables(df)
        print("✅ Dummy variables created successfully.")
    except Exception as e:
        print(f"❌ Error creating dummy variables: {e}")
        exit()

    try:
        plot_Correlation_Heatmap(df)
        plot_Boxplot(df)
        print("✅ Visualizations created.")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

    try:
        # Train the Random Forest model
        model, x_train, x_test, y_test = train_RFmodel(x, y)
        print("✅ Model training complete.")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        exit()

    try:
        MAE = evaluate_model(model, x_train, x_test, y_test)
        print(f"✅ Model evaluated. Mean Absolute Error: {MAE}")
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
