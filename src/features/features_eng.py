import pandas as pd

def create_new_variables(df):
    try:
        # Create indicator variable for properties with 2 beds and 2 baths
        df['popular'] = ((df.beds == 2) & (df.baths == 2)).astype(int)

        # Create a new variable recession
        df['recession'] = ((df.year_sold >= 2010) & (df.year_sold <= 2013)).astype(int)

        # Create a property age feature
        df['property_age'] = df.year_sold - df.year_built

        # Remove rows where property_age is less than 0
        df = df[df.property_age >= 0]

        print("✅ Feature engineering completed.")
        return df

    except KeyError as ke:
        print(f"❌ Missing expected column in DataFrame: {ke}")
        raise
    except Exception as e:
        print(f"❌ Error in create_new_variables: {e}")
        raise


def Create_dummy_variables(df):
    try:
        if 'property_type' not in df.columns:
            raise KeyError("'property_type' column not found in the DataFrame.")

        if 'price' not in df.columns:
            raise KeyError("'price' column not found in the DataFrame.")

        # Create dummy variables for 'property_type'
        df = pd.get_dummies(df, columns=['property_type']).astype(int)

        # Separate input features in x
        x = df.drop('price', axis=1)

        # Store the target variable in y
        y = df['price']

        print("✅ Dummy variables created and target separated.")
        return x, y

    except KeyError as ke:
        print(f"❌ Column error: {ke}")
        raise
    except Exception as e:
        print(f"❌ Error in Create_dummy_variables: {e}")
        raise
