from dependencies import *

def split_data(df):
    try:
        target_0 = df[df['target'] == 0]
        target_4 = df[df['target'] == 4]
        
        train_0, test_0 = train_test_split(target_0, test_size=0.2, random_state=42)   # Split target 0 into training (80%) and testing (20%)
        train_4, test_4 = train_test_split(target_4, test_size=0.2, random_state=42)

        df_train = pd.concat([train_0, train_4])   # Combine the training data from both target groups
        df_test = pd.concat([test_0, test_4])

        X_train = df_train['cleaned_text']
        Y_train = df_train['target']

        x_test = df_test['cleaned_text']
        y_test = df_test['target']

        return X_train, Y_train, x_test, y_test

    except Exception as e:
        print(f"Error occured in SPLIT_DATA.py: {e}")

