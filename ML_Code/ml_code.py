from dependencies import *


## Import Dataset ##
file_path = 'C:/Users/work/Sensation/reduced_tweet_dataset.csv'

#Target: sentiemnt (0=negative, 4=positive)   Text: actual tweet
df = pd.read_csv(file_path, encoding='latin-1')

## Create a new cleaned text column ##
df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

## Process Text
X_train, Y_train, x_test, y_test = split_data(df)
X_train_vect, x_test_vect = vectorizer_text(X_train, x_test, vectorizer)
joblib.dump(vectorizer,'vectorizer.pkl')


## SVM model ##
svm_model = SVC(kernel = 'linear', cache_size=2000, verbose = True)
print("training started")
start_time = time.time()
svm_model = svm_model.fit(X_train_vect, Y_train)
print(f"Training finished in {time.time() - start_time} seconds")
y_pred = svm_model.predict(x_test_vect)

print(classification_report(y_test, y_pred))

# Save the model
model_file_path = "svm_model.pkl"
joblib.dump(svm_model, model_file_path)
print(f"Model saved to {model_file_path}")
