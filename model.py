import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class SentimentRecommentation:

    def __init__(self):
        self.model = pickle.load(open('pickle_files/logistic_regression.pkl', 'rb'))
        self.features_data = pickle.load(open('pickle_files/features.pkl', 'rb'))
        self.user_rating_data = pickle.load(open('pickle_files/user_final_rating.pkl', 'rb'))
        self.dataset = pickle.load(open('pickle_files/clean_data.pkl', 'rb'))
        self.input_data = pd.read_csv('data/sample30.csv')
        self.dataset = pd.concat([self.input_data[['id','name','brand','categories','manufacturer']], self.dataset], axis=1)


    def recommenderSystem(self, user):
        items_as_per_final_ratings = self.user_rating_data.loc[user].sort_values(ascending=False)[0:20].index
        vectorizer = TfidfVectorizer(vocabulary = self.features_data)
        result = self.dataset[self.dataset.id.isin(items_as_per_final_ratings)]
        fit_data = vectorizer.fit_transform(result['combined_reviews'])
        result = result[['id']]
        result['predictions'] = self.model.predict(fit_data)
        result['predictions'] = result['predictions'].map({'Positive':1, 'Negative':0})
        result = result.groupby('id').sum()
        result['positive_percent']=result.apply(lambda x: x['predictions']/sum(x), axis=1)
        final_list=result.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.dataset[self.dataset.id.isin(final_list)][['id', 'brand', 'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)