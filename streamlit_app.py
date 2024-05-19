%%writefile app.py
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import requests
! pip install streamlit -q
!wget -q -O - ipv4.icanhazip.com

books=pd.read_csv('Books.csv')
ratings=pd.read_csv('Ratings.csv')
users=pd.read_csv('Users.csv')

ratings_books = ratings.merge(books,on='ISBN')
def clean_booktitle(title):
    return str(title).title().strip()
ratings_books['Book-Title'] = ratings_books['Book-Title'].apply(clean_booktitle)
ratings_books = ratings_books[ratings_books['User-ID'].map(ratings_books['User-ID'].value_counts()) > 50]
ratings_books = ratings_books[ratings_books['Book-Title'].map(ratings_books['Book-Title'].value_counts()) > 50]
ratings_books = ratings_books.reset_index(drop=True)

data_matrix = ratings_books.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
data_matrix.fillna(0,inplace=True)
similarity_scores_books = cosine_similarity(data_matrix)

def recommend_books(ratings_books,book_name):

    if ratings_books['Book-Title'].str.contains(book_name).any()==False:
        return -1

    index = np.where(data_matrix.index==book_name)[0][0]

    similar_items = list(enumerate(similarity_scores_books[index]))
    similar_items = sorted(similar_items,key = lambda x:x[1],reverse=True)[1:8]

    book_recommend = []
    for i in similar_items:
        temp_df = ratings_books[ratings_books['Book-Title'] == data_matrix.index[i[0]]]
        book_recommend.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
    return book_recommend

import streamlit as st
st.title("Book Recommender System")
book_title = st.sidebar.text_input("Enter a Book Title") 
if st.sidebar.button("Recommend"):
  if book_title:
    recommended_books = recommend_books(ratings_books, book_title)
    st.subheader("Recommended Books:")
    st.write(recommended_books)
