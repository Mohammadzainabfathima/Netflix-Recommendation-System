# 🎬 Netflix Movie Recommendation System

A content-based movie recommendation system that suggests movies similar to a selected title. This project uses NLP techniques, vectorization, and cosine similarity to replicate the core behavior of Netflix-style recommendations.

---

## 🚀 Project Overview

This project recommends movies based on similarities in:
- Overview text  
- Genres  
- Cast  
- Crew (Director)  
- Keywords  

We combine these features into a single text representation and compute similarity scores across all movies.

---

## 🧠 Recommendation Approach

### ✔ Content-Based Filtering  
The system works by analyzing movie attributes and finding similar titles using:

- **Text Cleaning**
- **Stemming**
- **Count Vectorization**
- **Cosine Similarity**

---

## 🧰 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Jupyter Notebook  
- NLTK (for text preprocessing)

---

## 📂 Dataset

The project uses publicly available TMDB datasets:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These contain metadata such as:
- Title
- Genres
- Cast
- Crew
- Overview
- Keywords

---

## 🔧 How the System Works

1. Load and merge movie + credits data  
2. Extract & clean important features  
3. Convert text into vectors  
4. Compute cosine similarity  
5. Build a function that returns top 5 recommendations  

---

## 📈 Example Recommendation

Input:
Avatar

makefile
Copy code

Output:
John Carter
Guardians of the Galaxy
Star Trek
Prometheus
The Last Airbender

yaml
Copy code

---

## 🛠 Run the Project

1. Clone the repository:
```bash
git clone https://github.com/your-username/netflix-movie-recommendation.git
Install requirements:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python recommend.py
📦 Project Structure
sql
Copy code
📁 Netflix-Recommendation-System
│── recommend.py
│── tmdb_5000_movies.csv
│── tmdb_5000_credits.csv
│── similarity.pkl
│── movies.pkl
│── requirements.txt
└── README.md
🤝 Contributions
Improvements like collaborative filtering, Streamlit UI, or API integration are welcome!

📜 License
MIT License

python
Copy code

---

# ✅ **recommend.py (Full Code From Scratch)**

```python
# -------------------------------
# Netflix Movie Recommendation System
# -------------------------------

import pandas as pd
import numpy as np
import ast
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# -------------------------------
# Helper Functions
# -------------------------------

def convert_obj(text):
    """
    Convert stringified list to actual list.
    Example: "[{'id': 28, 'name': 'Action'}]" → ['Action']
    """
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    """
    Extract only top 3 cast members.
    """
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
        if len(L) == 3:
            break
    return L

def fetch_director(text):
    """
    Extract director name from crew list.
    """
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def remove_spaces(L):
    """
    Remove spaces inside words for better vector matching.
    Example: "Sam Worthington" → "SamWorthington"
    """
    return [i.replace(" ", "") for i in L]

def stem_text(text):
    y = []
    for word in text.split():
        y.append(ps.stem(word))
    return " ".join(y)

# -------------------------------
# Load Dataset
# -------------------------------

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# Keep required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove missing values
movies.dropna(inplace=True)

# Convert JSON columns to lists
movies['genres'] = movies['genres'].apply(convert_obj)
movies['keywords'] = movies['keywords'].apply(convert_obj)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Clean lists
movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(remove_spaces)

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# -------------------------------
# Create "tags" Column
# -------------------------------

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Convert back to string
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Stemming
new_df['tags'] = new_df['tags'].apply(stem_text)

# -------------------------------
# Vectorization
# -------------------------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# -------------------------------
# Recommendation Function
# -------------------------------

def recommend(movie):
    if movie not in new_df['title'].values:
        print("Movie not found!")
        return
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nRecommended movies similar to '{movie}':\n")
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# -------------------------------
# Save Model for Later Use
# -------------------------------

pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# -------------------------------
# Run Example
# -------------------------------
if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)


