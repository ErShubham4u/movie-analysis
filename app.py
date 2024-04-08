import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Data/movie_success_rate.csv")

movies = load_data()

# Suppressing warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load pre-trained model
@st.cache_data
def load_model():
    return joblib.load('Data/model.pkl')
model = load_model()

def plot_top_grossing_movies():
    top_grossing_movies = movies.sort_values(by='Revenue (Millions)', ascending=False).head(10)
    st.subheader("Top Grossing Movies")
    st.write(top_grossing_movies[['Title', 'Revenue (Millions)']].set_index('Title'))

def plot_revenue_trends():
    revenue_trends = movies.groupby('Year')['Revenue (Millions)'].sum()
    st.subheader("Revenue Trends Over the Years")
    st.line_chart(revenue_trends)

def plot_rating_distribution():
    st.subheader("Rating Analysis")
    st.subheader("Distribution of Movie Ratings")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(movies['Rating'], kde=True, ax=ax, color='skyblue')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Movie Ratings')
    st.pyplot(fig)

def plot_rating_vs_revenue():
    st.subheader("Relationship between Ratings and Revenue")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=movies, x='Rating', y='Revenue (Millions)', ax=ax, color='salmon')
    plt.xlabel('Rating')
    plt.ylabel('Revenue (Millions)')
    plt.title('Relationship between Ratings and Revenue')
    st.pyplot(fig)

def plot_genre_distribution():
    genre_distribution = movies['Genre'].value_counts()
    st.subheader("Genre Distribution")
    st.bar_chart(genre_distribution)

def plot_director_analysis():
    director_counts = movies['Director'].value_counts()
    st.subheader("Director Analysis")
    st.subheader("Number of Movies Directed by Each Director")
    st.bar_chart(director_counts)

    director_avg_ratings = movies.groupby('Director')['Rating'].mean()
    st.subheader("Average Rating of Movies Directed by Each Director")
    st.bar_chart(director_avg_ratings)

def plot_runtime_analysis():
    st.subheader("Runtime Analysis")
    st.subheader("Distribution of Movie Runtimes")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(movies['Runtime (Minutes)'], kde=True, ax=ax, color='lightgreen')
    plt.xlabel('Runtime (Minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Movie Runtimes')
    st.pyplot(fig)

    avg_runtime_trends = movies.groupby('Year')['Runtime (Minutes)'].mean()
    st.subheader("Average Runtime Trends Over the Years")
    st.line_chart(avg_runtime_trends)

def plot_metascore_analysis():
    st.subheader("Metascore Analysis")
    st.subheader("Distribution of Metascores")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(movies['Metascore'], kde=True, ax=ax, color='lightcoral')
    plt.xlabel('Metascore')
    plt.ylabel('Frequency')
    plt.title('Distribution of Metascores')
    st.pyplot(fig)

    st.subheader("Relationship between Metascores and Revenue")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=movies, x='Metascore', y='Revenue (Millions)', ax=ax, color='gold')
    plt.xlabel('Metascore')
    plt.ylabel('Revenue (Millions)')
    plt.title('Relationship between Metascores and Revenue')
    st.pyplot(fig)

def plot_yearly_trends():
    yearly_movie_counts = movies['Year'].value_counts().sort_index()
    st.subheader("Yearly Trends")
    st.subheader("Trends in Number of Movies Released Each Year")
    st.line_chart(yearly_movie_counts)

    genre_yearly_distribution = movies.groupby(['Year', 'Genre']).size().unstack(fill_value=0)
    st.subheader("Distribution of Genres Over the Years")
    st.bar_chart(genre_yearly_distribution)

def plot_actor_analysis():
    actors = movies['Actors'].str.split(', ', expand=True).stack().value_counts()
    st.subheader("Actor Analysis")
    st.subheader("Number of Movies Each Actor Has Appeared In")
    st.bar_chart(actors)

def plot_success_analysis():
    st.subheader("Success Analysis")
    st.subheader("Distribution of Movies by Success Category")
    st.bar_chart(movies['Success'].value_counts())

    success_proportion = movies['Success'].value_counts(normalize=True)
    st.subheader("Proportion of Successful vs. Unsuccessful Movies")
    st.bar_chart(success_proportion)

from sklearn.impute import SimpleImputer

def movie_success_prediction(input_data):
    st.subheader("Movie Success Prediction")

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.transform(input_data)

    # Make predictions
    predictions = model.predict(X_imputed)

    # Map predictions to labels
    success_labels = {0: "Not Successful", 1: "Successful"}
    predicted_labels = success_labels[predictions[0]]  # Assuming only one prediction is made

    # Display predictions
    st.write("Predicted Success:", predicted_labels)



def main():
    st.title("Movie Analysis Dashboard")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    sections = [
        "Top Grossing Movies",
        "Revenue Trends Over the Years",
        "Distribution of Movie Ratings",
        "Relationship between Ratings and Revenue",
        "Genre Distribution",
        "Director Analysis",
        "Runtime Analysis",
        "Metascore Analysis",
        "Yearly Trends",
        "Actor Analysis",
        "Success Analysis",
        "Movie Success Prediction"
    ]
    section = st.sidebar.radio("Go to", sections)
    



    

    # Display selected section
    if section == "Top Grossing Movies":
        plot_top_grossing_movies()
    elif section == "Revenue Trends Over the Years":
        plot_revenue_trends()
    elif section == "Distribution of Movie Ratings":
        plot_rating_distribution()
    elif section == "Relationship between Ratings and Revenue":
        plot_rating_vs_revenue()
    elif section == "Genre Distribution":
        plot_genre_distribution()
    elif section == "Director Analysis":
        plot_director_analysis()
    elif section == "Runtime Analysis":
        plot_runtime_analysis()
    elif section == "Metascore Analysis":
        plot_metascore_analysis()
    elif section == "Yearly Trends":
        plot_yearly_trends()
    elif section == "Actor Analysis":
        plot_actor_analysis()
    elif section == "Success Analysis":
        plot_success_analysis()
    elif section == "Movie Success Prediction":
        #Input field of each features
        rank = st.number_input("Rank")
        runtime = st.number_input("Runtime (Minutes)")
        rating = st.number_input("Rating")
        votes = st.number_input("Votes")
        revenue = st.number_input("Revenue (Millions)")
        metascore = st.number_input("Metascore")
        action = st.checkbox("Action")
        adventure = st.checkbox("Adventure")
        animation = st.checkbox("Animation")
        biography = st.checkbox("Biography")
        comedy = st.checkbox("Comedy")
        crime = st.checkbox("Crime")
        drama = st.checkbox("Drama")
        family = st.checkbox("Family")
        fantasy = st.checkbox("Fantasy")
        history = st.checkbox("History")
        horror = st.checkbox("Horror")
        music = st.checkbox("Music")
        musical = st.checkbox("Musical")
        mystery = st.checkbox("Mystery")
        romance = st.checkbox("Romance")
        scifi = st.checkbox("Sci-Fi")
        sport = st.checkbox("Sport")
        thriller = st.checkbox("Thriller")
        war = st.checkbox("War")
        western = st.checkbox("Western")
        if st.button("Predict"):
            input_data = [[rank, runtime, rating, votes, revenue, metascore, action, adventure, animation, biography,
                           comedy, crime, drama, family, fantasy, history, horror, music, musical, mystery, romance,
                           scifi, sport, thriller, war, western]]  
        movie_success_prediction(input_data)
        

if __name__ == "__main__":
    main()
