# Movie-Review-Sentiment-Analysis
Machine Learning Algorithm that uses the TextBlob library to obtain features/characteristics from movie reviews and predict whether a specific movie review is positive or negative in nature based on its characteristics

I use the same approach outlined in the 2004 Stanford University paper 'Sentiment Extraction and Classification of Movie Reviews'

 link to paper write up here: https://nlp.stanford.edu/courses/cs224n/2004/huics_kgoel_final_project.htm 
 
Using the approach outlined in this paper, I isolate the adjectives in every review since they are the most likely parts of speech to contain information about sentiment towards a movie. After isolating the adjectives, I score them by sentiment. The sum of all of the adjective polarities is the score for the sentiment of the movie. To further refine my algorithm, I reverse the polarity of all adjectives that come after negating words like "no" "never" "not" or "n't". Using this approach, my algorithm can accurately interpret sentiment of a movie review 68% of the time on average but can acheive accuracy as high as 71%.
