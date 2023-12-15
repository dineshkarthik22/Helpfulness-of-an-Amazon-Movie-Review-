### Study to Predict Helpfulness Index of Amazon Reviews

# ABSTRACT
Customer reviews play a significant role in influencing the decisions of potential buyers. Reviews are also collected by major corporations to generate product and service recommendations. Amazon movie reviews often face the issue of being inundated with numerous reviews for a specific film. Many of these reviews can be of low quality and readability. Because of these reasons, gaining insights about a product becomes a potentially tedious and time-consuming task. Analyzed numeric, textual, psychological, and temporal factors apart from the content of the reviews that influence the helpfulness of a review. Among all machine learning models that are compared, the XGBoost model performs the best for the prediction task demonstrating minimal Mean Squared Error.

# INTRODUCTION
Amazon has the unique option of voting on a review as “Helpful”. This enhances the process of informed decision-making and economizes users’ time by mitigating the necessity to peruse all reviews. Identifying helpful reviews allows individuals to quickly assess the most relevant and insightful opinions, streamlining their decision-making process. Helpful reviews are likely to contain more detailed information, thoughtful analysis, and constructive criticism. By prioritizing such reviews, users can gain a more comprehensive understanding of a movie's strengths and weaknesses, contributing to a more informed viewing choice. A system that recommends helpful reviews improves the overall user experience on the platform. For our use case, analysis of movie reviews not only benefits users but also allows filmmakers to quickly gauge public opinion. The goal of this project is to thoroughly analyze all linguistic, psychological, and temporal factors that can influence a user's decision to classify a review as helpful or not.

## DATASET
The total number of data points available in the Amazon movies review dataset was 4.6 million reviews. Since this was a very large corpus the data was shuffled and 500,000 reviews were randomly chosen. [8]

| Statistic                     | Value    |
|-------------------------------|----------|
| Number of Reviews             | 108,571  |
| Number of Reviewers           | 83,811   |
| Number of Movies              | 4,706    |
| Time Period                   | Aug '97 - Jul '14  |
| Mean Helpfulness Index        | 0.69     |
| Reviews with No Helpful Votes | 238,736  |

**Table 1:** Statistics on Dataset Considered

### Data Pre-processing
1. According to, Yang et al. [1] 80% of the reviews have fewer than five votes. Additionally, newly authored reviews and less well-known products have fewer opportunities to be read by other customers, and thus, cannot receive many votes. Hence, reviews with no helpfulness votes are removed. The intuition behind removing reviews with zero votes is that while training we do not wish to create a negative bias for reviews with no votes.
2. Products that have less than 10 reviews in total are removed from the dataset. Since with less number of reviews, the helpfulness ratio is skewed. Resulting in a dataset of 108,57 reviews.
3. Outliers where the helpfulness index was greater than 1 and less than -1 were discarded.

### Exploratory Data Analysis
# Polarity and ratings
To get a better score for the sentiment of reviews, the polarity score is calculated using TextBlob. The polarity suggests a Gaussian distribution. Whereas the rating distribution suggests spikes at ratings 1, 4, and 5. This prompts us to infer that even though users are rating movies on a high or low extreme, they are not likely to be negatively or positively polarized towards it. As described in the box plot, with changing ratings the increase in polarity is very gradual.

# Review Content
Analysis of the most frequently occurring words in a sentence. To find the most frequently occurring words the stopwords and punctuations are removed and then the words are stemmed. In the context of movie reviews the stopwords from English corpora are appended with {"movie", "film", "see", "watch", "dvd"}. The results from this analysis are used in the bag of words model.

# Helpfulness Index
The helpfulness votes ratio histogram does not follow any probabilistic distribution. From the box plot, we can infer that the reviews with higher ratings tend to have a higher helpfulness index.

# PREDICTIVE TASK
In the course of our exploratory data analysis, we discern intriguing trends concerning the helpfulness index, notable correlations with the rating, polarity, age of the review, and the number of reviews and reviews by the user. The goal of this project is to predict the helpfulness index of a given review and rank the reviews based on their helpfulness for a particular movie. Based on the features of movie reviews, the model should learn their impact on the helpfulness index and predict the helpfulness of new reviews.

# Metrics Evaluation
To evaluate the value predicted by the model, we can use the Mean Squared Error (MSE), and Mean Absolute Error (MAE). The dataset is trained on a training set and regularized based on the validation set MSE. The model with the least MSE across training, validation, and test sets is considered to be the best model for this prediction task.

# Correlation Matrix
From Figure 10, Correlations between the overall rating of a movie and the review’s polarity are strongly correlated and this agrees with our general intuition that when a review has positive polarity it is highly likely to have a higher rating compared to neutral, negative reviews. The length of a review and its score exhibit a certain degree of correlation, as one would anticipate. This aligns with our earlier observation that products receiving higher reviews also tend to have longer reviews. The review length of a review is normalized based on the minimum and maximum length values.

## MODEL DESCRIPTION

# Overview
For building this predictive model, various regression models within the scikit-learn module in Python can be employed to forecast the helpfulness of a review. In this particular project, we have considered a range of models, including Linear Regression, Ridge Regression, Decision Trees, Random Forest Regressor, and XGBoost Regressor. The scikit-learn module allows for parameter customization, such as the regularizer for Ridge Regression. Some parameters like n_estimators and max_depth are available to fine-tune the Random Forest Regressor. The performance of these models is assessed against both the baseline model and each other in a comparative analysis.

# Model Comparison
For this Amazon movie reviews dataset considered, the regular Linear regression and Ridge regression models performed poorly when compared to XGBoost and Random Forest Regression models. The latter had a good MSE when compared to the former. Between the latter, XGBoost performed the best and had a very low MSE value. Decision trees performed poorly even after fine-tuning. After undergoing multiple epochs of training, the Linear Regression model began to exhibit signs of overfitting on the training data, resulting in an elevated Mean Squared Error (MSE) in both the validation and test sets. The XGBoost algorithm demonstrates superior performance, even in a cold-start scenario where features such as "Review_By_User," "Num_Reviews," and "Confirmity_Val" have minimal impact on the model's overall performance. Whereas models like Random Forest Regressor and Decision trees had a significant increase in MSE.


| Model                    | MSE               | MAE               |
|--------------------------|-------------------|-------------------|
| XGBoost                  | 0.0571            | 0.1992            |
| Random Forest Regressor  | 0.0590            | 0.2030            |
| Ridge Regression         | 0.0645            | 0.2191            |
| Linear Regression        | 0.0645            | 0.2191            |
| Decision Trees           | 0.1105            | 0.2425            |

**Table 2:** Results obtained

### RESULTS

# Scores
The models were trained against 86856 movie reviews and were tested against 21715 movie reviews. The helpfulness index value is predicted for a movie and the top 10 reviews for a particular movie (query) are ranked based on the helpfulness index value. The error in predicted and actual values is calculated using Mean Square Error. The MSE of the baseline model is 0.084 while the MSE of the XGBoost model is 0.057. This indicates that the inclusion of features discussed above played a crucial role in improving the model’s performance.

# Significance of results
There is a high chance that a new review, for an existing movie, that is potentially helpful is ranked low because of significantly fewer votes. Using the model we can predict the ratings for the latest reviews and assign a relevant rank to the same.

### CONCLUSION
An ensemble model XGBoost performs well for the Amazon Movie Reviews dataset with an MSE of 0.057, and MAE of 0.199 and can produce a ranking of reviews based on helpfulness. Features like the polarity, overall rating, and the number of reviews for a movie are significant in determining the helpfulness ratio. Including one hot encoding of the most common words did not improve the MSE across all the models considered.

## REFERENCES
[1] Yang, Y.; Yan, Y.; Qiu, M.; Bao, F. Semantic analysis and helpfulness prediction of text for online product reviews. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing, Beijing, China, 26–31 July 2015; pp. 38–44.
[2] Hudgins, Triston; Joseph, Shijo; Yip, Douglas; and Besanson, Gaston () "Identifying Features and
Predicting Consumer Helpfulness of Product Reviews," SMU Data Science Review: Vol. 7: No. 1, Article 11.
[3] C Danescu-Niculescu-Mizil, G Kossinets, J Kleinberg and L Lee. How Opinions are Received by Online Communities: A Case Study on Amazon.com Helpfulness Votes. Proceedings of WWW, pp. 141--150, 2009.
[4] Alsmadiv, Abdalraheem, Shadi Alzu'bi, Mahmoud Al-Ayyoub and Yaser Jararweh. “Predicting Helpfulness of Online Reviews.” ArXiv abs/2008.10129 (2020): n. Pag.
[5] Park, Yoon-Joo. 2018. "Predicting the Helpfulness of Online Customer Reviews across Different Product Types" Sustainability 10, no. 6: 1735.
[6] D Chehal, P Gupta, P Gulati Predicting the Usefulness of E-Commerce Products’ Reviews using Machine Learning Techniques. International Journal of Computing and Informatics November 2023.
[7] Y. Liu, X. Huang, A. An, and X. Yu. Modeling and predicting the helpfulness of online reviews. IEEE international conference on data mining, 2008.
[8] J. McAuley and J. Leskovec. From amateurs to connoisseurs: Modeling the evolution of user expertise through online reviews. International World Wide Web Conference Committee (IW3C2)., 2013
[9] S. Krishnamoorthy. Linguistic features for review helpfulness prediction. Expert Systems with Applications, 2015.
