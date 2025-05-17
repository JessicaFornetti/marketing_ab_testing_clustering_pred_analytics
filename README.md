# marketing_ab_testing_clustering_pred_analytics

This project provides a comprehensive analysis of a [digital marketing campaign dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset/data), using funnel analysis, A/B testing for campaign variants, customer segmentation through clustering, and predictive modeling for conversion outcome in Python. The goal is to gain actionable insights into campaign performance and build a robust model to predict which customers are likely to convert.

This project demonstrates a practical application of data analysis, hypothesis testing, unsupervised learning, and supervised learning techniques in a marketing context, aiming to understand customer behavior, segment audiences effectively, and predict key marketing outcomes, ultimately contributing to more targeted and successful digital marketing strategies.

The main steps of the project are outlined below:
- **Data Loading and Preprocessing:** Involves loading the digital marketing campaign dataset, handling categorical features, and creating new engagement metrics to enrich the analysis.
- **Exploratory Data Analysis (EDA):** A thorough examination of the dataset to understand its characteristics and identify potential patterns. This involved the following steps:
    * Visualizing the distribution of quantitative variables using histograms and boxplots to identify skewness and outliers.
    * Visualizing the frequency of different categories within categorical features using bar charts.
    * Investigating linear relationships between quantitative features using a correlation matrix.
    * Examining conversion rates across different categories within each categorical feature to identify potential drivers of conversion.
    * Comparing the distributions of quantitative features for converting and non-converting customers using violin and kernel density estimate (KDE) plots.
    * Exploring the correlation between features suspected to be related to gain deeper understanding of their interactions.
- **Marketing Funnel Analysis:** Explores user progression for each campaign type(Conversion, Awareness, Retention and Consideration) through each different marketing channels (Email, Social Media, PPC), using interactive bar charts to visualize user drop-off rates at each stage of the funnel.
- **A/B Testing:** to determine which campaign (Conversion or Retention) performs better using t-test, and test if there are signifcant differences between Age, Gender and Income groups for Retention and Conversion campaigns aslso using t-tests, and then conducted separate one-way ANOVA tests for each campaign type and for each of the following factors: Age Group, Gender, and Income Group.
- **A/B Testing and Comparative Analysis:** This section focused on evaluating the performance of different campaign strategies and exploring demographic influences on campaign outcomes. Specifically:
    * **Campaign Performance Comparison:** Independent t-tests were employed to statistically determine if there was a significant difference in the conversion rate between the Conversion and Retention marketing campaigns. Further t-tests were conducted to determine if there was a significant difference in the conversion rate between the Conversion and Retention marketing campaigns on the following demographic profiles :a ge group, gender, and income group.
    * **Demographic Influence on Campaign Outcome:** To understand how different demographic factors impacted the success of each campaign individually, separate one-way ANOVA tests were performed for each campaign type.
- **Customer Segmentation via Clustering:** Comparaison of different clustering methods (Kmeans, Agglomerative Clustering, Gaussian Mixture Model and DBSCAN) to segment customers based on relevant behavioral and demographic features, followed by the analysis of these clusters through PCA and t-SNE visualizations, heatmaps, boxplots, KDE plots and radar profiles to understand distinct customer groups. 3 different customer profiles were identified with specific tailored recommendations for each segment.
- **Predictive Modeling for Conversion:** Focuses on building and comparing classification models (Logistic Regression, Random Forest, XGBoost) to predict the Conversion outcome. This involved the following steps:
    * Preprocessing pipelines with scaling and one-hot encoding.
    * Addressing class imbalance using ADASYN (which worked better than SMOTE and SMOTETOMEK on this dataset).
    * Hyperparameter tuning for each model using GridSearchCV to optimize performance, evaluated using ROC AUC to account for class imbalance.
    * Evaluation of the best models using classic evaluation metrics (AUC, f1 score, precision, recall ...), confusion matrix, ROC curves and Precision-Recall curve.
- **Feature Importance Analysis with SHAP:** Utilizes SHAP (SHapley Additive exPlanations) values to understand the impact of different features on the conversion predictions for all models, to identify customer profiles with higher Conversion rates.
- **Key Insights, Answers to Core Questions, and Strategic Recommendations:** This section synthesizes the findings from the entire analysis to provide actionable insights. It directly addresses the three primary questions guiding the project:
    * **Which campaign performs best?** Identifying top-performing campaigns based on conversion rates, funnel efficiency and A/B tests.
    * **Which customers are likely to convert?** Characterizing the key attributes and behaviors of customer segments with high conversion probabilities, derived from both clustering and SHAP values analysis.
    * **How can marketing efforts be optimized?** Offering concrete recommendations for improving campaign strategies, targeting specific customer segments, and enhancing the overall marketing funnel based on the main takeaways and insight from each stage of the analysis.

**Repository Overview:**

This repository contains 1 [Jupyter notebook](Notebook.ipynb) and a [condensed report of the main insights](Summary%20of%20key%20findings%20and%20recommendations.pdf) .
