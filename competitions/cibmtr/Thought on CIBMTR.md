![image](https://github.com/user-attachments/assets/527c438a-9a17-4144-8a3c-8966ec418d49)Joining the [competition](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions) late, I had the privilege of reviewing useful notebooks and discussions, such as those by [@cdeotee](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003) and [@ambrosM](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/551544). Based on their observations, I decided to use _XGBoost_, _LightGBM_, and _CatBoost_. Why? In recent [Kaggle](https://www.kaggle.com/competitions) competitions, like [Backpack Prediction](https://www.kaggle.com/competitions/playground-series-s5e2), [Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1), and [Insurance Prediction](https://www.kaggle.com/competitions/playground-series-s4e12), these models have performed well across various use cases. I saw this as a great opportunity to learn something new in my data science journey. Here are a few observations I noted in this competition:

### The Data
1. The data was synthetically generated (as mentioned in the competition rules).
2. The dataset includes six race groups, each with approximately 4,800 samples, totaling around 28,800 samples. This equal distribution is unrealistic for natural populations, reinforcing the synthetic nature of the data.
3. A histogram of the target values shows that half of the patients die within 20 months, while those surviving past 20 months have a high probability of living much longer. This suggests two distinct phases: an initial high-risk period followed by a lower-risk period.
4. Survival at 20 months varies by race, ranging from 40% to 60%, with "More than one race" having the highest survival rate and "White" the lowest.
5. The `age_at_hct` variable is unbalanced, as [@ambrosM](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/551544) pointed out: _The age of 0.044 years (i.e., 16 days) occurs 1,005 times in the training dataset, whereas every other age occurs at most six times._

> For this problem, I implemented a **StratifiedKFold** strategy (5 folds):
```python
group_cols = train["efs"].astype(str) + '_' + train["race_group"].astype(str)
``` 
> **Why efs and race_group?** (_We’re not directly stratifying age_at_hct._)  
> In survival analysis, maintaining similar proportions of events (e.g., deaths) and censored observations across folds is critical. This ensures each fold reflects the overall event rate, allowing the model to learn and evaluate survival patterns consistently. The balanced race groups (4,800 samples each) suggest artificial sampling, but survival differences persist. Stratifying by race ensures each fold has a similar distribution of race groups, preserving their impact on survival outcomes.

### Feature Engineering
1. I added more features based on this [notebook](https://www.kaggle.com/code/poemcourt/prl-fe-global-metric).
2. I created a moderator for the target variables, using two targets (**Kaplan-Meier** and **Nelson-Aalen**). I built eight models: 2 XGBRegressor, 4 LGBRegressor, and 2 CatBoostRegressor (one set for each target). Ultimately, I chose Nelson-Aalen as the final predictor (0.6784 CV score and 0.686 LB score).
3. During training, I used target encoding and explored variations (median, mean, max, min), but only median encoding improved the overall CV score (~0.02).

### Training and Observations
1. The more features I added, the higher the LB score. When I removed some added features, the CV score sometimes increased, but the LB score drastically decreased.
2. Stacking more models improved the LB score. Initially, I used a single XGB model, achieving a 0.677 CV score and 0.678 LB score. This increased to 0.683 LB when I combined three models (XGB, LGB, CatBoost).
3. CV scores were highly sensitive to hyperparameter tuning, and adding weights to models increased the LB score (_though this may lead to overfitting_).
4. My final model was a weighted stacked ensemble with a regressor (Elastic Net).

### The Competitors and Their Approaches
1. Many great notebooks include: [Top 2](https://www.kaggle.com/code/karakasatarik/2nd-place-solution-inference), [Top 4](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566528), [Top 5](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566541), [Top 6](https://www.kaggle.com/code/myprofileurl/6th-place-two-step-model), and the [winner’s discussion](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566550).
2. **The Tricks:** ([Winner’s Solution](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566550))
    - Add efs as a feature during training and focus on performance for samples where `efs==1`. Set `efs=1` when making inferences on LB data.
    - Apply sample weights of 0.6:0.4 for `efs==1` and `efs==0`.
    - **Why this trick works:** Initially, I trained the regressor only on samples where `efs==1`. When using this model to infer samples where `efs==0`, it still showed a clear correlation between predictions and ground truth. This is surprising because `efs_time` for `efs==0` is meaningless due to various reasons for data censoring. I suspect this is due to the **SurvGAN** algorithm. Interestingly, similar patterns exist between samples where `efs==0` and `efs==1`. Including samples where `efs==0` significantly improved the regressor’s performance on `efs==1`!
3. **Almost No Feature Engineering in These Top Notebooks.**
4. [Top 2](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566522) pipeline: ![image](https://github.com/user-attachments/assets/ca9112d4-5ed5-42fa-9c15-f8e08d24eee8)

5. All top 10 solutions (**perhaps**) combine two or all three of XGB, LGB, and CatBoost with a neural network (e.g., TabM, GraphNN).

### What Needs to Be Done?
- Perform a more thorough analysis before training. (Survival analysis and ranking prediction are new domains for me to explore.) Experiment with different options to determine the final model pipeline.
- Try additional models, including neural networks, to increase model variety.
