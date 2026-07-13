# Anu_capstone_project_part3
### Decision Tree Baseline

-   **Training Accuracy (Decision Tree):** 0.8317
-   **Test Accuracy (Decision Tree):** 0.7590

This Decision Tree model likely shows **signs of overfitting** because the training accuracy (0.8317) is significantly higher than the test accuracy (0.7590). This indicates that the model learned the training data, including its noise, too well and struggles to generalize to new, unseen data. Decision Trees are described as **high-variance models** because they fit the training data greedily at each split without revisiting earlier decisions. This means they can capture very specific patterns and noise in the training data, leading to a model that performs exceptionally well on the data it has seen but generalizes poorly to new, unseen data.

### Controlled Decision Tree

-   **Training Accuracy (Controlled Decision Tree):** 0.6971
-   **Test Accuracy (Controlled Decision Tree):** 0.6972

By setting `max_depth=5` and `min_samples_split=20`, the overfitting observed in the baseline Decision Tree has been significantly reduced. The gap between training and test accuracy is now very small (0.6971 vs 0.6972), indicating much better generalization performance.

-   **`max_depth`**: This parameter limits how deep the tree can grow. By restricting the depth, the model is prevented from creating very complex decision rules that might perfectly fit the training data but fail on new data. This **reduces the variance** of the model (makes it less sensitive to the specific training data) at the cost of potentially increasing bias (making it less able to capture true underlying patterns).

-   **`min_samples_split`**: This parameter sets the minimum number of samples required to split an internal node. If a node has fewer samples than this threshold, it will not be split further. This prevents the tree from creating splits that are based on very few data points, which are often noisy or specific to the training set. It helps in **avoiding splits that respond to noise in small subsets**, thereby regularizing the model and reducing overfitting.

### Gini vs. Entropy Comparison

-   **Test Accuracy (Decision Tree - Gini, max_depth=5):** 0.6972
-   **Test Accuracy (Decision Tree - Entropy, max_depth=5):** 0.6967

Both Gini impurity and Entropy are measures used to determine the quality of a split in a Decision Tree. They quantify the impurity or disorder of a set of samples.

-   **Gini Impurity Formula:** $G = 1 - \sum_{i=1}^{C} p_i^2$
    *   Where $p_i$ is the probability of an item belonging to class $i$ and $C$ is the number of classes.

-   **Entropy Formula:** $E = -\sum_{i=1}^{C} p_i \log_2(p_i)$
    *   Where $p_i$ is the probability of an item belonging to class $i$ and $C$ is the number of classes.

**What it means for a node to have Gini = 0:**
If a node has a Gini impurity of 0, it means that all samples within that node belong to the same class. In other words, the node is **pure**. There is no disorder, and all the data points are homogeneous with respect to the target variable. Such a node would not need to be split further, as it perfectly classifies the samples within it.

### Random Forest Classifier

-   **Training Accuracy (Random Forest):** 0.7058
-   **Test Accuracy (Random Forest):** 0.7051
-   **ROC-AUC (Random Forest):** 0.7777

**Top 5 Features by Importance:**

| Feature          | Importance |
| :--------------- | :--------- |
| Quantity         | 0.7816     |
| CustomerID       | 0.1086     |
| InvoiceHour      | 0.0365     |
| InvoiceMonth     | 0.0195     |
| InvoiceDayOfWeek | 0.0157     |

#### How Random Forest Computes Feature Importance

In a Random Forest, feature importance is typically calculated based on the **mean decrease in impurity** (e.g., Gini impurity or entropy) or **mean decrease in accuracy** if the model is trained with `feature_importances_` set to 'permutation'. When using impurity-based importance, for each feature, the Random Forest algorithm measures how much the impurity decreases (or how much information gain occurs) when that feature is used for splitting across all trees in the forest. The importances are then averaged across all trees and normalized to sum to 1. A higher value indicates that the feature is more effective at splitting the data into pure nodes, thus being more important for prediction.

This method differs significantly from **linear regression coefficients** because:
1.  **Non-linear Relationships:** Random Forest can capture non-linear relationships and interactions between features, whereas linear regression coefficients assume a linear relationship. An important feature in a Random Forest might not have a large linear coefficient.
2.  **Units and Scaling:** While our features are scaled for both, Random Forest's impurity reduction is less directly tied to the scale of the feature itself, focusing more on its discriminatory power across different splits. Linear regression coefficients are directly influenced by the scale of the features.
3.  **Ensemble vs. Single Model:** Random Forest aggregates information from many trees, providing a more robust measure of importance compared to a single linear model's coefficients, which can be unstable or misleading in the presence of multicollinearity.

#### Bagging Concept in Random Forest

Random Forest utilizes a technique called **bagging** (Bootstrap Aggregating) to build an ensemble of decision trees. The core ideas behind bagging are:

1.  **Bootstrap Sampling:** For each tree in the forest, a **random sample of the training data is drawn with replacement** (a bootstrap sample). This means that each tree is trained on a slightly different subset of the original data, and some instances may appear multiple times in a single tree's training set, while others may not appear at all.
2.  **Random Feature Subset for Splits:** At each node split in a decision tree, instead of considering all features, only a **random subset of features** (typically $\sqrt{\text{number of features}}$ for classification tasks) is considered as candidates for the split. This decorrelates the trees, making them more independent.

**How this ensemble averaging reduces variance compared to a single deep decision tree:**
A single deep decision tree (like our baseline) is a high-variance model prone to overfitting because it can learn the noise in the training data too well. By training multiple trees on different bootstrap samples of the data and with random subsets of features, the Random Forest creates diverse individual trees that likely overfit to different aspects of the data. When the predictions from these many diverse trees are averaged (for regression) or voted (for classification), the errors and overfitting tendencies of individual trees tend to cancel each other out. This **averaging process significantly reduces the overall variance** of the ensemble model, leading to a more stable and generalizable prediction, while generally maintaining a low bias.

### Feature Ablation Study for Random Forest

-   **Features Removed (lowest 5 importance scores):** `['Country_Saudi Arabia', 'Country_RSA', 'Country_Czech Republic', 'Country_Bahrain', 'Country_Poland']`
-   **ROC-AUC (Full Random Forest Model):** 0.7777
-   **ROC-AUC (Reduced Random Forest Model):** 0.7788

In this ablation study, the 5 features identified as having the lowest importance scores by the Random Forest model were removed, and a new model was trained on the reduced feature set. The ROC-AUC of the reduced model (0.7788) is slightly *higher* than that of the full model (0.7777).

This result suggests that the removed features were genuinely **uninformative** or possibly even introduced a small amount of noise or distraction to the model. Their removal either had no negative impact or led to a marginal improvement in the model's ability to discriminate between classes on the test set. This aligns with the expectation that features with very low importance scores contribute little to the model's predictive power.

#### Implications of Deploying a Simpler, Lower-Dimensional Model in Production

Deploying a simpler, lower-dimensional model (one with fewer features) can offer several advantages in a production environment, even if the AUC degradation is minimal or, as in this case, slightly improved:

1.  **Inference Cost:**
    *   **Reduced Computational Load:** Fewer features mean less data to process for each prediction. This directly translates to faster inference times and lower CPU/GPU usage per prediction.
    *   **Lower Memory Footprint:** A model with fewer features requires less memory to store the model parameters and to process input data. This can be crucial in environments with limited resources, such as edge devices or real-time prediction services.
    *   **Cost Savings:** Faster inference and lower resource consumption lead to reduced operational costs, especially in cloud-based deployments where billing is often tied to compute time and memory usage.

2.  **Maintenance Burden:**
    *   **Simpler Data Pipelines:** Less features simplify the data preprocessing pipeline. There are fewer transformations to apply, fewer potential sources of errors in data extraction or feature engineering, and less code to maintain.
    *   **Easier Debugging and Monitoring:** With fewer moving parts, it's easier to understand why a model is making a particular prediction and to debug issues if they arise. Monitoring feature drift or data quality is also simpler when dealing with a smaller set of features.
    *   **Reduced Dependency Management:** Each feature might depend on external data sources or complex extraction logic. Removing features reduces the number of these dependencies, making the system more robust and easier to update.

3.  **Acceptable AUC Degradation:**
    *   In this specific case, the AUC *improved* slightly, indicating that removing these features was beneficial. However, even if there was a *small degradation* (e.g., from 0.7777 to 0.7750), it might still be acceptable. The decision to accept AUC degradation is a business one, balancing model performance against operational benefits.
    *   **Business Value vs. Statistical Significance:** A very small difference in AUC might be statistically significant but not practically meaningful for the business. If the cost savings, speed improvements, and reduced maintenance far outweigh a marginal drop in a performance metric, then the simpler model is often preferred.
    *   **Robustness:** Simpler models can sometimes be more robust to changes in data distribution over time, as they rely on fewer potentially volatile features.

In conclusion, removing uninformative features, as demonstrated by this ablation study, is a highly beneficial practice for improving model efficiency and maintainability in production, often without sacrificing (and sometimes even improving) predictive performance.
