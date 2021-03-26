#' # Automated wine rating prediction
#' This report assesses the performance of the predictor on a partial dataset and discusses 
#' its applicability.
#'
#'
#' # Setup
#' First, we import and setup everything we need.
import sys
import re

from pathlib import Path
from pickle import load as pload
from joblib import load as jload
from scipy.sparse import load_npz
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
#'
#' We can then load our predictor, features and outcome and finally
#' some extra information that we will use to interpret the predictor
#' behaviour.
shared_dir = Path("/usr/share/data")
regressor_file = str(shared_dir / "models/regressor.joblib")
features_file_sparse = str(shared_dir / "data/processed/test_set/X.npz")
features_file_dense = str(shared_dir / "data/processed/test_set/X.npy")
outcome_file = str(shared_dir / "data/processed/test_set/y.npy")
feature_info_file = str(shared_dir / "data/processed/test_set/feature_info.pkl")

regressor = jload(regressor_file)

features_path_sparse = Path(features_file_sparse)
if features_path_sparse.exists():
  X = load_npz(features_file_sparse).todense()
else:
  X = np.load(features_file_dense)

y = np.load(outcome_file)

with open(feature_info_file, 'rb') as ifh:
  feature_info = pload(ifh)
#'
#'
#' # Prediction performance
#'
#' We can now use our trained predictor to predict unseen data (the test set we loaded).
y_pred = regressor.predict(X)
errors = y - y_pred
abs_errors = np.abs(errors)
#'
#' ## Global measures
#' The mean absolute error is essentially the amount by which a typical 
#' prediction is wrong. In that case how many points in the rating scale.
print(np.round(np.mean(np.abs(errors)), 1))
#'
#' The median absolute error is similar although less affected by large errors.
#' Interpretation: half of the predictions have a smaller absolute error and half 
#' have a greater absolute error.
print(np.round(np.median(np.abs(errors)), 1))
#'
#' Another common metric is the root mean square error. 
#' It is similar to the mean absolute error but penalizes more 
#' heavily large errors. This is what the predictor has been optimized for.
print(np.round(np.mean(np.sqrt((errors) ** 2)), 1))
#'
#' Finally, the minimum and maximum errors shows what happens in the best and 
#' worst case scenarios
print(f'Minimum error: {np.round(np.min(np.abs(errors)), 1)}')
print(f'Maximum error: {np.round(np.max(np.abs(errors)), 1)}')
#'
#' An average absolute error of 1.6 points on a 80 - 100 scale is good but
#' more details are needed to assess the performance of the predictor.
#'
#'
#' ## Distribution of errors
#' The histogram of errors shows us a more detailed picture. 
#' Most are centered around the average and are ranging from -3 to 3.
plt.hist(errors)
plt.xlabel("Wine score error")
plt.ylabel("Frequency")
plt.show()
#'
#' To get a more precise view of the predictor preformance, we can take a look 
#' at the scatterplot of true versus predicted ratings.
#' A perfect predictor would have all points on the diagonal line.
#' Our predictor has a tendency to underestimate high ratings and 
#' overestimate low ratings.
plt.scatter(y, y_pred)
plt.xlabel("True rating")
plt.ylabel("Predicted rating")
line_min = min(np.min(y), np.min(y_pred))
line_max = max(np.max(y), np.max(y_pred))
plt.plot([line_min, line_max], [line_min, line_max], color='k', linestyle='-', linewidth=1)
plt.xlabel("True wine score")
plt.ylabel("Predicted wine score")
plt.show()
#'
#' It may be easier to assess the performance in terms of accuracy, over and underestimation of the ratings.
total = len(errors)
print(f"{np.round(100 * np.sum(abs_errors < 0.5) / total, 1)} % of the wines are almost perfectly well predicted (i.e would get the exact true rating if rounded)")
print(f"{np.round(100 * np.sum(abs_errors <= 1) / total, 1)} % of the wines are very well predicted or better (at most one point away from the true rating)")
print(f"{np.round(100 * np.sum(abs_errors <= 2) / total, 1)} % of the wines are well predicted or better (at most two points away from the true rating)")
print(f"{np.round(100 * np.sum(abs_errors <= 3) / total, 1)} % of the wines are moderately well predicted or better (at most three points away from the true rating)")
#'
#' ## Under and overestimation and their consequences
#' We see below that our predictor overestimates ratings slightly more often than it underestimates it.
print(f"{np.round(100 * np.sum(errors > 2) / total, 1)}% of the wines are predicted more than 2 points too high")
print(f"{np.round(100 * np.sum(errors > 3) / total, 1)}% of the wines are predicted more than 3 points too high")
print()

print(f"{np.round(100 * np.sum(errors < -2) / total, 1)}% of the wines are predicted more than 2 points too low")
print(f"{np.round(100 * np.sum(errors < -3) / total, 1)}% of the wines are predicted more than 3 points too low")
#'
#' Note all errors are equal. In particular, overestimating the rating of expensive wines is very much 
#' undesirable.
#' We see below that the price of wines with overestimated ratings (by more than 3 points) is well above the 
#' average. 
#' Caution will therefore be needed when considering the ratings of expensive  wines.
#' On the contrary, wines with underestimated ratings tend to be slightly less expensive than the average.
price = np.squeeze(np.asarray(X[:,1]))
print("Minimum, median, average, and maximum wine price")
print(f"{np.min(price)}   {np.median(price)}   {np.round(np.mean(price), 1)}   {np.max(price)}")
print()
print("Minimum, median, average, and maximum price for wines with overestimated ratings")
over_pred = errors > 3
price_over = price[over_pred]
print(f"{np.min(price_over)}   {np.median(price_over)}   {np.round(np.mean(price_over), 1)}   {np.max(price_over)}")
print()
print("Minimum, median, average, and maximum price for wines with understimated ratings")
under_pred = errors < -3
price_under = price[under_pred]
print(f"{np.min(price_under)}   {np.median(price_under)}   {np.round(np.mean(price_under), 1)}   {np.max(price_under)}")
#'
#'
#' # Predictor behaviour interpretation
#' Knowing how our predictor performs is one thing, but knowing how it uses the 
#' information in the dataset is necessary to know how confident we can be in its 
#' predictions and allows us to foresee potential pitfalls.
importances = regressor.estimator_.feature_importances_
n_top_features=20
n_top_words=20

col_names = feature_info["feature_names"]
sorted_idx = importances.argsort()[::-1]
sorted_col_names = np.array(col_names)[sorted_idx]
sorted_mean_imp = importances[sorted_idx]
#'
#' We can see below the 20 features (i.e. data characteristics) considered most
#' important by the predictor.
#' The price plays an important role which is not unexpected given its usual 
#' correlation with product quality in general.
#' The winery, region, province, and country of origin are also very useful for prediction
#' which is not unexpected.
#' Interestingly, the taster is very important too which suggest some are harsher
#' critics than others.
#' Finally, the year is also among the important features as could be expected.
#' Note that none of the taster descriptions are among the top features.
fig, ax = plt.subplots()
ax.barh(width=sorted_mean_imp[:n_top_features],
        y=sorted_col_names[:n_top_features])
ax.set_title("Feature importances")
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()
#'
#' Since there are no topics in the 20 best features, the paragraph below is irrelevant.
#'
#' Among the features above, some are called "topics". These come from the description, title and 
#' designation fields and are essentially groups of words that have been classified as pertaining 
#' to a common topic.
#' A topic can contain a large number of words so we display the 20 most important words of the 20
#' most important topics to get a better idea of what they are about. 
#' Note that partial words are often found because we performed stemming which consists in replacing 
#' a word by its root in order to group similar words in a single term.
for col_name in sorted_col_names[:n_top_features]:
    match = re.match("^([^_]+)_topic_([0-9]+)$", col_name)
    if match is not None:
        feature, index = match.groups()
        index = int(index)
        decomposition_component = feature_info[f"{feature}_decomposition_components"][index]
        vectorizer_tokens = feature_info[f"{feature}_vectorizer_tokens"]
        
        s = f"{col_name}: "
        s += " ".join([vectorizer_tokens[i]
                       for i in decomposition_component.argsort()[:-n_top_words - 1:-1]])
        print(s)
        print()
#'
#'        
#' # Discussion and conclusion
#' ## Data range and predictor applicability
#' The very first point that must be discussed pertains to the provided data.
#' All rating are between 80 and 100. This hints for a scale ranging between 0 and 100
#' which indicates that only good wines have been included in the dataset. As such, the performance
#' assessed in this report only makes sense for wines whose rating would already be in that range.
#' In practice, this limits the applicability of our predictor to wines already assumed to be in the 
#' top 20% ratings.
#'
#' ## Performance in practice
#' An overall absolute error of 1.6 points on a scale to 80 to 100 is good but this measure is not
#' sufficient to assess the usefulness of the predictor.
#' We report 67.4% of the predictions with an absolute error smaller than or equal to 2 points, 
#' and 86.6% smaller than or equal to 3 points.
#' This should allow the client to easily screen a large catalog and accurately select 
#' wines of the desired quality for a more thorough evaluation.
#'
#' However, the average price of the most overestimated ratings (i.e. by 3 points or more), is much higher
#' than the global average (54.7 versus 34.4). We therefore advise caution when considering the 
#' predicted ratings of expensive wines.
#' 
#' ## Reliability
#' As mention above, the predictor is not reliable for wines whose ratings are expected to be below 
#' 80. For those in the 80 - 100 range, the predictor strongly relies on the price to make its 
#' predictions which is not unexpected.
#' However, this can be easily abused, especially since inflating prices to suggest quality is not
#' an uncommon business practice.
#'
#' The country, region and province of origin, along with the grape variety define the character 
#' of a wine which explains their importance here.
#' The taster also has some influence on the prediction. This quite expected as some can be
#' harsher raters than others.
#' Finally, wine quality is known to vary between year which explains why this feature is important.
#'
#' The fact that no feature using textual information (title, description, designation) indicates
#' that the extraction of relevant information form unstructured text is difficult.
#' This may change with a larger training set however.
#'
#' Overal these results make us confident that the predictor is using relevant data to make its 
#' predictions.
#'
#' ## Conclusion
#' To sum up, we assess this predictor to be suitable for the screening of large catalogs, provided
#' the wines are known to be of high enough quality, while keeping in mind that the rating of expensive
#' wines may be overestimated.
#'
#' Using more sophisticated methods and training the resulting predictor on a larger dataset would 
#' unquestionably improve its accuracy.
