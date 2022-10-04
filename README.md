# ANNEXURE - COMMON CONTENT

### YAML Configuration:

Along with the above-mentioned requirements and changes, include a YAML file which contains instructions for the execution of the Usage file.

YAML file contains details regarding input format, output format, and usage file name which included above mentioned main function.

Support input and output options are JSON, text, image, audio & video

| **Key** | **Required** |
| --- | --- |
| version | Optional |
| meta.name | Mandatory |
| output.type | Mandatory |
| input.type | Mandatory |
| requirements | Optional |
| performance | Optional |
| env | Mandatory |

**Example** :

**saved by name:** blobcity.yaml

```
version: 1
meta:
 name: Model1
input:
 type: image
output:
 type: image
requirements: requirements.txt
main: usage.ipynb
performance:
 - accuracy: 0.5
 - f1score: 0.5 
env:
 MODEL_PATH: ./model.pkl
``` 

The relative path of the saved models should be mentioned in the ‘env’ section of the YAML configuration file. One can have n numbers of environment variables utilized in the Main function but should have a matching environment variable mentioned in the YAML file and Main function. The variable mentioned in the env section must be all Capitalized letters without any whitespace.

## Reporting model performance

Include any of the following fields within the performance section of the YAML. It is not compulsory to include the performance section, but it is recommended that you include. Mention as many parameters as possible. These parameters help other users evaluate your model and compare it with other models

| accuracy | **Accuracy Percentage** Indicates the prediction accuracy of the manner, from a range of 0 - 1.0, where 0 means 0% accuracy, and a value of 1.0 means a 100% accuracy. |
| --- | --- |
| r2 | **R squared** Proportion of the variance in the independent variable that is predictable from independent variables. |
| mse | **Mean square error** Average of the square data between the original and predicted values of data` |
| mae | **Mean absolute error** Average of the absolute value of the difference between the true values and predicted values |
| rmse | **Root mean square error** Square root of the second sample moment of the differences between predicted values and observed values of the quadratic mean of the differences |
| f1score | **F1-Score** Weighted average of both precision and recall |
| tpr | **True Positive Rate (TPR)** Ratio between the number of true positives to the total number of true positives and false negatives |
| fpr | **False Positive Rate(FPR)** Ratio between the number of false positives to the total number of false positives and true negatives |
| logloss | **Logarithmic Loss (LogLoss)**Negative average of summation of the product of the observation's actual value to the log of the prediction probability subtracted by (1-yi)ln(1-pi), logloss = [yilnpi + (1-yi)ln(1-pi)] |
| precision | **Precision** Ratio between the number of true positives to the total number of predicted positives |
| recall | **Recall** Ratio between the number of true positives to the total number of true positives and false negatives |
| rmsle | **Root Mean Squared Logarithmic Error (RMSLE)**Measure of the ratio between predicted and actual values as a function of the log |
| ari | **Adjusted Rand index (ARI)**Computes a similarity measure between two clustering by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted or true clusterings |
| fowlkes | **Fowlkes Mallows Score (FMI)**Geometric mean between the precision and recall |
| silhouette | **Silhouette Score** Calculated using the mean intra-cluster distance and mean near-cluster distance for each sample |
| calinski | **Calinski Harabaz Index (CHI)**Variance ratio criterion defined as the ratio between the within-cluster dispersion and the between-cluster dispersion |
| cvscore | **Cross-Validation Score (cv score)**Also known as Monte Carlo cross-validation, creates multiple random splits of the dataset into training and validation data. For each such split, the model is fit to the training data, and predictive accuracy is assessed using the validation data. The results are then averaged over the splits. |
| fid | **Fréchet Inception Distance** Lower the fid, the better the quality. In other words the similarity between real and generated images is close.fid compares the statistics of generated samples to real samples, instead of evaluating generated samples in a vacuum.|
| is | **Inception Score** The Inception Score (IS) is an algorithm used to assess the quality of images created by a [generative](https://en.wikipedia.org/wiki/Generative_model) image model such as a [generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network) (GAN).The score is calculated based on the output of a separate, pretrained [Inceptionv3](https://en.wikipedia.org/wiki/Inceptionv3) image classification model applied to a sample of (typically around 30,000) images generated by the generative model. |
| ndb | **Number of Statistically-Different Bins** Given two sets of samples from the same distribution, the number of samples that fall into a given bin should be the same up to sampling noise. |
| jsd | **Jensen-Shannon Divergence** The Jensen-Shannon divergence is a principled divergence measure which is always finite for finite random variables. It quantifies how "distinguishable" two or more distributions are from each other. |
| lpips | **Learned Perceptual Image Patch Similarity** The Learned Perceptual Image Patch Similarity (LPIPS) is used to judge the perceptual similarity between two images. LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network. This measure has been shown to match human perseption well. A low LPIPS score means that image patches are perceptual similar. |
| mmd | **Maximum mean discrepancy** Maximum mean discrepancy (MMD) is a kernel based statistical test used to determine whether given two distribution are the same which is proposed in [[1]](https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references). MMD can be used as a loss/cost function in various machine learning algorithms such as density estimation, generative models, and also in invertible neural networks utilized in inverse problems. As opposed to generative adversarial networks (GANs) which require a solution to a complex min-max optimization problem, MMD criteria can be used as simpler discriminator. |
| c2st | **Classifier Two-Sample Tests** This test estimates if a target is predictable from features by comparing the loss of a classifier learning the true target with the distribution of losses of classifiers learning a random target with the same average.The null hypothesis is that the target is independent of the features - therefore the loss a classifier learning to predict the target should not be different from the one of a classifier learning independent, random noise. |
| iou | **Intersection over Union** IoU metric in object detection evaluates the degree of overlap between the ground(gt) truth and prediction(pd). IoU is defined as follows area of intersection divided by area of union between ground-truth and predicted box. |
| bleu | **Bilingual evaluation understudy** Bilingual evaluation understudy is an algorithm for [evaluating](https://en.wikipedia.org/wiki/Evaluation_of_machine_translation) the quality of text which has been [machine-translated](https://en.wikipedia.org/wiki/Machine_translation) from one [natural language](https://en.wikipedia.org/wiki/Natural_language) to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation |
| wer | **Word error rate** Word error rate (WER) is a common metric of the performance of [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition) or [machine translation](https://en.wikipedia.org/wiki/Machine_translation) system.The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one). |
| meteor | **Meteor** Meteor evaluates a translation by computing a score based on explicit word-to-word matches between the translation and a given reference translation. If more than one reference translation is available, the translation is scored against each reference independently, and the best scoring pair is used |
| rouge | **Rouge** ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating [automatic summarization](https://en.wikipedia.org/wiki/Automatic_summarization) and [machine translation](https://en.wikipedia.org/wiki/Machine_translation) software in [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation. |
| kappa | **Kappa** The kappa statistic compares the observed accuracy to an expected accuracy or the accuracy expected from random chance. One of the flaws of pure accuracy is that if a class is imbalanced then making predictions at random could give a high accuracy score. Kappa accounts for this by comparing the model accuracy to the expected accuracy based on the number of instances in each class. |
| mcc | **Matthews Correlation Coefficient** The MCC is essentially a correlation coefficient between the observed and predicted classifications. As with any correlation coefficient, its value will lie between -1.0 and +1.0. A value of +1 would indicate a perfect model. |
| ap | **Average Precision** Average precision is the area under the PR curve. AP summarizes the PR Curve to one scalar value. Average precision is high when both precision and recall are high, and low when either of them is low across a range of confidence threshold values. The range for AP is between 0 to 1. |
| map | **Mean Average Precision** The mean Average Precision or mAP score is calculated by taking the mean AP over all classes and/or overall IoU thresholds, depending on different detection challenges that exist |
| cer | **Character Error Rate** CER calculation is based on the concept of [Levenshtein distance](https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510#9bd1), where we count the minimum number of character-level operations required to transform the ground truth text (aka _reference text_) into the OCR output. |
| ar | **Average Recall** Average Recall is the recall averaged over all IoU ∈ [0.5,1.0] and can be computed as two times the area under the recall-IoU curve |
| mrr | **Mean Reciprocal Rank** Evaluate the responses retrieved given their probability of being correct. Used heavily in all information-retrieval tasks, including article search and e-commerce search. |
| mape | **Mean Absolute Percentage Error** MAPE is a measure of prediction accuracy of a forecasting method in [statistics](https://en.wikipedia.org/wiki/Statistics). It usually expresses the accuracy as a ratio. |
| perplexity | **Perplexity** is a measurement of how well a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) or [probability model](https://en.wikipedia.org/wiki/Probability_model)[predicts](https://en.wikipedia.org/wiki/Prediction) a [sample](https://en.wikipedia.org/wiki/Sample_(statistics)). It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample. |

Also include the requirement.txt file in the submission with the required version of the library mentioned in it.

Sample folder structure to generate a snapshot on the BlobCity AI Cloud

![](RackMultipart20221003-1-t7tsc8_html_6d3f00d3dbf45257.png)
