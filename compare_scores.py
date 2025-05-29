import torch
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the true labels (y) for the validation set.
# You'll need to generate/save this once from your data preparation.
# For example, in VGNAE_LP.test, save `y.detach().cpu()`
# y_true_val = torch.load('y_true_val.pt').numpy() 

# Load predictions for p=0
final_preds_p0 = torch.load('final_preds_p0_val.pt').numpy()
edge_predictions_p0 = torch.load('edge_predictions_p0_val.pt') # Shape [13, num_val_edges]

# Load predictions for p=1
final_preds_p1 = torch.load('final_preds_p1_val.pt').numpy()
edge_predictions_p1 = torch.load('edge_predictions_p1_val.pt') # Shape [13, num_val_edges]

# --- Sanity Checks (Optional but good) ---
# print(f"AUC with final_preds_p0: {roc_auc_score(y_true_val, final_preds_p0)}")
# print(f"AUC with final_preds_p1: {roc_auc_score(y_true_val, final_preds_p1)}")
# These should match your ~91%

# --- Hypothesis 1: final_preds_p0 is very similar to final_preds_p1 ---
similarity_metric = np.corrcoef(final_preds_p0, final_preds_p1)[0, 1]
print(f"Correlation between final_preds_p0 and final_preds_p1: {similarity_metric}")
abs_diff = np.mean(np.abs(final_preds_p0 - final_preds_p1))
print(f"Mean absolute difference: {abs_diff}")
# If correlation is very high (e.g., >0.99) and abs_diff is very low, they are similar.

# --- Hypothesis 2: For p=0, sparse subgraphs give weak predictions ---
# edge_predictions_p0[0] is Score(original_graph, val_edges)
# edge_predictions_p0[1:] are Scores from the 12 sparse subgraphs pg_i_p0

score_original_p0 = edge_predictions_p0[0].numpy()
scores_sparse_avg_p0 = torch.mean(edge_predictions_p0[1:], dim=0).numpy()

print(f"Variance of Score(original_graph, p=0): {np.var(score_original_p0)}")
print(f"Mean of Score(original_graph, p=0): {np.mean(score_original_p0)}")

print(f"Variance of average scores from 12 sparse subgraphs (p=0): {np.var(scores_sparse_avg_p0)}")
print(f"Mean of average scores from 12 sparse subgraphs (p=0): {np.mean(scores_sparse_avg_p0)}")

# You can also look at individual sparse ones:
# for i in range(1, 13):
# print(f"Var of sparse_pg_{i} (p=0): {np.var(edge_predictions_p0[i].numpy())}")
# print(f"Mean of sparse_pg_{i} (p=0): {np.mean(edge_predictions_p0[i].numpy())}")
# Low variance and mean around 0.5 might indicate weak/uninformative predictions.

# Compare final_preds_p0 with score_original_p0 (from edge_predictions_p0[0])
correlation_final_p0_vs_orig = np.corrcoef(final_preds_p0, score_original_p0)[0, 1]
abs_diff_final_p0_vs_orig = np.mean(np.abs(final_preds_p0 - score_original_p0))
print(f"Correlation final_preds_p0 vs. its original_graph component: {correlation_final_p0_vs_orig}")
print(f"Mean abs diff final_preds_p0 vs. its original_graph component: {abs_diff_final_p0_vs_orig}")
# If final_preds_p0 is dominated by original_graph's scores, this correlation will be high.

# --- Hypothesis 3: For p=1, all components are similar to original_graph's score ---
score_original_p1 = edge_predictions_p1[0].numpy() 
# Check if all scores in edge_predictions_p1 are similar to score_original_p1
all_scores_similar_p1 = True
for i in range(1, 13):
    corr = np.corrcoef(score_original_p1, edge_predictions_p1[i].numpy())[0,1]
    mad = np.mean(np.abs(score_original_p1 - edge_predictions_p1[i].numpy()))
    print(f"p=1: Corr between original and pg_{i}: {corr}, MAD: {mad}")
    if corr < 0.98 or mad > 0.05: # Define some similarity threshold
        all_scores_similar_p1 = False
print(f"For p=1, are all processed graph scores similar to original_graph's score? {all_scores_similar_p1}")

# And check similarity of final_preds_p1 to score_original_p1
correlation_final_p1_vs_orig = np.corrcoef(final_preds_p1, score_original_p1)[0, 1]
abs_diff_final_p1_vs_orig = np.mean(np.abs(final_preds_p1 - score_original_p1))
print(f"Correlation final_preds_p1 vs. its original_graph component: {correlation_final_p1_vs_orig}")
print(f"Mean abs diff final_preds_p1 vs. its original_graph component: {abs_diff_final_p1_vs_orig}")
# This correlation should be extremely high if hypothesis holds.
