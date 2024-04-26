import numpy as np
from sklearn.feature_selection import mutual_info_regression

def calculate_entropy():

    # Calculate the joint entropy
    joint_entropy = mutual_info_regression(X_scaled, y_scaled)

    # Calculate the marginal entropy for each input variable
    marginal_entropy = []
    for i in range(X_scaled.shape[1]):
        entropy = mutual_info_regression(X_scaled[:, i].reshape(-1, 1), y_scaled)
        marginal_entropy.append(entropy)

    # Calculate the average joint entropy
    average_joint_entropy = np.mean(joint_entropy)

    # Calculate the average marginal entropy
    average_marginal_entropy = np.mean(marginal_entropy)

    # Calculate the average conditional entropy
    average_conditional_entropy = average_joint_entropy - average_marginal_entropy

    # Calculate the average mutual information (transinformation)
    average_mutual_information = average_marginal_entropy - average_conditional_entropy

    return average_marginal_entropy, average_joint_entropy, average_conditional_entropy, average_mutual_information

# Assuming you have a dataset called 'data' with shape (n_samples, 14) where the last column is the output variable

average_marginal_entropy, average_joint_entropy, average_conditional_entropy, average_mutual_information = calculate_entropy()

print("Average marginal Entropy:", average_marginal_entropy)
print("Average Joint Entropy:", average_joint_entropy)
print("Average Conditional Entropy:", average_conditional_entropy)
print("Average Mutual Information (Transinformation):", average_mutual_information)

sheet['A1'] = "average_marginal_entropy"
sheet['B1'] = "average_joint_entropy"
sheet['C1'] = "average_conditional_entropy"
sheet['D1'] = "average_mutual_information"
sheet['E1'] = "Columns"
sheet['A2'] = average_marginal_entropy
sheet['B2'] = average_joint_entropy
sheet['C2'] = average_conditional_entropy
sheet['D2'] = average_mutual_information
sheet['E2'] = "With all Columns"
wb.save('Results.xlsx')
