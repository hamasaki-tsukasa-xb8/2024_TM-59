import numpy as np
# 類似率の行列（人間Aは傾向Aとどれくらい似ているのか？） 行　人間、　列　傾向

def make_similality_matrix():
    similarity_matrix = np.array([
    [0.80, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.80, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.80, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.80, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.80]])
    return similarity_matrix


# Function to select a random number from 1 to 5 and add it to the array if not already present
def select_and_add(num, matrix):
    if num not in matrix:
        matrix = np.append(matrix, num)
        return num


def make_tendency_matrix():
    # ある特徴を持つ人間のサイト遷移確率　（傾向Aから傾向E）　5つの要素は傾向AからE　行　サイト、　列　サイト
    # Number of sites and tendencies
    num_sites = 5
    num_tendencies = 5
    change_matrix = np.array([])

    # Initialize the transition probabilities matrix with zeros
    tendency_transition_probabilities = np.zeros((num_tendencies, num_sites, num_sites))

    # Assign high transition probability to one unique site for each tendency and ensure other conditions
    for t in range(num_tendencies):
        change_matrix = np.array([])

        for i in range(num_sites):
            # Assign a high probability (>= 0.5) to one unique site
            high_prob_site = np.random.choice(num_sites)
            select_number = select_and_add(high_prob_site, change_matrix)
            tendency_transition_probabilities[t, i, select_number] = np.random.uniform(0.5, 0.85)
            
            # Assign probabilities to other sites ensuring they sum to 1
            remaining_prob = 1.0 - tendency_transition_probabilities[t, i, select_number]
            other_probs = np.random.dirichlet(np.ones(num_sites - 1)) * remaining_prob
            
            # Ensure the second highest probability is >= 0.15
            sorted_other_probs = np.sort(other_probs)
            if sorted_other_probs[-1] < 0.15:
                diff = 0.15 - sorted_other_probs[-1]
                sorted_other_probs[-1] += diff
                sorted_other_probs[:-1] -= diff / (num_sites - 2)
            
            # Assign the probabilities to the other sites
            idx = 0
            for j in range(num_sites):
                if j != select_number:
                    tendency_transition_probabilities[t, i, j] = sorted_other_probs[idx]
                    idx += 1

    # Ensure no negative probabilities
    tendency_transition_probabilities = np.clip(tendency_transition_probabilities, 0, None)

    # print("======================================================================================")

    # Round the transition probabilities to 2 decimal places
    tendency_transition_probabilities = np.round(tendency_transition_probabilities, 2)

    # Print the transition probabilities matrix
    # print("Tendency Transition Probabilities (5x5x5 matrix):")
    # print(tendency_transition_probabilities)

    return tendency_transition_probabilities

make_tendency_matrix()