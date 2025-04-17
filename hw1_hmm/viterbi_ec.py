from collections import defaultdict, Counter
from math import log
import numpy as np

def viterbi_ec(train, test, alpha=0.01):
    # 1. Count tag, tag pairs, (word, tag) pairs
    initial_counter = Counter()
    tag_counter = Counter()
    tag_tag_counter = defaultdict(Counter)
    word_tag_counter = defaultdict(Counter)

    for sentence in train:
        initial_counter[sentence[1][1]] += 1
        for i in range(len(sentence)-1):
            if sentence[i][0] not in ["START", "END"]:
                tag1 = sentence[i][1]
                tag2 = sentence[i+1][1]
                tag_counter[sentence[i][1]] += 1
                word_tag_counter[sentence[i][0]][sentence[i][1]] += 1
                tag_tag_counter[tag1][tag2] += 1

    #2. Find each probability and take the log of each probability

    #initial probability
    total_count_for_initial = sum(initial_counter.values())
    initial_probabilities = {tag: np.log(count / total_count_for_initial) \
                             for tag, count in initial_counter.items()}

    # Transition counter - P(tagB|tagA)
    transition_probabilities = {
        prior_tag: {tag: np.log(count / tag_counter[prior_tag]) \
                    for tag, count in tag_counts.items()}
        for prior_tag, tag_counts in tag_tag_counter.items()
    }

    # Apply smoothed_prob for emission probabilities
    emission_probabilities = {}

    for word, tag_counts in word_tag_counter.items():
        counts = [tag_counts.get(tag, 0) for tag in tag_counter]
        smoothed_values = smoothed_prob(counts, 0.01)
        emission_probabilities[word] = {tag: np.log(prob) \
                                        for tag, prob in zip(tag_counter, smoothed_values)}

    unseen_counts = [0] * len(tag_counter)
    unseen_smoothed_values = smoothed_prob(unseen_counts, 0.01)
    unseen_emission = {tag: np.log(prob) for tag, prob in zip(tag_counter, unseen_smoothed_values)}

    def get_smoothed_emission(tag):
        return unseen_emission[tag]

    #3. Find the maximum probability path
    tagged_sentences = []
    for sentence in test:
        core_sentence = sentence[1:-1]  
        n = len(core_sentence)

        # DP table and backpointer
        tag_probability_matrix = [{} for _ in range(n)]
        reminder = [{} for _ in range(n)]

        # initialization step
        valid_emissions = emission_probabilities.get(core_sentence[0], unseen_emission)
        for tag in tag_counter:
            tag_probability_matrix[0][tag] = \
                initial_probabilities.get(tag, EPSILON_LOG) + valid_emissions.get(tag, EPSILON_LOG)

        # recursion step
        for i in range(1, n):
            valid_emissions = emission_probabilities.get(core_sentence[i], unseen_emission)
            previous_tags = tag_probability_matrix[i-1].keys()  
            for current_tag, emission_probability in valid_emissions.items():
                max_probability = float('-inf')
                best_previous_tag = None
                for previous_tag in previous_tags:
                    transition_probabilty = transition_probabilities.get(previous_tag, {}).get(current_tag, EPSILON_LOG)
                    probability = tag_probability_matrix[i-1][previous_tag] + transition_probabilty \
                          + emission_probability
                    if probability > max_probability:
                        max_probability = probability
                        best_previous_tag = previous_tag

                tag_probability_matrix[i][current_tag] = max_probability
                reminder[i][current_tag] = best_previous_tag

        max_final_probability = float('-inf')
        best_final_tag = None
        for tag, probability in tag_probability_matrix[-1].items():
            if probability > max_final_probability:
                max_final_probability = probability
                best_final_tag = tag

        best_path = [best_final_tag]
        for i in range(n-1, 0, -1):
            best_path.insert(0, reminder[i][best_path[0]])

        tagged_sentence = [("START", "START")] + list(zip(core_sentence, best_path)) + [("END", "END")]
        tagged_sentences.append(tagged_sentence)

    return tagged_sentences