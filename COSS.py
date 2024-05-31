import math
import random
import pandas as pd
import statistics as stats
import numpy as np
import math
from collections import Counter
from scipy.special import gammaln
import csv
import os

class Service:
    def __init__(self, service_name, labels):
        self.service_name = service_name
        self.labels = labels
        self.score = 0.0
        self.human_score = 0.0
        self.actual_score = 0.0
        self.human_accuracy = 0.0
        self.actual_accuracy = 0.0

    def __str__(self):
        return self.service_name + " " + str(self.score)


class SampleContext:
    def __init__(self, text, ground_truth, labels):
        self.text = text
        self.labels = labels # List of labels from all services
        self.ground_truth = ground_truth
        self.psi = 0.0
        self.updated_psi = 0.0
        self.human_label = -1

    def __str__(self):
        return str(self.text) + " " + str(self.labels) + " " + str(self.psi) + " " + str(self.human_label) + " GT:" + str(self.ground_truth) 
    

class Probability:
    def __init__(self, freq, highest_freq_label, prob):
        self.freq = freq
        self.highest_freq_label = highest_freq_label
        self.prob = prob # Probability of the highest freq label occuring next
        self.prev_prob = 0.0

    def __str__(self):
        return str(self.freq) + " " + str(self.prob)
    

def multivariate_beta(alpha):
    """
    Calculate the multivariate beta function using a numerically stable approach.
    :param alpha: a list of parameters
    :return result: result of multivariate beta
    """
    sum_alpha = np.sum(alpha)
    log_numerator = np.sum(gammaln(alpha))
    log_denominator = gammaln(sum_alpha)
    
    # Subtraction to perform division in logarithmic space
    log_result = log_numerator - log_denominator
    
    # Exponentiate to get the actual result
    result = np.exp(log_result)
    return result


def spearman(ranking_mv, ranking_sample):
    """
    Calculate the difference for two rankings
    :param ranking_mv: the ranking of each service from the majority voting result
    :param ranking_sample: the ranking of each service from a sample data 
    """
    if len(ranking_mv) != len(ranking_sample):
        return -1.0
    
    n = len(ranking_sample)

    # Rank the data
    def rank_data(ranking):
        ranks = {}
        original_ranks = {}
        for i, x in enumerate(sorted(ranking)):
            if x in original_ranks:
                ranks[x] = (original_ranks[x] + i + 1) / 2 
                # ranks[x] = i + 1
            else:
                ranks[x] = i + 1
                original_ranks[x] = i + 1
        return [ranks[x] for x in ranking]

    ranked_mv = rank_data(ranking_mv)
    ranked_sample = rank_data(ranking_sample)

    # Compute the differences between ranks
    differences = [ranked_mv[i] - ranked_sample[i] for i in range(n)]

    # Square the differences
    squared_differences = [d ** 2 for d in differences]

    # Sum up the squared differences
    sum_squared_differences = sum(squared_differences)

    # Calculate Spearman's rank correlation coefficient
    rho = 1 - (6 * sum_squared_differences) / (n * (n**2 - 1))

    return rho


def calc_cert(service_names, services_labels, set_of_contexts):
    """
    Calculate CERT for all samples (or each SOP)
    :param service_names: list of service names
    :param services_labels: a list of lists of labels, one list for each service
    :param set_of_contexts: a list of contexts, one element for each sample
    :return text_contexts: a list of SampleContext objects containing label from each services
    """

    # Extract invocation outcomes
    services = []
    for i in range(len(service_names)):
        services.append(service(service_names[i], services_labels[:, i])) # Copy from the ith column

    # Create a list of SampleContext objects
    text_contexts = []
    for i in range(len(set_of_contexts)):
        labels = []
        for j in range(len(services)):
            labels.append(services[j].labels[i])
        text_contexts.append(SampleContext(set_of_contexts[i][0], set_of_contexts[i][1], labels))

    # Obtain majority voting labels (most frequent label)
    majority_voting_labels = []
    for i in range(services_labels.shape[0]):
        try:
            mode = stats.mode(services_labels[i, :])
        except stats.StatsError:
            # Choose label randomly if there is a tie
            mode = random.choice(services_labels[i].tolist())
        majority_voting_labels.append(mode)

    # Assign scores to services
    for i in range(len(services)):
        score = 0
        for j in range(len(services[i].labels)):
            if services[i].labels[j] == majority_voting_labels[j]:
                score += 1
        services[i].score = score

    # Sort services according to their scores
    sorted_services = sorted(services, key=lambda x: x.score, reverse=True)

    # Store ranking of services based on scores
    ranking_mv = []
    for service in services:
        for i, m in enumerate(sorted_services):
            if service == m:
                ranking_mv.append(i + 1)

    for i in range(len(text_contexts)):
        ranking_sample = []
        match_count = sum(label == majority_voting_labels[i] for label in text_contexts[i].labels)

        # Store ranking of services based on invocation outcomes
        for j in range(len(text_contexts[i].labels)):
            if text_contexts[i].labels[j] == majority_voting_labels[i]:
                ranking_sample.append(1)
            else:
                ranking_sample.append(1 + match_count)

        # Calculate CERT
        text_contexts[i].psi = 1 - spearman(ranking_mv, ranking_sample)
    
    return text_contexts


def update_psi(text_contexts, probs, b):
    """
    Calculate and update PSI for all samples (or each SOP)
    :param text_contexts: a list of SampleContext objects containing label from each services
    :param probs: a dictionary with service labels as key and Probability object as item
    :param b: an exponential weight
    :return sorted_contexts: a list of SampleContext objects sorted based on PSI
    """
    all_labels = [''.join(map(str, context.labels)) for context in text_contexts]
    label_counts = Counter(all_labels)

    updated_text_contexts = text_contexts.copy()
    
    for i in range(len(updated_text_contexts)):
        # Ignore samples that have already been labelled
        if not updated_text_contexts[i].human_label == -1:
            updated_text_contexts[i].updated_psi = float('-inf')
            continue

        labels = ''.join(map(str,updated_text_contexts[i].labels))

        prob = getattr(probs.get(labels, None), 'prob', 1)
        prev_prob = getattr(probs.get(labels, None), 'prev_prob', 0)
        
        total_labels = sum(label_counts.values())
        ratio = label_counts[labels] / total_labels

        # Get stability index
        stability_index = abs(prob - prev_prob)

        # Get stability threshold
        threshold = (1 - ratio) * math.exp(-b * ratio)

        # Update stability gap
        probability = max((stability_index - threshold), 0)

        # Update PSI
        updated_text_contexts[i].updated_psi = text_contexts[i].psi * probability

    # Sort contexts according to their PSI
    random.shuffle(updated_text_contexts)
    sorted_contexts = sorted(updated_text_contexts, key=lambda x: x.updated_psi, reverse=True)
    
    return sorted_contexts


def calc_probabilities(sorted_contexts, probs, label_set, k, beta_param):
    """
    Calculate and update probabilities that will be used in calculation of stability index
    :param sorted_contexts: a list of SampleContext objects sorted based on PSI
    :param probs: a dictionary with service labels as key and Probability object as item
    :param label_set: a list of possible labels
    :param k: amount of contexts to be labelled at one iteration
    :param beta_param: a list of shape parameters for beta distribution (should have the same length as label set)
    :return sorted_contexts: a list of SampleContext objects sorted based on PSI
    :return probs: a dictionary with service labels as key and Probability object as item
    """
    if not len(label_set) == len(beta_param):
        raise ValueError("Expect label_set and beta_param to have the same length")
    
    # Get labels of the top k contexts
    top_k_labels = []
    for i in range(k):
        top_k_labels.append(''.join(map(str, sorted_contexts[i].labels)))
    
    # Get all contexts with the same labels as the top k contexts
    same_labels_contexts = [context for context in sorted_contexts if ''.join(map(str, context.labels)) in top_k_labels]

    # Randomly select k number of contexts from the same_labels_contexts
    selected_contexts = random.sample(same_labels_contexts, k)

    # Get human labels
    human_labels_dict = {}
    for context in selected_contexts:
        labels = ''.join(map(str, context.labels))

        human_label = str(context.ground_truth)

        context.human_label = human_label
        human_labels_dict.setdefault(labels, []).append(human_label)

    # Calculate probabilities (used for stability index)
    for label, human_labels in human_labels_dict.items():
        # Get the count of each labels for each combination of labels
        categorized_human_labels = dict()
        
        if label in probs:
            categorized_human_labels = probs[label].freq
        
        for key, value in Counter(human_labels).items():
            categorized_human_labels[key] = categorized_human_labels.get(key, 0) + value

        for key in label_set:
            if key not in categorized_human_labels.keys():
                categorized_human_labels[key] = 0

        sorted_labels = sorted(categorized_human_labels.items(), key=lambda x:x[1], reverse=True)
        highest_freq_label = sorted_labels[0][0]

        def get_prob(cur_label):
            categorized_labels = categorized_human_labels.copy()

            for label, alpha in zip(sorted(categorized_labels.keys()), beta_param):
                categorized_labels[label] += alpha

            denominator = list(categorized_labels.values())

            categorized_labels[cur_label] += 1
            numerator = list(categorized_labels.values())

            result = multivariate_beta(numerator) / multivariate_beta(denominator)
            return result
        
        prob = get_prob(highest_freq_label)

        if label in probs and probs[label].highest_freq_label != -1:
            human_labels_dict.setdefault(labels, []).append(human_label)
            probs[label].prev_prob = probs[label].prob
            probs[label].prob = prob
            probs[label].freq = categorized_human_labels
            probs[label].highest_freq_label = highest_freq_label
        else:
            probs[label] = Probability(categorized_human_labels, highest_freq_label, prob)

    return sorted_contexts, probs


def overwrite_labels_with_probabilities(probs, text_contexts):
    """
    Extrapolate the label distribution using calculated probabilities
    :param probs: a dictionary with service labels as key and Probability object as item
    :param text_contexts: a list of SampleContext objects
    """
    labelled = dict()
    for context in text_contexts:
        if context.human_label != -1:
            label =  ''.join(map(str, context.labels))
            if label in labelled:
                labelled[label] += 1
            else:
                labelled[label] = 1

    for label_combination, probability_object in probs.items():
        # Retrieve contexts with the specific label combination
        relevant_contexts = [context for context in text_contexts if ''.join(map(str, context.labels)) == label_combination]
        
        # Calculate the desired number of 1s and 0s based on the probability
        total_data_points = len(relevant_contexts)

        if probability_object.highest_freq_label == '1':
            num_ones_prob = probability_object.freq.get('1', 0) / (probability_object.freq.get('1', 0) + probability_object.freq.get('0', 0))
            num_ones_desired = math.ceil(total_data_points * num_ones_prob)
            
            for context in relevant_contexts:
                if num_ones_desired != 0:
                    context.human_label = 1
                    num_ones_desired -= 1
                else:
                    context.human_label = 0
        
        elif probability_object.highest_freq_label == '0':
            num_zeros_prob = probability_object.freq.get('0', 0) / (probability_object.freq.get('1', 0) + probability_object.freq.get('0', 0))
            num_zeros_desired = math.ceil(total_data_points * num_zeros_prob)
            
            for context in relevant_contexts:
                if num_zeros_desired != 0:
                    context.human_label = 0
                    num_zeros_desired -= 1
                else:
                    context.human_label = 1
 
    # Get samples that the SOP they belong has not be chosen
    for context in text_contexts:
        if context.human_label == -1:
            # Overwrite human label with the majority voting label
            try:
                majority_label = stats.mode(context.labels)
            except stats.StatsError:
                majority_label = random.choice(context.labels)
            context.human_label = majority_label


def get_service_accuracy(services, text_contexts):
    """
    Calculate the accuracy of services
    :param services: a list of service objects
    :param text_contexts: a list of SampleContext objects
    """
    for i in range(len(services)):
        services[i].human_score = 0
        services[i].actual_score = 0

        for context in text_contexts:
            if str(context.labels[i]) == str(context.human_label):
                services[i].human_score += 1
            if str(context.labels[i]) == str(context.ground_truth):
                services[i].actual_score += 1

        services[i].human_accuracy = services[i].human_score / len(text_contexts)
        services[i].actual_accuracy = services[i].actual_score / len(text_contexts)
    

def main(sample_size, k, b):
    # Get files with services' invocation outcomes
    df = pd.read_csv('path_to_input_csv')

    names = df.columns.values.tolist()[2:] # Get the name of the services from the header

    set_of_services = df.iloc[:, 2:].to_numpy() # Get the labels of the services
    set_of_contexts = df.iloc[:, 0:2].to_numpy() # Get the contexts

    # Make service objects
    services = []
    for i in range(len(names)):
        services.append(Service(names[i], set_of_services[:, i]))

    label_set = ['0','1']
    beta_param = [1, 1]

    # Get number of iterations
    iteration = math.ceil(sample_size / k)
    text_contexts = calc_cert(names, set_of_services, set_of_contexts)
    probs = dict()
    human_labelled = 0

    for k in range(int(iteration)):
        if human_labelled + k > sample_size:
            k = sample_size - human_labelled
        
        human_labelled += k
        
        sorted_contexts = update_psi(text_contexts, probs, b)
        sorted_contexts, probs = calc_probabilities(sorted_contexts, probs, label_set, k, beta_param)
        for sorted_context in sorted_contexts:
            for text_context in text_contexts:
                if sorted_context.text == text_context.text:
                    text_context.labels = sorted_context.labels
                    text_context.human_label = sorted_context.human_label
                    
    # Extrapolate human labels
    overwrite_labels_with_probabilities(probs, text_contexts)

    get_service_accuracy(services, text_contexts)
        
    output_file = "output_csv_filename"
    header = ['Sample_Size', 'service1','service2','service3']

    file_exists = os.path.exists(output_file)

    results = [-1.0, -1.0, -1.0]

    for service in services:
        if service.service_name == header[1]:
            results[0] = round(service.human_accuracy, 4)
        elif service.service_name == header[2]:
            results[1] = round(service.human_accuracy, 4)
        elif service.service_name == header[3]:
            results[2] = round(service.human_accuracy, 4)

    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow([human_labelled] + results)


if __name__ == '__main__':
    random.seed()
    sample_sizes = [30, 40, 50, 60, 70, 80, 90, 100, 130, 160, 190]
    
    k = 4
    b = 4

    for size in sample_sizes:
        for i in range(200):
            main(size, k, b)
