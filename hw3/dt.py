import math
import Debug


def divide(data, attr_index, attr_vals_list):
    """Divides the data into buckets according to the selected attr_index.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: A list that includes K data lists in it where K is the number
     of values that the attribute with attr_index can take
    """

    values = attr_vals_list[attr_index]
    buckets = []
    for val in values:
        buckets.append([])

    for example in data:
        attributeValue = example[attr_index]
        bucketIndexForExample = values.index(attributeValue)

        if (bucketIndexForExample < 0 or bucketIndexForExample >= len(buckets)):
            Debug.Print("Wrong indexing in divide function - dt.py")
            continue

        buckets[bucketIndexForExample].append(example)

    return buckets


def entropy(data, attr_vals_list):
    """
    Calculates the entropy in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated entropy (float)
    """

    labels = attr_vals_list[-1]
    labelCounts = [0] * len(labels)

    for example in data:
        exampleLabel = example[-1]
        labelIndex = labels.index(exampleLabel)
        labelCounts[labelIndex] += 1

    negativeEntropy = 0
    exampleCount = len(data)
    Debug.RaiseDataIsEmptyWarning(exampleCount == 0, "entropy")

    for labelCount in labelCounts:
        if (labelCount == 0):
            continue

        p = labelCount / exampleCount
        negativeEntropy += p * math.log(p, 2)

    return -1 * negativeEntropy


def info_gain(data, attr_index, attr_vals_list):
    """
    Calculates the information gain on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: information gain (float), buckets (the list returned from divide)
    """

    attributeExampleCount = len(data)
    Debug.RaiseDataIsEmptyWarning(attributeExampleCount == 0, "info_gain")

    division = divide(data, attr_index, attr_vals_list)

    averageAttributeEntropy = 0
    for subset in division:
        subsetExampleCount = len(subset)
        if (subsetExampleCount == 0):
            continue

        subsetEntropy = entropy(subset, attr_vals_list)
        averageAttributeEntropy += subsetEntropy * (
            subsetExampleCount / attributeExampleCount)

    dataEntropy = entropy(data, attr_vals_list)

    return dataEntropy - averageAttributeEntropy, division


def gain_ratio(data, attr_index, attr_vals_list):
    """
    Calculates the gain ratio on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: gain_ratio (float), buckets (the list returned from divide)
    """

    attributeExampleCount = len(data)
    Debug.RaiseDataIsEmptyWarning(attributeExampleCount == 0, "gain_ratio")

    gain, division = info_gain(data, attr_index, attr_vals_list)

    negativeIntrinsic = 0
    for subset in division:
        subsetExampleCount = len(subset)
        if (subsetExampleCount == 0):
            continue

        if (subsetExampleCount == attributeExampleCount):
            negativeIntrinsic = -0.00001
            break

        countRatio = subsetExampleCount / attributeExampleCount
        negativeIntrinsic += math.log(countRatio, 2) * countRatio

    gainRatio = gain / (-1 * negativeIntrinsic)
    return gainRatio, division


def gini(data, attr_vals_list):
    """
    Calculates the gini index in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated gini index (float)
    """

    labels = attr_vals_list[-1]
    labelCounts = [0] * len(labels)

    for example in data:
        exampleLabel = example[-1]
        labelIndex = labels.index(exampleLabel)
        labelCounts[labelIndex] += 1

    gini = 1
    exampleCount = len(data)
    Debug.RaiseDataIsEmptyWarning(exampleCount == 0, "gini")

    for labelCount in labelCounts:
        if (labelCount == 0):
            continue

        p = labelCount / exampleCount
        gini -= p * p

    return gini


def avg_gini_index(data, attr_index, attr_vals_list):
    """
    Calculates the average gini index on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: average gini index (float), buckets (the list returned from divide)
    """

    attributeExampleCount = len(data)
    Debug.RaiseDataIsEmptyWarning(attributeExampleCount == 0, "avg_gini_index")

    division = divide(data, attr_index, attr_vals_list)

    averageGini = 0
    for subset in division:
        subsetExampleCount = len(subset)
        if (subsetExampleCount == 0):
            continue

        subsetGini = gini(subset, attr_vals_list)
        averageGini += subsetGini * (
            subsetExampleCount / attributeExampleCount)

    return averageGini, division


def chi_squared_test(data, attr_index, attr_vals_list):
    """
    Calculated chi squared and degree of freedom between the selected attribute and the class attribute
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: chi squared value (float), degree of freedom (int)
    """
    pass
