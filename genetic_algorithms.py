import random
import numpy as np
import copy
import itertools

def fitness_proportionate_selection(population, fitness, k):
    indices = fitness_proportionate_selection_index(fitness, k)
    individuals = [population[i] for i in indices]
    return individuals

def fitness_proportionate_selection_index(fitness, k):
    fitness = np.array(fitness)
    fitness = fitness / np.sum(fitness)
    return np.random.choice(np.arange(len(fitness)), size=k, replace=False, p=fitness)

def tournament_selection(population, fitness, k, tournament_size):
    indices = tournament_selection_index(fitness, k, tournament_size)
    individuals = [population[i] for i in indices]
    return individuals

def tournament_selection_index(fitness, k, tournament_size):
    pf = list(zip(range(len(fitness)), fitness))
    indices = []
    for _ in range(k):
        sample = random.sample(pf, tournament_size)
        best = max(sample, key=lambda x: x[1])
        indices.append(best[0])
        pf.remove(best)
    return indices

def stochastic_universal_sampling(population, fitness, k):
    indices = stochastic_universal_sampling_index(fitness, k)
    individuals = [population[i] for i in indices]
    return individuals

# def _stochastic_universal_sampling(population, fitness, k):
#     fitness_sum = sum(fitness)
#     interval = fitness_sum / k
#     pointer = interval * random.random()

#     indices = set([])
#     fitness_accumulate = fitness[0]
#     index = 0
#     for _ in range(k):
#         while fitness_accumulate < pointer:
#             index += 1
#             fitness_accumulate += fitness[index]
        
#         indices.add(index)
#         pointer += interval

#     individuals = [population[i] for i in indices]

#     if len(indices) < k:
#         for i in sorted(indices, reverse=True):
#             population.pop(i)
#             fitness.pop(i)
#         individuals += _stochastic_universal_sampling(population, fitness, k - len(indices))

#     return individuals

def stochastic_universal_sampling_index(fitness, k):
    fitness = copy.deepcopy(fitness)
    return _stochastic_universal_sampling_index(fitness, k)

def _stochastic_universal_sampling_index(fitness, k):
    fitness_sum = sum(fitness)
    interval = fitness_sum / k
    pointer = interval * random.random()

    indices = set([])
    fitness_accumulate = fitness[0]
    index = 0
    for _ in range(k):
        while fitness_accumulate < pointer:
            index += 1
            fitness_accumulate += fitness[index]
        
        indices.add(index)
        pointer += interval

    if len(indices) < k:
        for i in sorted(indices, reverse=True):
            fitness.pop(i)
        indices.update(_stochastic_universal_sampling_index(fitness, k - len(indices)))

    return indices


def kpoint_crossover(parentA, parentB, k):
    parent_length = len(parentA)
    points = random.sample(range(1, parent_length - 1), k)
    points.sort()
    child1, child2 = [], []
    prev_point = 0
    for point in points + [parent_length]:
        child1 += parentA[prev_point:point]
        child2 += parentB[prev_point:point]
        child1, child2 = child2, child1
        prev_point = point
    return child1, child2

def uniform_crossover(parentA, parentB, weightA, weightB):
    x = weightA / (weightA + weightB)
    child = [bitA if random.random() < x else bitB for bitA, bitB in zip(parentA, parentB)]
    return child

def partially_matched_crossover(parentA, parentB, k):
    parent_length = len(parentA)
    points = random.sample(range(1, parent_length - 1), k)
    points.sort()
    points = points + [parent_length]
    childA1, childA2, childB1, childB2 = [], [], [], []

    # Create Set for indicies of each segment group
    segments_indices = (set([]), set([]))
    prev_point = 0
    segments_no = 0
    for point in points:
        segments_indices[segments_no].update(range(prev_point, point))
        segments_no = 1 - segments_no
        prev_point = point

    # MAP element of Parent A to same element of Parent B and vice versa
    parentB_dict = {parentB[i] : i for i in range(parent_length)}
    mapAB = []
    mapBA = [None] * parent_length
    for i in range(parent_length):
        value = parentA[i]
        index = parentB_dict[value]
        mapAB.append(index)
        mapBA[index] = i
    
    prev_point = 0
    segment_no = 1
    for point in points:
        childA1 += parentA[prev_point:point]
        childB1 += parentB[prev_point:point]
        
        segment_indices = segments_indices[segment_no]
        for i in range(prev_point, point):
            indexA = mapBA[i]
            indexB = mapAB[i]
            while indexA in segment_indices:
                indexA = mapBA[indexA]

            while indexB in segment_indices:
                indexB = mapAB[indexB]

            childA2.append(parentA[indexA])
            childB2.append(parentB[indexB])

        childA1, childA2 = childA2, childA1
        childB1, childB2 = childB2, childB1
        segment_no = 1 - segment_no
        prev_point = point

    return childA1, childA2, childB1, childB2

def cycle_crossover(parentA, parentB, k):
    parent_length = len(parentA)

    # MAP element of Parent A to same element of Parent B and (not) vice versa
    parentB_dict = {parentB[i] : i for i in range(parent_length)}
    mapAB = []
    for i in range(parent_length):
        value = parentA[i]
        index = parentB_dict[value]
        mapAB.append(index)

    grouped = set([])
    groups = []
    
    for i in range(parent_length):
        if i not in grouped:
            start_index = i
            i = mapAB[i]

            if i != start_index:
                group = [start_index]
                while i != start_index:
                    group.append(i)
                    grouped.add(i)
                    i = mapAB[i]

                groups.append(group)
    
    # Select Groups
    groups_length = len(groups)
    if groups_length < k or k <= 0:  k = groups_length

    selected_groups = random.sample(groups, k)
    binary_product = itertools.product((True, False), repeat=k)
    binary_product = itertools.islice(binary_product, 1, 2 ** k - 1)

    children = []
    parentA = np.array(parentA)
    parentB = np.array(parentB)
    for k_binary in binary_product:
        childA = np.copy(parentA)
        for binary, group in zip(k_binary, selected_groups):
            if binary:
                childA[group] = parentB[group]
    
        children.append(childA)

    return children

def order_1_crossover(parentA, parentB, k):
    parent_length = len(parentA)
    points = random.sample(range(1, parent_length - 1), k)
    points.sort()
    points += [parent_length]
    # points = POINTS
    childA1, childA2, childB1, childB2 = [], [], [], []

    # Exclude elements from parents to respective child
    childA_excluded = (set([]), set([]))
    childB_excluded = (set([]), set([]))
    childA1_excluded, childA2_excluded = childA_excluded
    childB1_excluded, childB2_excluded = childB_excluded
    prev_point = 0
    for point in points:
        for i in range(prev_point, point):
            childA1_excluded.add(parentA[i])
            childB1_excluded.add(parentB[i])
        childA1_excluded, childA2_excluded = childA2_excluded, childA1_excluded
        childB1_excluded, childB2_excluded = childB2_excluded, childB1_excluded

        prev_point = point

    childA1_excluded, childA2_excluded = childA_excluded
    childB1_excluded, childB2_excluded = childB_excluded

    # Optimisation?
    childA1_pointer, childA2_pointer = 0, 0
    childB1_pointer, childB2_pointer = 0, 0

    prev_point = 0
    for point in points:
        childA1 += parentA[prev_point:point]
        childB1 += parentB[prev_point:point]
        for i in range(prev_point, point):
            # ChildA2
            parentB_element = parentB[childA2_pointer]
            childA2_pointer += 1
            while parentB_element in childA2_excluded:
                parentB_element = parentB[childA2_pointer]
                childA2_pointer += 1
            
            childA2.append(parentB_element)

            # ChildB2
            parentA_element = parentA[childB2_pointer]
            childB2_pointer += 1
            while parentA_element in childB2_excluded:
                parentA_element = parentA[childB2_pointer]
                childB2_pointer += 1

            childB2.append(parentA_element)

        childA1, childA2 = childA2, childA1
        childB1, childB2 = childB2, childB1
        childA1_pointer, childA2_pointer = childA2_pointer, childA1_pointer
        childB1_pointer, childB2_pointer = childB2_pointer, childB1_pointer
        childA1_excluded, childA2_excluded = childA2_excluded, childA1_excluded
        childB1_excluded, childB2_excluded = childB2_excluded, childB1_excluded

        prev_point = point

    return childA1, childA2, childB1, childB2

def edge_recombination_operator(parentA, parentB):
    parent_length = len(parentA)

    neighbours = {parentA[i]: set() for i in range(parent_length)}
    for i in range(-1, parent_length - 1): # Equivalent to reversed(range(parent_length))
        front, back = i + 1, i - 1
        nodeA = neighbours[parentA[i]]
        nodeB = neighbours[parentB[i]]
        nodeA.add(parentA[front])
        nodeA.add(parentA[back])
        nodeB.add(parentB[front])
        nodeB.add(parentB[back])

    #selected_node = random.choice(parentA)
    if random.random() < 0.5:
        selected_node = parentA[0]
    else:
        selected_node = parentB[0]
    
    child = [selected_node]

    for _ in range(parent_length - 1):
        selected_node_neighbours = neighbours[selected_node]
        del neighbours[selected_node]

        if selected_node_neighbours:
            smallest_size = 5
            smallest_nodes = []
            for node in selected_node_neighbours:
                node_neighbours = neighbours[node]
                node_neighbours.discard(selected_node)
                node_length = len(node_neighbours)
                if node_length < smallest_size:
                    smallest_size = node_length
                    smallest_nodes = [node]
                elif node_length == smallest_size:
                    smallest_nodes.append(node)

            selected_node = random.choice(smallest_nodes)
        else:
            selected_node = random.choice(list(neighbours))

        child.append(selected_node)

    return [child]


def swap_mutation(child, k):
    indices = random.sample(range(len(child)), 2 * k) # Distinct Swap (Elements only swap once)
    for i,j in zip(indices[:-1:2], indices[1::2]):
        child[i], child[j] = child[j], child[i]

def scramble_mutation(child, k):
    indices = random.sample(range(len(child) + 1), 2 * k) # Distinct Swap (Elements only swap once)
    indices.sort()
    for i,j in zip(indices[:-1:2], indices[1::2]):
        segment = child[i:j]
        random.shuffle(segment)
        child[i:j] = segment

def cycle_scramble_mutation(child, k):
    indices = random.sample(range(len(child)), 2 * k) # Distinct Swap (Elements only swap once)
    indices.sort()
    if random.random() < 0.5:
        a,b = indices[-1], indices[0] # Didn't pop because optimisation
        segment = child[a:] + child[:b]
        random.shuffle(segment)
        c = len(segment) - b
        child[a:] = segment[:c]
        child[:b] = segment[c:]

        for i,j in zip(indices[1:-2:2], indices[2:-1:2]): # For Optimisation
            segment = child[i:j]
            random.shuffle(segment)
            child[i:j] = segment
    else:
        for i,j in zip(indices[:-1:2], indices[1::2]):
            segment = child[i:j]
            random.shuffle(segment)
            child[i:j] = segment

def inversion_mutation(child, k):
    indices = random.sample(range(len(child) + 1), 2 * k) # Distinct Swap (Elements only swap once)
    indices.sort()
    child_copy = [None] + child
    for i,j in zip(indices[:-1:2], indices[1::2]):
        child[i:j] = child_copy[j:i:-1]

def cycle_inversion_mutation(child, k):
    indices = random.sample(range(len(child)), 2 * k) # Distinct Swap (Elements only swap once)
    indices.sort()
    if random.random() < 0.5:
        a,b = indices[-1], indices[0] # Didn't pop because optimisation
        segment = child[a:] + child[:b]
        segment.reverse()
        c = len(segment) - b
        child[a:] = segment[:c]
        child[:b] = segment[c:]

        child_copy = [None] + child
        for i,j in zip(indices[1:-2:2], indices[2:-1:2]): # For Optimisation
            child[i:j] = child_copy[j:i:-1]

    else:
        child_copy = [None] + child
        for i,j in zip(indices[:-1:2], indices[1::2]):
            child[i:j] = child_copy[j:i:-1]

def find_ksmallest_indices(a, k):
    return np.argpartition(a, k)[:k]

def find_klargest_indices(a, k):
    return np.argpartition(a, -k)[-k:]