from itertools import product

class ProbabilitySpace:
    def __init__(self, sample_space: set, probability_function: dict[tuple, float]):
        self._sample_space = sample_space
        self._probability_function = probability_function
        self._verify_probability_function()

    def _verify_probability_function(self):
        if any(prob < 0 for prob in self._probability_function.values()):
            raise ValueError("Probability values must be non-negative.")
        total_probability = sum(self._probability_function.values())
        if not round(total_probability, 10) == 1:
            raise ValueError("Total probability of all elementary outcomes must sum to 1.")
        if set(self._probability_function.keys()) != {tuple([s]) for s in self._sample_space}:
            raise ValueError("Probability function does not cover the entire sample space or includes undefined events.")

    def P(self, event: set) -> float:
        event_tuples = [tuple([e]) for e in event]
        return sum(self._probability_function.get(e, 0) for e in event_tuples)

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def event_space(self):
        # For large sample spaces, avoid generating the full event space
        if len(self._sample_space) > 10:
            return "Event space too large to display."
        return [set(s) for s in product(self._sample_space, repeat=2)]

    @property
    def probability_function(self):
        return self._probability_function

    def __str__(self):
        sample_space_str = f"Sample Space: {self._sample_space}"
        prob_func_str = f"Probability Function: {self._probability_function}"
        # Display a warning for large event spaces instead of the actual space
        if len(self._sample_space) > 10:
            event_space_str = f"Event Space: Too large to display (2^{len(self._sample_space)} elements)."
        else: 
            unique_frozensets = {frozenset(s) for s in list(self.event_space)}
            unique_list = [set(s) for s in unique_frozensets]
            event_space_str = f"Event Space: {unique_list}"
        return f"{sample_space_str}\n{event_space_str}\n{prob_func_str}"

def get_new_probability_space(original_space: ProbabilitySpace, random_variable: dict) -> ProbabilitySpace:
    new_probabilities = {}
    for original_event, probability in original_space.probability_function.items():
        new_event_value = sum(random_variable.get(e, 0) for e in original_event)
        new_event = (new_event_value,)
        new_probabilities[new_event] = new_probabilities.get(new_event, 0) + probability

    new_sample_space = {e[0] for e in new_probabilities.keys()}
    return ProbabilitySpace(new_sample_space, new_probabilities)


# Create the sample space and probability function for two six-sided dice
sample_space = {(i, j) for i in range(1, 7) for j in range(1, 7)}
probability_function = {(outcome,): 1/36 for outcome in sample_space}

dice_space = ProbabilitySpace(sample_space, probability_function)

print(dice_space)

# Define a random variable X that maps each ordered pair to its sum
X = {(i, j): i + j for i in range(1, 7) for j in range(1, 7)}

new_probability_space = get_new_probability_space(dice_space, X)

# Compute the probability P({7, 11})
prob_sum_7_or_11 = new_probability_space.P({7, 11})
print(f"P(Sum of 7 or 11): {prob_sum_7_or_11}")

sample_space = {'H', 'T'}
probability_function = {
    ('H',): 0.5,
    ('T',): 0.5,
}

coin_flip_space = ProbabilitySpace(sample_space, probability_function)

print(coin_flip_space)

print("\nP(Heads):", coin_flip_space.P({'H'}))
print("P(Tails):", coin_flip_space.P({'T'}))
print("P(Heads or Tails):", coin_flip_space.P({'H', 'T'}))

X = {'H': 1, 'T': 0}

new_probability_space = get_new_probability_space(coin_flip_space, X)
