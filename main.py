import AI
import pandas as pd
import json
import copy


def fill_dictionary(rule: dict, iteration: int, dictionary: dict, beginning=True):
    for key in rule:
        if isinstance(rule[key], dict):
            fill_dictionary(rule[key], iteration, dictionary[key], False)
        else:
            dictionary[key] = rule[key][iteration if iteration < len(rule[key]) else -1]
    if beginning:
        return dictionary


def get_max_iter(rule: dict) -> int:
    max_iter: int = 0
    for key in rule:
        if isinstance(rule[key], dict):
            temp_num = get_max_iter(rule[key])
            if temp_num > max_iter:
                max_iter = temp_num
        elif len(rule[key]) > max_iter:
            max_iter = len(rule[key])

    return max_iter


def get_string(inp, beginning=True):
    string = []
    for key in inp:
        string.append(key)
        if isinstance(inp[key], dict):
            string[-1] = f"{string[-1]}.{get_string(inp[key], False)}"

    return ','.join(string) if beginning else '.'.join(string)


def Power_Iterate(rules: list[dict], dictionary: dict):
    outputs: dict = {}
    for rule in rules:
        rule_output: list = []
        output_string_name = get_string(rule)
        print(output_string_name)
        for iteration in range(get_max_iter(rule)):
            temp_dictionary = fill_dictionary(rule, iteration, copy.deepcopy(dictionary))
            ai: AI.AI = AI.AI(**temp_dictionary)
            ai.train()
            ai.test()
            rule_output.append(ai.history())
        outputs[output_string_name] = rule_output
    return outputs


def print_dict(dictionary: dict):
    for key in dictionary:
        print(f'{key}: {dictionary[key]}')


def main():
    rules = [
        {
            "number_of_epochs": list(range(1, 3)),
        },
        {
            "number_of_epochs": list(range(1, 3)),
            "sequential_data": {
                'embedding': {
                    'size': list(range(32, 129, 32))
                }
            }
        }
    ]

    output = Power_Iterate(rules, {
        "training_dataset": pd.read_csv("inputs/train.csv"),
        "testing_dataset": pd.read_csv("inputs/test.csv"),

        "number_of_epochs": 5,
        "activation_function": "relu",
        "learning_rate": 0.01,
        "MAX_LENGTH": 20,

        "padding": "post",
        "truncation": "post",

        "sequential_data": {
            "embedding": {
                "size": 32
            },
            "lstm": {
                "size": 64,
                "dropout": 0.2
            },
            "dense": {

            }
        },

        "metrics": ["accuracy"]
    })

    # print_dict()
    with open("output.json", "w") as file:
        json.dump(output, file, indent=2)


if __name__ == "__main__":
    main()
    pass
