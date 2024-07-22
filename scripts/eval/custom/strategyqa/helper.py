def label_to_index(input):
    map = {
        "true": 1,
        "false": 0,
    }
    answer = input['answer']
    return map[str(answer).lower()]