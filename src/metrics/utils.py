# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_dist(target_text, predicted_text) -> float:
    len_a = len(target_text)
    len_b = len(predicted_text)
    d = [[0 for _ in range(len_b + 1)] for _ in range(len_a + 1)]

    for i in range(1, len_a + 1):
        d[i][0] = d[i - 1][0] + 1

    for i in range(1, len_b + 1):
        d[0][i] = d[0][i - 1] + 1

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            d[i][j] = min(
                d[i - 1][j - 1]
                + (1 if target_text[i - 1] != predicted_text[j - 1] else 0),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )

    return d[-1][-1] / len(target_text)


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        if len(predicted_text) == 0:
            return 0
        return 1

    return calc_dist(target_text, predicted_text)


def calc_wer(target_text, predicted_text) -> float:
    splited_target = target_text.split()
    splited_predicted = predicted_text.split()
    if len(splited_target) == 0:
        if len(splited_predicted) == 0:
            return 0
        return 1

    return calc_dist(splited_target, splited_predicted)
