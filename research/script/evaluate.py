def get_report(answers, preds, clicks):
    """
    Calculate various metrics including Exact Match Rate and CTR.

    Parameters:
    answers: List of correct answer sets
    preds: List of predicted answer sets
    clicks: List of binary values indicating if the item was clicked (1) or not (0)

    Returns:
    str: Report containing Exact Match Rate, Expected CTR, and True CTR
    """
    exact_matched_indices = [
        idx
        for idx, (pred, ans) in enumerate(zip(preds, answers))
        if set(pred) == set(ans)
    ]
    exact_match = len(exact_matched_indices) / len(answers)

    match_scores = [
        (len(set(pred).intersection(ans)) / len(ans))
        for pred, ans in zip(preds, answers)
    ]

    matched_clicks = [score * clicks[idx] for idx, score in enumerate(match_scores)]
    expected_ctr_score = sum(matched_clicks) / len(clicks)

    true_ctr_score = sum(clicks) / len(clicks)

    report = (
        f"Exact matching: {exact_match:.3f}\n"
        f"Expected CTR: {expected_ctr_score:.3f}\n"
        f"True CTR: {true_ctr_score:.3f}\n"
    )
    return report
