"""Match sales prefixes to completed historical continuations in the same shop."""
import numpy as np

HORIZON = 13
MAX_AGE = 13
MIN_DONORS = 15
LOOKBACK = 13


def peer_forecast(history, champion):
    """history contains only weeks before the origin, with one row per distinct product."""
    sales = np.asarray(history, dtype=float)
    prediction = np.asarray(champion, dtype=float).copy()
    n_products, origin = sales.shape
    if prediction.shape != (n_products, HORIZON):
        raise ValueError('Expected one 13-week forecast per product')
    positive = sales > 0
    first = np.where(positive.any(axis=1), positive.argmax(axis=1), origin)
    ages = origin - first
    records = []
    for target in np.flatnonzero((ages >= 1) & (ages <= MAX_AGE)):
        age = ages[target]
        prefix = sales[target, first[target]:]
        # Exclusive end equal to origin is valid: the last label is week origin-1.
        donors = np.flatnonzero((first >= LOOKBACK) & (first + age + HORIZON <= origin))
        prefixes = sales[donors[:, None], first[donors, None] + np.arange(age)]
        volumes = prefixes.sum(axis=1)
        target_volume = prefix.sum()
        ratio = (1 + volumes) / (1 + target_volume)
        eligible = (ratio >= .5) & (ratio <= 2)
        donors = donors[eligible]
        selected = np.array([], dtype=int)
        if len(donors) >= MIN_DONORS:
            prefixes = prefixes[eligible]
            distance = np.abs(np.log2(ratio[eligible]))
            distance += np.abs((prefixes > 0).mean(axis=1) - (prefix > 0).mean())
            shares = prefixes / volumes[eligible, None]
            distance += .5 * np.abs(shares - prefix / target_volume).sum(axis=1)
            boundary = np.partition(distance, MIN_DONORS - 1)[MIN_DONORS - 1]
            selected = donors[distance <= boundary + 1e-12]
            followup = first[selected, None] + age + np.arange(HORIZON)
            assert followup.max() < origin and target not in selected
            prediction[target] = np.rint(np.median(sales[selected[:, None], followup], axis=0))
        records.append(dict(target=int(target), age=int(age), eligible=len(donors), used=len(selected)))
    return prediction, records
