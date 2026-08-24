from qrs.train.callbacks import EarlyStopping


def test_stops_after_patience_epochs_without_improvement():
    es = EarlyStopping(patience=2, mode="min")
    values = [1.0, 0.9, 0.95, 0.96, 0.97]  # improves, then plateaus
    stopped_at = None
    for i, v in enumerate(values):
        if es.step(v):
            stopped_at = i
            break
    assert stopped_at == 3  # two epochs (idx 2, 3) without improvement after best at idx 1


def test_tracks_best_value():
    es = EarlyStopping(patience=5, mode="min")
    for v in [1.0, 0.5, 0.8]:
        es.step(v)
    assert es.best == 0.5
