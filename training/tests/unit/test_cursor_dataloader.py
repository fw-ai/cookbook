from training.utils.dataloader import CursorDataLoader


def test_cursor_starts_from_resume_point():
    loader = CursorDataLoader(["a", "b", "c"], start_cursor=1)

    assert next(loader).value == "b"
    assert loader.data_consumed == 1


def test_cursor_advances_in_order_only():
    loader = CursorDataLoader(["a", "b", "c"])

    loader.mark_resolved(1)
    assert loader.data_consumed == 0

    loader.mark_resolved(0)
    assert loader.data_consumed == 2

    loader.mark_resolved(2)
    assert loader.data_consumed == 3


def test_cursor_ignores_already_resolved_indices():
    loader = CursorDataLoader(["a", "b"], start_cursor=1)

    loader.mark_resolved(0)
    assert loader.data_consumed == 1


def test_cursor_can_resume_past_available_rows():
    loader = CursorDataLoader(["a"], start_cursor=3)

    assert loader.data_consumed == 3
    assert list(loader) == []


def test_cursor_owns_epochs_without_reusing_row_objects():
    loader = CursorDataLoader([{"id": 1}], epochs=2)

    first = next(loader)
    first.value["seen"] = True
    second = next(loader)

    assert first.index == 0
    assert second.index == 1
    assert second.value == {"id": 1}


def test_cursor_shuffle_is_deterministic_per_epoch():
    rows = list(range(10))
    first = [item.value for item in CursorDataLoader(rows, epochs=2, shuffle=True, seed=7)]
    second = [item.value for item in CursorDataLoader(rows, epochs=2, shuffle=True, seed=7)]
    unshuffled = [item.value for item in CursorDataLoader(rows, epochs=2)]

    assert first == second
    assert first != unshuffled
    assert sorted(first[:10]) == rows
    assert sorted(first[10:]) == rows


def test_cursor_resume_reconstructs_epoch_shuffle_position():
    rows = list(range(6))
    full = [item.value for item in CursorDataLoader(rows, epochs=3, shuffle=True, seed=4)]
    resumed = [
        item.value
        for item in CursorDataLoader(rows, start_cursor=8, epochs=3, shuffle=True, seed=4)
    ]

    assert resumed == full[8:]
