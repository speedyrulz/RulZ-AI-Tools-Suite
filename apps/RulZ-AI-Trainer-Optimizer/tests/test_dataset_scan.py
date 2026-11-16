from rulz_optimizer.dataset_scan import scan


def test_scan(dataset):
    result = scan([dataset])
    assert result.image_count == 1 and result.has_captions
    assert result.trigger_found == 'person' and result.small
