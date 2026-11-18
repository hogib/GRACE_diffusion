

def test_parse_inflate():
    from grace_diffusion.evaluate import parse_inflate
    assert parse_inflate("1.4") == 1.4
    assert parse_inflate("16:1.3,32:1.5") == {16: 1.3, 32: 1.5}
