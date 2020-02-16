def rectify(array,
            bounds=None):
    if bounds is None:
        bounds = {0: (0, 100), 2: (0, 100)}
    for c, (b0, b1) in bounds.items():
        col = array[:, c]
        col[col < b0] = b0
        col[col > b1] = b1

defget_motp()
