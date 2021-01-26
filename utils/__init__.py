def none_or_int(value):
    """function used for argparser"""
    if value == 'None':
        return None
    else:
        return value