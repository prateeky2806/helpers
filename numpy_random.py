def random_choice(probs, axis=-1):
    '''
    Vectorises the numpy np.random.choice function.
    
    probs: prpbability matrix with last dimension summing to 1
    return: numbers of size probs.shape[:-1] within the range(0, probs.shape[-1]
    '''
    r = np.expand_dims(np.random.rand(*probs.shape[:axis]), axis=axis)
    return (probs.cumsum(axis=axis) > r).argmax(axis=axis)
