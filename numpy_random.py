def vectorised_random_choice(probs, axis=-1):
    '''
    Vectorises the numpy np.random.choice function.
    
    probs: prpbability matrix with last dimension summing to 1
    return: numbers of size probs.shape[:-1] within the range(0, probs.shape[-1]
    '''
    r = np.expand_dims(np.random.rand(*probs.shape[:axis]), axis=axis)
    return (probs.cumsum(axis=axis) > r).argmax(axis=axis)

def test_vectorised_random_choice(d1, d2, categorical_dist_size=10, num_samples=1000):
    '''Test case for vectorised choice'''
    probs = np.random.rand(d1, d2, categorical_dist_size)
    probs /= probs.sum(-1, keepdims=True)

    binc = np.zeros_like(probs)
    for i1 in range(0, d1):
        for i2 in range(0, d2):
            choices = [random_choice_prob_index(probs)[i1,i2] for i in range(num_samples)]
            binc[i1,i2] = np.bincount(choices, minlength=categorical_dist_size)/float(len(choices))
    print(f'Norm error in predicted probs vs true probs: {np.linalg.norm(binc - probs):.4f}/{d1*d2}')
