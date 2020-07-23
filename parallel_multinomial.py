def multinomial_parallel(counts, p, mode='both_diff'):
    """
    Sample from the multinomial distribution with multiple p vectors.
    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.
    The return value has the same shape as p.
    """
    count = counts.copy()
    if mode == 'constant_count':
        count = np.tile(count, p.shape[:-1])
    elif mode == 'constant_prob':
        p = np.tile(p, [count.shape[0], 1])

    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

def test_multinomial_parallel_time(num_trails=5, num_samples=10000000):
    import time, pdb

    t = []
    for i in range(num_trails):
        np.random.seed(2806)
        counts = np.random.randint(10, 100, num_samples)
        probs = np.random.random((num_samples, 5))
        probs = probs/probs.sum(-1, keepdims=True)
        st = time.time()
        samples_loop = np.empty_like(probs)
        for i in range(len(counts)):
            samples_loop[i] = np.random.multinomial(counts[i], probs[i,:])

        assert np.alltrue(samples_loop.sum(1) == counts)
        t.append(time.time()-st)
    print(f'Loop Time: Mean: {np.array(t).mean():.4f} Std: {np.array(t).std():.4f}')

    t = []
    for i in range(num_trails):
        np.random.seed(2806)
        counts = np.random.randint(10, 100, num_samples)
        probs = np.random.random((num_samples, 5))
        probs = probs/probs.sum(-1, keepdims=True)
        st = time.time()
        samples = multinomial_rvs(counts.copy(), probs)
        assert np.array_equal(samples.sum(1), counts)
        t.append(time.time()-st)
    print(f'Parallel Time: Mean: {np.array(t).mean():.4f} Std: {np.array(t).std():.4f}')
    
test_multinomial_parallel_time()
