def _info(s):
    print('---')
    print(s)
    print('---')


def _to_cpu(a, clone=True):
    '''
    cuda to cpu for numpy operations
    '''
    if clone:
        a = a.clone()
    a = a.detach().cpu()

    return a
