def handle_logprobs_error(func):

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError) as e:
            if 'logprobs' in str(e):
                raise ValueError('This API model does not provide logprobs and cannot be used with truth methods that require logprobs. Please try a different API model or different truth method.')
            raise e
    return wrapper
