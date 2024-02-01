from pydoc import locate


def build_cfg(cfg):
    obj = locate(cfg["functional"])
    args = [] if cfg["args"] is None else cfg["args"]
    kwargs = {} if cfg["kwargs"] is None else cfg["kwargs"]
    return obj(*args, **kwargs)
