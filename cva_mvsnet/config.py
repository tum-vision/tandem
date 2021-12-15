import yaml
from ast import literal_eval
from os.path import dirname, join

DEFAULT_CONFIG_FILE = join(dirname(__file__), 'configs', 'default.yaml')


def _parse_dict(d, d_out=None, prefix=""):
    d_out = d_out if d_out is not None else {}
    for k, v in d.items():
        if isinstance(v, dict):
            _parse_dict(v, d_out, prefix=prefix + k + '.')
        else:
            if isinstance(v, str):
                try:
                    v = literal_eval(v)  # try to parse
                except (ValueError, SyntaxError):
                    pass  # v is really a string

            if isinstance(v, list):
                v = tuple(v)
            d_out[prefix + k] = v
    if prefix == "":
        return d_out


def load(fname):
    with open(fname, 'r') as fp:
        return _parse_dict(yaml.safe_load(fp))


def merge_from_config(config, config_merge):
    for k, v in config_merge.items():
        assert k in config, f"The key {k} is not in the base config for the merge."
        config[k] = v


def merge_from_file(config, fname):
    merge_from_config(config, load(fname))


def merge_from_list(config, list_merge):
    assert len(list_merge) % 2 == 0, "The list must have key value pairs."
    config_merge = _parse_dict(dict(zip(list_merge[0::2], list_merge[1::2])))
    merge_from_config(config, config_merge)


def default():
    return load(DEFAULT_CONFIG_FILE)