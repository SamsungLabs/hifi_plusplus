import collections
import inspect
import omegaconf
import dataclasses
import typing


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


def move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        x = x.__class__(move_to_device(t, device) for t in x)
    else:
        x = x.to(device)
    return x


def accuracy(real_logits=None, fake_logits=None):
    if real_logits is None and fake_logits is None:
        raise ValueError("at least one of the logits should be not None")

    real_acc = (
        real_logits.ge(0).float().mean() if real_logits is not None else 0.0
    )
    fake_acc = (
        fake_logits.le(0).float().mean() if fake_logits is not None else 0.0
    )
    if real_logits is None or fake_logits is None:
        return real_acc + fake_acc
    return (real_acc + fake_acc) / 2


class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, arg_keys):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [
            arg
            for arg in args
            if (arg[0] != "self" and arg[0] != "args" and arg[0] != "kwargs")
        ]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.classes.items()
            ],
        )

    def make_dataclass_from_args(self, name):
        return dataclasses.make_dataclass(
            name,
            [
                (k, v, dataclasses.field(default=v()))
                for k, v in self.args.items()
            ],
        )

    def add_to_registry(self, name, arg_keys=None):
        def add_class_by_name(cls):
            self.classes[name] = cls
            self.args[name] = self.make_dataclass_from_init(
                cls.__init__, name, arg_keys
            )
            return cls

        return add_class_by_name
