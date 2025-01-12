import torch
from algorithm.cyy_torch_algorithm.computation.sample_gvjp.sample_gvjp_hook import \
    SampleGradientVJPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException
from torch import nn


def test_CV_vjp():
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleGradientVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "check gradients", print_result
    )
    trainer.train()
    hook.reset()


def test_NLP_vjp():
    return
    config = DefaultConfig("IMDB", "TransformerClassificationModel")
    config.dc_config.dataset_kwargs["max_len"] = 300
    config.model_config.model_kwargs["max_len"] = 300
    config.model_config.model_kwargs["d_model"] = 100
    config.model_config.model_kwargs["nhead"] = 5
    config.model_config.model_kwargs["num_encoder_layer"] = 1
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.1
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    trainer.model_util.freeze_modules(module_type=nn.Embedding)
    trainer.model_evaluator.need_input_features = True
    hook = SampleGradientVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_result
    )
    trainer.train()
