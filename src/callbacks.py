import numpy as np
from typing import List

from catalyst.core import Callback, CallbackOrder, State
from sklearn.metrics import f1_score, average_precision_score
from catalyst.dl.callbacks.mixup import MixupCallback
from catalyst.dl import CheckpointCallback

class F1Callback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "f1"):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = targ.argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)
        score = f1_score(y_true, y_pred, average="macro")
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


class mAPCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        score = average_precision_score(targ, clipwise_output, average=None)
        score = np.nan_to_num(score).mean()
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


# class MixupCallback(CriterionCallback):

#     def __init__(self,
#                  input_key: str = "targets",
#                  output_key: str = "logits",
#                  fields: List[str] = ("features",),
#                  alpha=1.0,
#                  on_train_only=True):

#         assert isinstance(input_key, str) and isinstance(output_key, str)
#         assert (
#             len(fields) > 0
#         ), "At least one field for MixupCallback is required"
#         assert alpha >= 0, "alpha must be>=0"

#         super().__init__(input_key=input_key, output_key=output_key, **kwargs)

#         self.on_train_only = on_train_only
#         self.fields = fields
#         self.alpha = alpha
#         self.lam = 1
#         self.index = None
#         self.is_needed = True


#     def _compute_loss_value(self, runner: IRunner, criterion):
#         if not self.is_needed:
#             return super()._compute_loss_value(runner, criterion)

#         pred = runner.output[self.output_key]
#         y_a = runner.input[self.input_key]
#         y_b = runner.input[self.input_key][self.index]

#         loss = self.lam * criterion(pred, y_a) + (1 - self.lam) * criterion(
#             pred, y_b
#         )
#         return loss

#     def on_loader_start(self, runner: IRunner):

#         self.is_needed = not self.on_train_only or runner.is_train_loader


#     def on_batch_start(self, runner: IRunner):

#         if not self.is_needed:
#             return

#         if self.alpha > 0:
#             self.lam = np.random.beta(self.alpha, self.alpha)
#         else:
#             self.lam = 1

#         self.index = torch.randperm(runner.input[self.fields[0]].shape[0])
#         self.index.to(runner.device)

#         for f in self.fields:
#             runner.input[f] = (
#                 self.lam * runner.input[f]
#                 + (1 - self.lam) * runner.input[f][self.index]
#             )


def get_callbacks(config: dict):
    required_callbacks = config["callbacks"]
    callbacks = []
    for callback_conf in required_callbacks:
        name = callback_conf["name"]
        params = callback_conf["params"]
        callback_cls = globals().get(name)

        if callback_cls is not None:
            callbacks.append(callback_cls(**params))
    
    callbacks.append(CheckpointCallback(save_n_best=0))
    return callbacks


def send_slack_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def send_slack_error_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'
    data = json.dumps({"text":":no_entry_sign:" + message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)

