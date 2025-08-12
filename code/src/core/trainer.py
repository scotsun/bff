"""Trainer class."""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from core.downstream_models import (
    ForecastingAE,
    BackboneModel,
    SimpleTimeNN,
    DiscreteTimeNN,
    DiscreteFailureTimeNLL,
    MultiModalSNN,
)
from core.bootstrap import CDAUCBootstrapping, AUCBootstrapping
from core.data_utils import MODALITY_DATA_SELECT, SUPPRESS_MODALITY


class EarlyStopping:
    """Early Stopping regularizer."""

    def __init__(self, patience, save_path, min_delta=0, mode="min") -> None:
        self.patience = patience
        self.save_path = save_path
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.best_metric_val = None
        self.early_stop = False

    def _save_checkpoint(self, model):
        torch.save(model, self.save_path)
        return

    def step(self, metric_val, model):
        if self.best_metric_val is None:
            self.best_metric_val = metric_val
            self._save_checkpoint(model)
            return

        match self.mode:
            case "min":
                if metric_val < self.best_metric_val - self.min_delta:
                    self.best_metric_val = metric_val
                    self.counter = 0
                    self._save_checkpoint(model)
                    return
                else:
                    self.counter += 1
                    if self.counter > self.patience:
                        self.early_stop = True
                        return
            case "max":
                if metric_val > self.best_metric_val + self.min_delta:
                    self.best_metric_val = metric_val
                    self.counter = 0
                    self._save_checkpoint(model)
                    return
                else:
                    self.counter += 1
                    if self.counter > self.patience:
                        self.early_stop = True
                        return
            case _:
                raise ValueError("incorrect modes")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
                if self.early_stopping and self.early_stopping.early_stop:
                    break
        print()  # return a blank line

    def _train(self, **kwarg):
        pass

    def _valid(self, **kwarg):
        pass


class DiscreteTimeNNTrainer(Trainer):
    def __init__(
        self,
        model: DiscreteTimeNN,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        criterion: DiscreteFailureTimeNLL,
        bin_boundaries,
        verbose_period: int,
        device: torch.device,
        contrastive_module: None | MultiModalSNN = None,
        modality_selection: str = "final_check",
    ) -> None:
        super().__init__(model, optimizer, early_stopping, verbose_period, device)
        self.criterion = criterion
        self.train_minibatch_loss = []
        self.bin_boundaries = bin_boundaries
        self.train_surv_outcome = None
        self.contrastive_module = contrastive_module
        self.modality_selection = modality_selection

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model: DiscreteTimeNN = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        events, times = [], []
        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in dataloader:
                optimizer.zero_grad()
                event = batch["event"].float().to(device)
                time = batch["time"].float().to(device)

                prediction, z_cs, z_ts, _ = model(
                    inputs=batch["inputs"].to(device),
                    masks=batch["mask"].to(device),
                )

                loss = criterion(prediction, event, time)
                if self.contrastive_module:
                    loss_snn_c, loss_snn_t = self.contrastive_module(
                        z_cs, z_ts, batch["mask"]
                    )
                    # torch.nn.utils.clip_grad_norm_(self.contrastive_module.parameters(), max_norm=5)
                    (loss + 1 * loss_snn_c + 1 * loss_snn_t).backward()
                    bar.set_postfix(
                        loss=float(loss),
                        loss_snn=float(loss_snn_c),
                        loss_snn_t=float(loss_snn_t),
                    )
                else:
                    loss.backward()
                    bar.set_postfix(loss=float(loss))
                optimizer.step()

                self.train_minibatch_loss.append(float(loss))
                bar.update()

                events.append(event)
                times.append(time)
        self.train_surv_outcome = Surv.from_arrays(
            event=torch.cat(events).cpu(), time=torch.cat(times).cpu()
        )

    def _valid(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        bootstrapping: bool = False,
    ):
        model: DiscreteTimeNN = self.model
        device = self.device
        model.eval()

        risk_predictions, events, times = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                event = batch["event"].float().to(device)
                time = batch["time"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)
                # modality selection
                if self.modality_selection:
                    select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                    event, time = event[select], time[select]
                    masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                    inputs = inputs[select]

                prediction, _, _, _ = model(
                    inputs=inputs,
                    masks=masks,
                )
                # note: prediction[:, -1] is the logits for being censored
                events.append(event)
                times.append(time)
                risk_predictions.append(prediction[:, :-1].detach().cpu())
            risk_predictions = torch.cat(risk_predictions).cumsum(dim=1)
            times = torch.cat(times).cpu()
            self.valid_surv_outcome = Surv.from_arrays(
                event=torch.cat(events).cpu(), time=times
            )

            t_max = float(times.max())
            if t_max < self.bin_boundaries[0]:
                raise ValueError("inconsistent value error!")
            auc_t, auc_integrated = cumulative_dynamic_auc(
                self.train_surv_outcome,
                self.valid_surv_outcome,
                risk_predictions[:, 1 : int(t_max // 365 - 1)],
                self.bin_boundaries[
                    2 : int(t_max // 365)
                ],  # eval starting from year 2 since start
            )

            logging = {
                "auc_t": auc_t.tolist(),
                "intg_auc": float(auc_integrated),
            }
            if verbose:
                print("auc_t", auc_t.round(3))
                print("intg_auc", auc_integrated.round(3))

            if self.early_stopping:
                self.early_stopping.step(auc_integrated, model)

            if bootstrapping:
                boots = CDAUCBootstrapping(
                    self.train_surv_outcome,
                    self.valid_surv_outcome,
                    risk_predictions[:, 1 : int(t_max // 365 - 1)],
                    self.bin_boundaries[2 : int(t_max // 365)],
                )
                boots.bootstrap(1000)
                ci_auc_t, ci_intg_auc = boots.auc_ci()
                print(ci_auc_t, ci_intg_auc)
                logging["ci_auc_t"] = ci_auc_t.tolist()
                logging["ci_intg_auc"] = ci_intg_auc.tolist()
            return logging


class BinaryTrainer(Trainer):
    def __init__(
        self,
        model: BackboneModel,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
        contrastive_module: None | MultiModalSNN = None,
        modality_selection: str = "final_check",
    ) -> None:
        super().__init__(model, optimizer, early_stopping, verbose_period, device)
        self.criterion = criterion
        self.train_minibatch_loss = []
        self.contrastive_module = contrastive_module
        self.modality_selection = modality_selection

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model: BackboneModel = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in dataloader:
                y = batch["event"].float().to(device)
                optimizer.zero_grad()
                yh, z_cs, z_ts, _ = model(
                    inputs=batch["inputs"].to(device),
                    masks=batch["mask"].to(device),
                )
                yh = yh.squeeze()

                loss = criterion(yh, y)
                if self.contrastive_module:
                    loss_snn_c, loss_snn_t = self.contrastive_module(
                        z_cs, z_ts, batch["mask"]
                    )
                    (loss + 1 * loss_snn_c + 1 * loss_snn_t).backward()
                    bar.set_postfix(
                        loss=float(loss),
                        loss_snn=float(loss_snn_c),
                        loss_snn_t=float(loss_snn_t),
                    )
                else:
                    loss.backward()
                    bar.set_postfix(loss=float(loss))
                optimizer.step()

                self.train_minibatch_loss.append(float(loss))
                bar.update()

    def _valid(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        bootstrapping: bool = False,
    ):
        model: BackboneModel = self.model
        device = self.device
        model.eval()

        ys, yhs = [], []
        with torch.no_grad():
            for batch in dataloader:
                y = batch["event"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)
                # modality selection
                if self.modality_selection:
                    select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                    y = y[select]
                    masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                    inputs = inputs[select]

                yh, _, _, _ = model(
                    inputs=inputs,
                    masks=masks,
                )

                ys.append(y.cpu())
                yhs.append(yh.detach().cpu())
            ys, yhs = torch.cat(ys), torch.cat(yhs)
            auc = roc_auc_score(ys, yhs)
            pr = average_precision_score(ys, yhs)
            logging = {
                "auc": float(auc),
                "pr": float(pr),
            }
            if verbose:
                print("auc: {:.3f}, pr: {:.3f}".format(auc, pr))
            if self.early_stopping:
                self.early_stopping.step(auc, model)

            if bootstrapping:
                boots = AUCBootstrapping(ys, yhs)
                boots.bootstrap(1000)
                ci_auc, ci_pr = boots.auc_ci()
                print(ci_auc, ci_pr)
                logging["ci_auc"] = ci_auc.tolist()
                logging["ci_pr"] = ci_pr.tolist()
            return logging


class ForecastAEPreTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        early_stopping,
        verbose_period,
        device,
        modality_selection: str = "first_check",
    ):
        super().__init__(model, optimizer, early_stopping, verbose_period, device)
        self.criterion = nn.MSELoss()
        if modality_selection not in ["first_check", "mid_check"]:
            raise ValueError("incorrect modality selection")
        self.modality_selection = modality_selection

    def _train(self, dataloader, verbose, epoch_id):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in dataloader:
                optimizer.zero_grad()
                inputs, masks = batch["inputs"], batch["mask"]
                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]

                x_future_hat, x_future, _ = model(
                    inputs=inputs.to(device),
                    masks=masks.to(device),
                )
                loss = criterion(x_future, x_future_hat)
                loss.backward()
                bar.set_postfix(mse_loss=float(loss))
                optimizer.step()
                bar.update()

    def _valid(self, dataloader, verbose, epoch_id):
        model = self.model
        device = self.device
        model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, masks = batch["inputs"], batch["mask"]
                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]

                x_future_hat, x_future, _ = model(
                    inputs=inputs.to(device),
                    masks=masks.to(device),
                )
                loss = self.criterion(x_future, x_future_hat)
                total_loss += loss.item()
        if verbose:
            print("mse:", total_loss / len(dataloader))


class ForecastDownstreamTrainer(Trainer):
    def __init__(
        self,
        ae: ForecastingAE,
        model: SimpleTimeNN,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        criterion: DiscreteFailureTimeNLL,
        bin_boundaries,
        verbose_period: int,
        device: torch.device,
        modality_selection: str = "first_check",
    ) -> None:
        super().__init__(model, optimizer, early_stopping, verbose_period, device)
        self.criterion = criterion
        self.bin_boundaries = bin_boundaries
        self.train_surv_outcome = None
        self.modality_selection = modality_selection
        self.ae = ae
        for param in self.ae.parameters():
            param.requires_grad = False

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        events, times = [], []
        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in dataloader:
                optimizer.zero_grad()
                event = batch["event"].float().to(device)
                time = batch["time"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)

                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]
                event, time = event[select], time[select]

                _, _, last_h = self.ae(inputs, masks)

                prediction = model(last_h)

                loss = criterion(prediction, event, time)
                loss.backward()

                bar.set_postfix(loss=float(loss))
                optimizer.step()
                bar.update()

                events.append(event)
                times.append(time)
        self.train_surv_outcome = Surv.from_arrays(
            event=torch.cat(events).cpu(), time=torch.cat(times).cpu()
        )

    def _valid(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        bootstrapping: bool = False,
    ):
        model = self.model
        device = self.device
        model.eval()

        risk_predictions, events, times = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                event = batch["event"].float().to(device)
                time = batch["time"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)

                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]
                event, time = event[select], time[select]

                _, _, last_h = self.ae(inputs, masks)
                prediction = model(last_h)

                # note: prediction[:, -1] is the logits for being censored
                events.append(event)
                times.append(time)
                risk_predictions.append(prediction[:, :-1].detach().cpu())
            risk_predictions = torch.cat(risk_predictions).cumsum(dim=1)
            times = torch.cat(times).cpu()
            self.valid_surv_outcome = Surv.from_arrays(
                event=torch.cat(events).cpu(), time=times
            )

            t_max = float(times.max())
            if t_max < self.bin_boundaries[0]:
                raise ValueError("inconsistent value error!")
            auc_t, auc_integrated = cumulative_dynamic_auc(
                self.train_surv_outcome,
                self.valid_surv_outcome,
                risk_predictions[:, 1 : int(t_max // 365 - 1)],
                self.bin_boundaries[
                    2 : int(t_max // 365)
                ],  # eval starting from year 2 since start
            )

            logging = {
                "auc_t": auc_t.tolist(),
                "intg_auc": float(auc_integrated),
            }
            if verbose:
                print("auc_t", auc_t.round(3))
                print("intg_auc", auc_integrated.round(3))

            if self.early_stopping:
                self.early_stopping.step(auc_integrated, model)

            if bootstrapping:
                boots = CDAUCBootstrapping(
                    self.train_surv_outcome,
                    self.valid_surv_outcome,
                    risk_predictions[:, 1 : int(t_max // 365 - 1)],
                    self.bin_boundaries[2 : int(t_max // 365)],
                )
                boots.bootstrap(1000)
                ci_auc_t, ci_intg_auc = boots.auc_ci()
                print(ci_auc_t, ci_intg_auc)
                logging["ci_auc_t"] = ci_auc_t.tolist()
                logging["ci_intg_auc"] = ci_intg_auc.tolist()
            return logging


class ForecastDownstreamBinaryTrainer(Trainer):
    def __init__(
        self,
        ae: ForecastingAE,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        criterion: nn.BCELoss,
        verbose_period: int,
        device: torch.device,
        modality_selection: str = "first_check",
    ) -> None:
        super().__init__(model, optimizer, early_stopping, verbose_period, device)
        self.criterion = criterion
        self.train_surv_outcome = None
        self.modality_selection = modality_selection
        self.ae = ae
        for param in self.ae.parameters():
            param.requires_grad = False

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in dataloader:
                optimizer.zero_grad()
                y = batch["event"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)

                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]
                y = y[select]

                _, _, last_h = self.ae(inputs, masks)
                yh = model(last_h).squeeze()

                loss = criterion(yh, y)
                loss.backward()
                bar.set_postfix(loss=float(loss))
                optimizer.step()

                bar.update()

    def _valid(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        bootstrapping: bool = False,
    ):
        model = self.model
        device = self.device
        model.eval()

        ys, yhs = [], []
        with torch.no_grad():
            for batch in dataloader:
                y = batch["event"].float().to(device)
                masks = batch["mask"].to(device)
                inputs = batch["inputs"].to(device)

                select = MODALITY_DATA_SELECT[self.modality_selection](masks)
                masks = SUPPRESS_MODALITY[self.modality_selection](masks)[select]
                inputs = inputs[select]
                y = y[select]

                _, _, last_h = self.ae(inputs, masks)
                yh = model(last_h)

                ys.append(y.cpu())
                yhs.append(yh.detach().cpu())
            ys, yhs = torch.cat(ys), torch.cat(yhs)
            auc = roc_auc_score(ys, yhs)
            pr = average_precision_score(ys, yhs)
            logging = {
                "auc": float(auc),
                "pr": float(pr),
            }
            if verbose:
                print("auc: {:.3f}, pr: {:.3f}".format(auc, pr))
            if self.early_stopping:
                self.early_stopping.step(auc, model)

            if bootstrapping:
                boots = AUCBootstrapping(ys, yhs)
                boots.bootstrap(1000)
                ci_auc, ci_pr = boots.auc_ci()
                print(ci_auc, ci_pr)
                logging["ci_auc"] = ci_auc.tolist()
                logging["ci_pr"] = ci_pr.tolist()
            return logging
