import torch
import numpy
import wandb
from tqdm import tqdm

TEAM_NAME = "knife_team"
PROJECT_NAME = "building-segmentation"


class SegmentationModel:
    def __init__(self, model: torch.nn.Module, model_name: str = "model", device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name

    def __call__(self, x):
        return self.model(x)

    def train(self, loss, train_loader, val_loader,
              metrics: dict, optimizer, target_metric="f1",
              epochs=10, wandb_logging=False, path_to_save_model=None,
              verbose=True):

        if wandb_logging:
            run = wandb.init(
                entity=TEAM_NAME,
                project=PROJECT_NAME,
                config={
                    "architecture": self.model_name,
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "optimizer": optimizer.__class__.__name__,
                    "loss": loss.__class__.__name__,
                    "lr": optimizer.param_groups[0]['lr'],
                    "target_metric": target_metric,
                }
            )

        best_metric = 0
        train_logs_list, valid_logs_list = [], []

        self.model.train()

        for epoch in range(1, epochs + 1):
            results = {}
            for metric_name, metric in metrics.items():
                results[metric_name] = []
            results['train_loss'] = []

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            pbar.set_description(f"Epoch {epoch}")

            for batch_idx, (data, target) in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()

                results['train_loss'].append(loss_value.item())
                for metric_name, metric in metrics.items():
                    results[metric_name].append(metric(output, target))

                pbar.set_postfix({
                    "loss": numpy.mean(results['train_loss']),
                })

            results['train_loss'] = numpy.mean(results['train_loss'])
            for metric_name, metric in metrics.items():
                results[metric_name] = numpy.mean(results[metric_name])

            train_logs_list.append(results)
            if verbose:
                print(f'Train logs: {results}')

            results = {}
            for metric_name, metric in metrics.items():
                results[metric_name] = []
            results['val_loss'] = []

            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss_value = loss(output, target)

                    results['val_loss'].append(loss_value.item())
                    for metric_name, metric in metrics.items():
                        results[metric_name].append(metric(output, target))

            results['val_loss'] = numpy.mean(results['val_loss'])
            for metric_name, metric in metrics.items():
                results[metric_name] = numpy.mean(results[metric_name])
            valid_logs_list.append(results)

            if verbose:
                print(f'Val logs: {results}')

            if results[target_metric] > best_metric:
                print(f'New best model! Val {target_metric}: {results[target_metric]}')
                best_metric = results[target_metric]
                if path_to_save_model is not None:
                    torch.save(self.model.state_dict(), path_to_save_model)

            if wandb_logging:
                wandb.log({"train_loss": train_logs_list[-1]['train_loss'],
                           "val_loss": valid_logs_list[-1]['val_loss']})

                for metric_name, metric in metrics.items():
                    wandb.log({f"train_{metric_name}": train_logs_list[-1][metric_name],
                               f"val_{metric_name}": valid_logs_list[-1][metric_name]})

        if wandb_logging:
            run.finish()

        return train_logs_list, valid_logs_list

    def predict(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for batch_idx, data in tqdm(enumerate(test_loader)):
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.detach().cpu().numpy())

        return predictions


