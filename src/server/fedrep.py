from argparse import ArgumentParser, Namespace

from src.client.fedrep import FedRepClient
from src.server.fedavg import FedAvgServer


class FedRepServer(FedAvgServer):
    algorithm_name: str = "FedRep"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedRepClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--train_body_epoch", type=int, default=1)
        return parser.parse_args(args_list)

    def save_learning_curve_plot(self):
        """Save the learning curves of FL-bench experiment."""
        import matplotlib
        from matplotlib import pyplot as plt

        matplotlib.use("Agg")
        linestyle = {
            "before": {"train": "dotted", "val": "dashed", "test": "solid"},
            "after": {"train": "dotted", "val": "dashed", "test": "solid"},
        }
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.aggregated_client_metrics[stage][split]) > 0:
                    plt.plot(
                        [
                            metrics.accuracy
                            for metrics in self.aggregated_client_metrics[stage][split]
                        ],
                        label=f"{split}set ({stage}LocalTraining)",
                        ls=linestyle[stage][split],
                    )

        plt.title(f"{self.algorithm_name}_{self.args.dataset.name}")
        plt.ylim(0, 100)
        plt.xlabel("Communication Rounds")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(self.output_dir / f"metrics_acc.png", bbox_inches="tight")

        self.save_learning_curloss_plot()

    def save_learning_curloss_plot(self):
        """Save the learning curves of FL-bench experiment."""
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use("Agg")
        linestyle = {
            "before": {"train": "dotted", "val": "dashed", "test": "solid"},
            "after": {"train": "dotted", "val": "dashed", "test": "solid"},
        }
        plt.clf()
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.aggregated_client_metrics[stage][split]) > 0:
                    plt.plot(
                        [
                            metrics.loss
                            for metrics in self.aggregated_client_metrics[stage][split]
                        ],
                        label=f"{split}set ({stage}LocalTraining)",
                        ls=linestyle[stage][split],
                    )

        plt.title(f"{self.algorithm_name}_{self.args.dataset.name}")
        # plt.ylim(0, 100)
        plt.xlabel("Communication Rounds")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.output_dir / f"metrics_loss.png", bbox_inches="tight")

