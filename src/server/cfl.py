from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering

from src.server.fedavg import FedAvgServer
from src.utils.functional import vectorize


class CFLServer(FedAvgServer):
    algorithm_name: str = "CFL"
    all_model_params_personalized = True  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--eps_1", type=float, default=0.4)
        parser.add_argument("--eps_2", type=float, default=1.6)
        parser.add_argument("--min_cluster_size", type=int, default=2)
        parser.add_argument("--start_clustering_round", type=int, default=20)
        parser.add_argument("--multi_esp", type=int, default=5)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        assert (
            len(self.train_clients) == self.client_num
        ), "CFL doesn't support `User` type split."
        self.split_round = 0
        self.mean_norm=[0]
        self.max_norm=[0]
        self.split_round_list=[]
        self.clients_model_params_diff = [None for _ in self.train_clients]
        self.similarity_matrix = np.eye(len(self.train_clients))
        self.client_clusters = [list(range(len(self.train_clients)))]

    def train_one_round(self):
        client_packages = self.trainer.train()

        for client_id in self.selected_clients:
            self.clients_model_params_diff[client_id] = [
                -diff
                for diff in client_packages[client_id]["model_params_diff"].values()
            ]

        self.compute_pairwise_similarity()
        client_clusters_new = []
        for indices in self.client_clusters:
            max_norm = compute_max_diff_norm(
                [self.clients_model_params_diff[i] for i in indices]
            )
            mean_norm = compute_mean_diff_norm(
                [self.clients_model_params_diff[i] for i in indices]
            )
            self.mean_norm.append(mean_norm)
            self.max_norm.append(max_norm)
            if (
                mean_norm < self.args.cfl.eps_1
                and max_norm > self.args.cfl.eps_2
                and len(indices) > self.args.cfl.min_cluster_size
                and self.current_epoch >= self.args.cfl.start_clustering_round
            ):
            # if (
            #         mean_norm * self.args.cfl.multi_esp < max_norm  # max_norm 比 mean_norm 的4倍还要大
            #
            #     #     mean_norm < self.args.cfl.eps_1
            #     # and max_norm > self.args.cfl.eps_2
            #     and len(indices) > self.args.cfl.min_cluster_size
            #     and self.current_epoch >= self.args.cfl.start_clustering_round
            # ):
                cluster_1, cluster_2 = self.cluster_clients(
                    self.similarity_matrix[indices][:, indices]
                )
                client_clusters_new += [cluster_1, cluster_2]
                self.split_round = self.current_epoch
                self.split_round_list.append(self.current_epoch)
            else:
                client_clusters_new += [indices]

        self.client_clusters = client_clusters_new
        self.aggregate_clusterwise()

    @torch.no_grad()
    def compute_pairwise_similarity(self):
        self.similarity_matrix = np.eye(len(self.train_clients))
        for i, diff_a in enumerate(self.clients_model_params_diff):
            for j, diff_b in enumerate(self.clients_model_params_diff[i + 1 :], i + 1):
                if diff_a is not None and diff_b is not None:
                    score = torch.cosine_similarity(
                        vectorize(diff_a), vectorize(diff_b), dim=0, eps=1e-12
                    ).item()
                    self.similarity_matrix[i, j] = score
                    self.similarity_matrix[j, i] = score

    def cluster_clients(self, similarities):
        clustering = AgglomerativeClustering(
            metric="precomputed", linkage="complete"
        ).fit(-similarities)

        cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
        cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
        return cluster_1, cluster_2

    @torch.no_grad()
    def aggregate_clusterwise(self):
        for cluster in self.client_clusters:
            model_params_diff_list = [
                self.clients_model_params_diff[i]
                for i in cluster
                if self.clients_model_params_diff[i] is not None
            ]
            if (len(model_params_diff_list)) == 0:
                continue
            weights = torch.ones(len(model_params_diff_list)) * (
                1 / len(model_params_diff_list)
            )
            aggregated_diff = [
                torch.sum(torch.stack(diff, dim=-1) * weights, dim=-1)
                for diff in zip(*model_params_diff_list)
            ]
            for i in cluster:
                for key, diff in zip(self.public_model_param_names, aggregated_diff):
                    self.clients_personal_model_params[i][key].data += diff

        self.clients_model_params_diff = [None for _ in self.train_clients]

    def my_log(self):
        self.logger.log(list_to_str(self.mean_norm, "model mean norm:  "))
        self.logger.log(list_to_str(self.max_norm, "model max norm:  "))
        self.logger.log(list_to_str(self.split_round_list, "split-round: "))

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
        self.save_norm_plot()

    def save_norm_plot(self):
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use("Agg")
        linestyle = {
            "before": {"train": "dotted", "val": "dashed", "test": "solid"},
            "after": {"train": "dotted", "val": "dashed", "test": "solid"},
        }
        plt.clf()

        plt.plot(
            self.mean_norm,
            label=f"mean_norm",
            ls="dotted",
        )
        plt.plot(
            self.max_norm,
            label=f"max_norm",
            ls="dashed",
        )

        plt.title(f"The Model Norm")
        # plt.ylim(0, 100)
        plt.xlabel("Communication Rounds")
        plt.ylabel("norm")
        plt.legend()
        plt.savefig(self.output_dir / f"metrics_norm.png", bbox_inches="tight")

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



@torch.no_grad()
def compute_max_diff_norm(model_params_diff: list[list[torch.Tensor]]):
    flag = False
    for diff in model_params_diff:
        if diff is not None:
            flag = True
            break
    if flag:
        return max(
            [
                vectorize(diff).norm().item()
                for diff in model_params_diff
                if diff is not None
            ]
        )
    return 0


@torch.no_grad()
def compute_mean_diff_norm(model_params_diff: list[list[torch.Tensor]]):
    flag = False
    for diff in model_params_diff:
        if diff is not None:
            flag = True
            break
    if flag:
        return (
            torch.stack(
                [vectorize(diff) for diff in model_params_diff if diff is not None]
            )
            .mean(dim=0)
            .norm()
            .item()
        )
    return 0


def list_to_str(list, begin=""):
    return begin+ "[" + ", ".join(map(str, list)) + "]"