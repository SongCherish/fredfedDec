from argparse import ArgumentParser, Namespace
import numpy as np
from sympy import false

from src.client.fredfedrep import FredFedRepClient
from src.server.fedavg import FedAvgServer
import torch
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering

from src.server.fedavg import FedAvgServer
from src.utils.functional import vectorize, vectorizeAndDct
import functools
import inspect
import json
import os
import pickle
import random
import shutil
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import pprint as rich_pprint
from rich.progress import track
from torchvision import transforms

from data.utils.datasets import DATASETS, BaseDataset
from src.client.fedavg import FedAvgClient
from src.utils.constants import (
    DATA_MEAN,
    DATA_STD,
    FLBENCH_ROOT,
    LR_SCHEDULERS,
    MODE,
    OPTIMIZERS,
)
from src.utils.functional import (
    evaluate_model,
    fix_random_seed,
    get_optimal_cuda_device,
    initialize_data_loaders,
)
from src.utils.logger import Logger
from src.utils.metrics import Metrics
from src.utils.models import MODELS, DecoupledModel
from src.utils.trainer import FLbenchTrainer
from sklearn.cluster import AffinityPropagation

class FredFedRepServer(FedAvgServer):
    algorithm_name: str = "FredFedRep"
    all_model_params_personalized = True  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FredFedRepClient


    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--train_body_epoch", type=int, default=1)

        parser.add_argument("--eps_1", type=float, default=0.4)
        parser.add_argument("--eps_2", type=float, default=1.6)
        parser.add_argument("--min_cluster_size", type=int, default=2)
        parser.add_argument("--start_clustering_round", type=int, default=20)
        parser.add_argument("--cluster_count", type=int, default=2)
        parser.add_argument("--multi_esp", type=int, default=4)
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
        temp_client_clusters = [list(range(len(self.train_clients)))]#每轮都重新分簇
        for indices in temp_client_clusters:
            max_norm = compute_max_diff_norm(
                [self.clients_model_params_diff[i] for i in indices]
            )
            mean_norm = compute_mean_diff_norm(
                [self.clients_model_params_diff[i] for i in indices]
            )
            self.mean_norm.append(mean_norm)
            self.max_norm.append(max_norm)

            if (
                    # mean_norm * self.args.fredfedrep.multi_esp < max_norm    #max_norm 比 mean_norm 的4倍还要大

                    # mean_norm < self.args.fredfedrep.eps_1
                # and max_norm > self.args.fredfedrep.eps_2
                len(indices) > self.args.fredfedrep.min_cluster_size    #簇内客户端数量比 规定的阈值要大

                and
                    self.current_epoch >= self.args.fredfedrep.start_clustering_round   #训练一段时间后再开始分簇
                # and    self.current_epoch % self.args.fredfedrep.start_clustering_round == 0
                # and self.current_epoch // self.args.fredfedrep.start_clustering_round <= self.args.fredfedrep.cluster_count
            ):
                self.split_round = self.current_epoch
                self.split_round_list.append(self.current_epoch)
                # cluster_1, cluster_2 = self.cluster_clients(
                #     self.similarity_matrix[indices][:, indices]
                # )
                # client_clusters_new += [cluster_1, cluster_2]
                clusters = self.cluster_clients(
                    self.similarity_matrix[indices][:, indices]
                )
                client_clusters_new += clusters

            else:
                client_clusters_new += [indices]

        self.client_clusters = client_clusters_new
        self.logger.log("-------------------------------------------------------")
        self.logger.log(f"簇的数量： {client_clusters_new.__len__()}")
        for cluster in client_clusters_new:
            self.logger.log(f"簇的大小： {cluster.__len__()}")
        self.logger.log("-------------------------------------------------------")

        self.aggregate_clusterwise()

    @torch.no_grad()
    def compute_pairwise_similarity(self):
        self.similarity_matrix = np.eye(len(self.train_clients))
        for i, diff_a in enumerate(self.clients_model_params_diff):
            for j, diff_b in enumerate(self.clients_model_params_diff[i + 1 :], i + 1):
                if diff_a is not None and diff_b is not None:

                    score = torch.cosine_similarity(
                        vectorizeAndDct(diff_a), vectorizeAndDct(diff_b), dim=0, eps=1e-12
                    ).item()
                    self.similarity_matrix[i, j] = score
                    self.similarity_matrix[j, i] = score

    def cluster_clients(self, similarities):
        # clustering = AgglomerativeClustering(
        #     metric="precomputed", linkage="complete"
        # ).fit(-similarities)
        #
        # cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
        # cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
        # return cluster_1, cluster_2
        clustering = AffinityPropagation(affinity='precomputed',random_state=42).fit(similarities)
        # 获取聚类标签
        labels = clustering.labels_
        clusters_num = set(labels)
        clusters = []
        for i in clusters_num:
            cluster_i = np.argwhere(labels == i).flatten()
            clusters.append(cluster_i)

        return clusters


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
                    if "classifier" in key:#全局聚合不更新头部
                        # print(key)
                        continue
                    self.clients_personal_model_params[i][key].data += diff

        self.clients_model_params_diff = [None for _ in self.train_clients]

    def run_experiment(self):
        """The entrypoint of FL-bench experiment."""
        self.logger.log("=" * 20, self.algorithm_name, "=" * 20)
        self.logger.log("Experiment Arguments:")
        rich_pprint(
            OmegaConf.to_object(self.args), console=self.logger.stdout, expand_all=True
        )
        if self.args.common.save_log:
            rich_pprint(
                OmegaConf.to_object(self.args),
                console=self.logger.logfile_logger,
                expand_all=True,
            )
        if self.args.common.monitor == "tensorboard":
            self.tensorboard.add_text(
                f"ExperimentalArguments-{self.monitor_window_name_suffix}",
                f"{json.dumps(OmegaConf.to_object(self.args), indent=4)}",
            )

        begin = time.time()
        try:
            self.train()
        except KeyboardInterrupt:
            # when user manually terminates the run, FL-bench
            # indicates that run should be considered as useless and deleted.
            self.logger.close()
            del self.train_progress_bar
            if self.args.common.delete_useless_run:
                if os.path.isdir(self.output_dir):
                    shutil.rmtree(self.output_dir)
                return
        except Exception as e:
            self.logger.log(traceback.format_exc())
            self.logger.log(f"Exception occurred: {e}")
            self.logger.close()
            del self.train_progress_bar
            raise

        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algorithm_name}'s total running time: "
            f"{int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algorithm_name, "Experiment Results:", "=" * 20)
        self.logger.log(
            "[green]Display format: (before local fine-tuning) -> (after local fine-tuning)\n",
            "So if finetune_epoch = 0, x.xx% -> 0.00% is normal.\n",
            "Centralized testing ONLY happens after model aggregation, so the stats between '->' are the same.",
        )
        self.logger.log("=" * 20, "split-round:", self.split_round, "=" * 20)
        all_test_results = {
            epoch: {
                group: {
                    split: {
                        "loss": f"[red]{metrics['before'][split].loss:.4f} -> "
                                f"{metrics['after'][split].loss:.4f}[/red]",
                        "accuracy": f"[blue]{metrics['before'][split].accuracy:.2f}% -> "
                                    f"{metrics['after'][split].accuracy:.2f}%[/blue]",
                    }
                    for split, flag in [
                        (
                            "train",
                            self.args.common.test.client.train
                            or self.args.common.test.server.train,
                        ),
                        (
                            "val",
                            self.args.common.test.client.val
                            or self.args.common.test.server.val,
                        ),
                        (
                            "test",
                            self.args.common.test.client.test
                            or self.args.common.test.server.test,
                        ),
                    ]
                    if flag
                }
                for group, metrics in results.items()
            }
            for epoch, results in self.test_results.items()
        }

        self.logger.log(json.dumps(all_test_results, indent=4))
        self.logger.log(list_to_str(self.mean_norm,"model mean norm:  "))
        self.logger.log(list_to_str(self.max_norm,"model max norm:  "))
        self.logger.log(list_to_str(self.split_round_list,"split-round: "))


        if self.args.common.monitor == "tensorboard":
            for epoch, results in all_test_results.items():
                self.tensorboard.add_text(
                    f"Results-{self.monitor_window_name_suffix}",
                    text_string=f"<pre>{results}</pre>",
                    global_step=epoch,
                )

        self.show_max_metrics()

        self.logger.close()

        # plot the training curves
        if self.args.common.save_learning_curve_plot:
            self.save_learning_curve_plot()

        # save each round's metrics stats
        if self.args.common.save_metrics:
            self.save_metrics_stats()

        # save trained model(s) parameters
        if self.args.common.save_model:
            self.save_model_weights()

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