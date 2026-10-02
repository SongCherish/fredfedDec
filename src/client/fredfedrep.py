from src.client.fedper import FedPerClient


class FredFedRepClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)

    def package(self):
        """Upload encoder residuals only.

        The classifier is personalized in FredFedRep.  In particular, its
        residual must not be part of the update used by the server for DCT or
        clustering.
        """
        client_package = super().package()
        if self.return_diff:
            client_package["model_params_diff"] = {
                name: diff
                for name, diff in client_package["model_params_diff"].items()
                if "classifier" not in name
            }
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for E in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                if E < self.local_epoch - self.args.fredfedrep.train_body_epoch:
                    self.model.base.zero_grad()
                else:
                    self.model.classifier.zero_grad()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def finetune(self):
        self.model.train()
        self.dataset.train()
        full_model = False
        if full_model:
            # fine-tune the full model
            for E in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # freeze body, train head
                    if (
                        E
                        < self.args.common.test.client.finetune_epoch
                        - self.args.fedrep.train_body_epoch
                    ):
                        self.model.base.zero_grad()
                    # freeze head, train body
                    else:
                        self.model.classifier.zero_grad()
                    self.optimizer.step()
        else:
            # fine-tune the classifier only
            for _ in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.model.base.zero_grad()
                    self.optimizer.step()
