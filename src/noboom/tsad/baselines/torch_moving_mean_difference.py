import torch


class MovingMeanDifferenceAnomalyDetector:
    """Compute an anomaly score based on the moving mean and standard deviation.

    :param window_size: The size of the context for computing mean and standard deviation.
    :param use_mean: Whether to use the mean or just rely on standard deviation.
    :param std_coefficient: Weight of the standard deviation.
    :param offset: Threshold applied to anomaly score.
    :param order: If order is greater than 0, consider the difference between the current and the past order elements.
    :param two_sided: Whether to take the absolute value.
    """

    def __init__(
            self,
            window_size: int,
            use_mean: bool,
            std_coefficient: float,
            offset: float = 0.0,
            order: int = 0,
            two_sided: bool = False
    ):

        super().__init__()

        self.window_size = window_size
        self.m = int(use_mean)
        self.s = std_coefficient
        self.b = offset
        self.order = order
        self.two_sided = two_sided

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.score(inputs) > self.b).int()

    def score(self, inputs: torch.Tensor) -> torch.Tensor:

        if self.order:
            inputs = inputs[self.order:] - inputs[:-self.order]

        if self.two_sided:
            inputs = torch.abs(inputs)

        score = self.m * torch.mean(inputs.unfold(0, self.window_size, 1), -1)
        score += self.s * torch.std(inputs.unfold(0, self.window_size, 1), -1)

        return inputs[-len(score):] - score