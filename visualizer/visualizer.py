import os

import matplotlib.pyplot as plt
from summary_loader import SummaryLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class Visualizer:
    def __init__(self, root_folder: str):
        self.summary_loader = SummaryLoader()
        self.root_folder = root_folder
        pass

    def plot_model(self, model_name: str):
        path = os.path.join(self.root_folder, model_name)
        model_data = self.summary_loader.load_data(path)

        fig, ax = self.__setup_figure()
        x = range(1, len(model_data["Loss/Train"]) + 1)
        ax.set_xticks(x)
        ax.set_xlabel("Epoch")
        ax.set_yticks([i / 20 for i in range(21)])
        ax.set_ylabel("Loss")
        ax.plot(x, model_data["Loss/Train"], color="blue", label="Loss/Train")
        ax.plot(x, model_data["Loss/Val"], color="orange", label="Loss/Val")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(path, model_name + ".png"))
        pass

    def plot_models(self, model_names: list[str]):
        model_data = {}
        for name, alt_name in model_names:
            path = os.path.join(self.root_folder, name)
            data = self.summary_loader.load_data(path)
            model_data[alt_name] = data

        data_type = ["Train", "Val"]

        for mod in data_type:
            fig, ax = self.__setup_figure()
            ax.set_yticks([i / 20 for i in range(21)])
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")
            colors = ["red", "blue", "green"]

            for i, key_model_data in enumerate(model_data):
                data = model_data[key_model_data]
                x = range(1, len(data["Loss/Train"]) + 1)
                # ax.set_xticks(x)
                ax.plot(
                    x,
                    data[f"Loss/{mod}"],
                    color=colors[i],
                    label=f"Loss/{mod}/{key_model_data}",
                )

            ax.legend()
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    self.root_folder,
                    "-".join(key for key in model_data) + f"_{mod}.png",
                )
            )
        pass

    def __setup_figure(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax: plt.Axes = ax
        ax.grid(True)
        return fig, ax


if __name__ == "__main__":
    visuer = Visualizer("./output_models")
    # visuer.plot_models(
    #     [
    #         ("25000_lr001_b8", "B8"),
    #         ("25000_lr001_b32", "Standard"),
    #         ("25000_lr001_b64", "B64"),
    #     ]
    # )

    # visuer.plot_models(
    #     [
    #         ("25000_lr0001_b32", "LR0.001"),
    #         ("25000_lr001_b32", "Standard"),
    #         ("25000_lr01_b32", "LR0.1"),
    #     ]
    # )
    # visuer.plot_models(
    #     [
    #         ("25000_lr001_b32_nocc", "Scaled"),
    #         ("25000_lr001_b32", "Standard"),
    #     ]
    # )
    visuer.plot_models(
        [
            ("25000_lr0001_b32_perf", "Final"),
        ]
    )
