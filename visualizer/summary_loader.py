from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class SummaryLoader:
    def __init__(self):
        pass

    def load_data(self, path: str):
        # Create an EventAccumulator object
        event_acc = EventAccumulator(path)
        event_acc.Reload()  # Load events from the summary directory

        scalar_data = {"Loss/Train": [], "Loss/Val": []}
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            for event in events:
                scalar_data[tag].append(event.value)
                # scalar_data.append(
                #    {"Epoch": event.step, "Tag": tag, "Value": event.value}
                # )
        return scalar_data
