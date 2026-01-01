import matplotlib.pyplot as plt
from transunet.vis.vis_prediction import visualize_brats_prediction


class SegmentationResult:
    def __init__(self, subject, source_path, raw_mask, metadata):
        self.subject = subject,
        self.source_path = source_path
        self.raw_mask = raw_mask
        self.metadata = metadata

    def save(self, path):
        self.filepath = path
        figure = visualize_brats_prediction(
            pred=self.raw_mask, return_fig=True)
        figure.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(figure)
