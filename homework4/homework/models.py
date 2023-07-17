import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    heatmap = heatmap.to(device)
    heatmap_tensor = heatmap[None, None]
    pooled = F.max_pool2d(heatmap_tensor, kernel_size=max_pool_ks, stride=1, padding=max_pool_ks // 2)
    maxima = (heatmap == pooled) & (heatmap > min_score)
    if not maxima.any():
        return []
    # Find the coordinates of the local maxima
    maxima_2d = maxima.squeeze().nonzero(as_tuple=False)
    scores = heatmap[maxima_2d[:, 0], maxima_2d[:, 1]]

    # Sort the peaks by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]
    sorted_indices = maxima_2d[sorted_indices]

    # Limit the number of peaks to max_det
    sorted_scores = sorted_scores[:max_det]
    sorted_indices = sorted_indices[:max_det]

    # Convert indices to coordinates (cx, cy)
    sorted_xs = sorted_indices[:, 1]
    sorted_ys = sorted_indices[:, 0]

    # Create a list of peaks [(score, cx, cy)]
    peaks = [(score.item(), cx.item(), cy.item()) for score, cx, cy in zip(sorted_scores, sorted_xs, sorted_ys)]

    return peaks
    # raise NotImplementedError('extract_peak')


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        loss = -((1 - pt) ** self.gamma) * target * torch.log(pt) - (pt ** self.gamma) * (1 - target) * torch.log(
            1 - pt)
        return loss.mean()


def custom_regression_loss(predicted_size, gt_size):
    return F.smooth_l1_loss(predicted_size, gt_size)


def total_loss(heatmap_output, size, gt_heatmap, gt_size, w1=0.8, w2=0.2):
    # Calculate Focal Loss for heatmap prediction
    criterion_heatmap = FocalLoss()

    # Calculate regression loss for size prediction
    criterion_regression = torch.nn.MSELoss()

    # Calculate heatmap loss and size loss separately
    heatmap_loss = criterion_heatmap(heatmap_output, gt_heatmap)
    size_loss = criterion_regression(size, gt_size)

    # Calculate the total loss using the specified weights
    total_loss = w1 * heatmap_loss + w2 * size_loss

    return total_loss


# Example usage:
# Assuming you have the predicted_heatmap_output, predicted_size,
# ground_truth_heatmap, and ground_truth_size tensors

# Calculate the total loss using the custom loss function
# loss = total_loss(predicted_heatmap_output, predicted_size, ground_truth_heatmap, ground_truth_size, w1=1.0, w2=1.0)
class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
                # Create separate classifiers for each object class (kart, bomb, pickup)
        self.kart_classifier = torch.nn.Conv2d(c, 1, 1)
        self.bomb_classifier = torch.nn.Conv2d(c, 1, 1)
        self.pickup_classifier = torch.nn.Conv2d(c, 1, 1)
        self.size_predictor = torch.nn.Conv2d(c, 2, kernel_size=1)
                # self.pickup_classifier = torch.nn.Conv2d(c, n_output, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
            # return torch.sigmoid(self.classifier(z))
            # Apply separate classifiers for each object class (kart, bomb, pickup)
        kart_heatmap = torch.sigmoid(self.kart_classifier(z))
        bomb_heatmap = torch.sigmoid(self.bomb_classifier(z))
        pickup_heatmap = torch.sigmoid(self.pickup_classifier(z))

        # Reshape the output to (batch_size, 3, height, width)
        # heatmap_output = torch.stack([kart_heatmap.squeeze(), bomb_heatmap.squeeze(), pickup_heatmap.squeeze()],dim=1)
        heatmap_output = torch.cat([kart_heatmap, bomb_heatmap, pickup_heatmap], dim=1)
        # Calculate size from the heatmap
        size = self.size_predictor(z)
        # print (f'heatmap_output {heatmap_output.shape},size {size.shape}')
        return heatmap_output, size

    '''def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        img = image.to(device)
        print(f'image shape {img.shape}')
        self = self.to(device)
        with torch.no_grad():
            heatmap_output, size_output = self(img)
            print(f'img predecion size {heatmap_output.shape}')
            print(f'size predecion size {size_output.shape}')
        detections = [[], [], []]
        for i in range(3):
            channel_heatmap = heatmap_output[0, i, :, :]
            print(f'channel_heatmap  shape {channel_heatmap.shape}')
            channel_detections = extract_peak(channel_heatmap)
            print(f'going to loop over range')
            # Format the detections as (score, cx, cy, w/2, h/2)
            for score, cx, cy in channel_detections:
                # Set w=0, h=0 as object size is not predicted
                # Get the predicted size for the detected peak
              if 0 <= cx < size_output.size(3) and 0 <= cy < size_output.size(2):
                w, h = size_output[0, i, cy, cx].tolist()  # Convert tensor to Python float
              else:
                w, h = 0, 0  # If the peak is outside the size_output tensor, set w=0, h=0

            # Add the detection to the corresponding class list
              detections[i].append((score, cx, cy, w / 2, h / 2))
            # Limit the number of detections to 30 per image per class
              if len(detections[i]) >= 30:
                break

        return detections '''
    def detect(self, image):
      """
    Implement object detection here.

    @image: 3 x H x W image
    @return: Three lists of detections [(score, cx, cy, w/2, h/2), ...], one per class,
             return no more than 30 detections per image per class. You only need to predict width and height
             for extra credit. If you do not predict an object size, return w=0, h=0.

    Hint: Use extract_peak here
    Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
          scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
          out of memory.
    """
      img = image.to(device)
      dl = self.eval().to(device)
      with torch.no_grad():
        heatmap_output, size_output = dl(img)
        # print("size item shape" ,size_output.shape )

      detections = [[], [], []]
      for i in range(3):
        channel_heatmap = heatmap_output[0, i, :, :]
        channel_detections = extract_peak(channel_heatmap)

        # Format the detections as (score, cx, cy, w/2, h/2)
        # Format the detections as (score, cx, cy, w/2, h/2)
        for score, cx, cy in channel_detections:
            # Get the predicted size for the detected peak
            size = size_output.squeeze()  # Remove singleton dimensions if any
            if size.dim() == 0:
                # Handle the case when size_output is a scalar (Python float)
                w, h = 0.0, 0.0
            else:
                # Convert size_output[0, 0, cy, cx] and size_output[0, 1, cy, cx] to Python floats
                w = float(size[0, cy, cx]) if size.size(0) > 0 else 0.0
                h = float(size[1, cy, cx]) if size.size(0) > 1 else 0.0
            # Add the detection to the corresponding class list
            # print("w and h",w,h)
            detections[i].append((float(score), int(cx), int(cy), w / 2, h / 2))
            # Limit the number of detections to 30 per image per class
            if len(detections[i]) >= 30:
                break

      return detections


        # raise NotImplementedError('Detector.detect')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset

    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
