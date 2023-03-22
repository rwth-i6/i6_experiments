import torch
from torch import nn
from torch.onnx import export

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(50, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 12001),
    )

  def forward(self, x):
    y = self.linear_relu_stack(x)
    return y


def train_step(*, model: Model, data, train_ctx, **_kwargs):
  frames = data["data"]
  scripted_model = torch.jit.script(model)
  dummy_input = torch.randn(10, 3, 50, device="cuda")
  export(scripted_model, dummy_input, "model.onnx", verbose=True, input_names=["features"], output_names=["output"])
  assert False
  outputs = scripted_model(frames)
  print(scripted_model.graph)
  targets = data["classes"]
  loss = nn.CrossEntropyLoss(reduction="sum")(torch.swapaxes(outputs, 1, 2), targets)
  train_ctx.mark_as_loss(name="ce", loss=loss)
