from pathlib import Path
from tinygrad import Tensor, TinyJit, nn
from tinygrad.device import Device
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from export_model import export_model
import glob, os

class Model:
  def __init__(self):
    self.layers = [
      lambda x: x.flatten(1),
      nn.Linear(784, 512), Tensor.silu,
      nn.Linear(512, 512), Tensor.silu,
      nn.Linear(512, 10),
    ]
  def __call__(self, x): return x.sequential(self.layers)

if __name__ == "__main__":
  B, STEPS = 256, 250
  model_name = "mnist_mlp"
  dir_name = Path(__file__).parent / model_name
  dir_name.mkdir(exist_ok=True)

  X_train, Y_train, X_test, Y_test = mnist()
  model = Model()
  opt = nn.optim.Muon(nn.state.get_parameters(model))

  @TinyJit
  def train_step():
    samples = Tensor.randint(B, high=X_train.shape[0])
    opt.zero_grad()
    x = X_train[samples] * 2 / 255 - 1
    loss = model(x).sparse_categorical_crossentropy(Y_train[samples])
    loss.backward()
    opt.step()

  best_acc, best_file = 0, None
  for i in (t := trange(STEPS)):
    train_step()
    if i % 10 == 9:
      acc = (model(X_test * 2 / 255 - 1).argmax(1) == Y_test).mean().item() * 100
      if acc > best_acc:
        best_acc = acc
        best_file = dir_name / f"{model_name}_{int(acc)}.safetensors"
        safe_save(get_state_dict(model), best_file)
    t.set_description(f"acc: {best_acc:.1f}%")

  # EXPORT WEBGPU (CPU safe)
  Device.DEFAULT = "CPU"
  model = Model()
  load_state_dict(model, safe_load(best_file))
  input_tensor = Tensor.randn(1, 1, 28, 28).flatten(1) * 2 / 255 - 1  # MLP input
  prg, _, _, state = export_model(model, "webgpu", input_tensor, model_name=model_name)
  with open(dir_name / f"{model_name}.js", "w") as f: f.write(prg)
  safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")

  # CLEAN SPAM
  for f in glob.glob(str(dir_name / "*_*.safetensors")):
    os.remove(f)

  print(f"✅ MLP {best_acc:.1f}% → {model_name}.js ({len(prg)//1000} KB) READY!")